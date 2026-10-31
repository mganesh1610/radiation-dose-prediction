"""Real ADK/A2A regression: old terminal turns must not become new tasks.

Synthetic fixtures reproduce an unanswered protocol turn before a count query,
and a near-ten-turn chat whose stored history exceeds the outbound context cap.
No cloud/model provider is used.
"""
import asyncio
import json
import socket
import time
from types import SimpleNamespace
from typing import Any

import pytest
import uvicorn
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from research_agents.agents.evidence_specialist.agent import create_agent
from research_agents.persistence.current_turn_sessions import CurrentTurnSessions
from research_agents.policies.access import Principal
from research_agents.policies.agent_guard import AgentGuard
from research_agents.policies.model_context import request_bytes
from research_agents.policies.runtime import ChatConfig
from research_agents.services.coordinator_runner import CoordinatorExecutor
from research_agents.services.evidence_a2a import ServiceAuthentication, create_app
from research_agents.tools.chat_tools import ChatTools
from test_chat import key, setup


class CountModel(BaseLlm):
    model: str = "scripted"
    coordinator: bool = False
    seen: Any

    async def generate_content_async(self, llm_request, stream=False):
        self.seen.append({"coordinator": self.coordinator,
                          "bytes": request_bytes(llm_request),
                          "contents": llm_request.model_dump(mode="json")["contents"]})
        responses = [p.function_response for c in llm_request.contents for p in c.parts or [] if p.function_response]
        if self.coordinator:
            part = types.Part(function_call=types.FunctionCall(name="transfer_to_agent", args={"agent_name": "evidence_service"}))
        elif not responses:
            part = types.Part(function_call=types.FunctionCall(name="query_core_records", args={"metric": "sample_count", "study_id": "GSE123"}))
        else:
            part = types.Part(text="25 profiles. " + responses[-1].response["evidence_id"])
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.mark.parametrize("old_turns", [1, 9])
def test_terminal_history_is_stored_but_not_delegated_or_sent_to_models(old_turns):
    async def run():
        _, chat, _, _ = setup()
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
        config = ChatConfig("workspace", "core_test", "copilot", "https://research.test", f"http://127.0.0.1:{port}")
        chat.config = config
        guard = AgentGuard(chat.ledger, chat.access, "core_test")
        seen, queries = [], []

        class Queries:
            def run(self, principal, workspace, plan):
                queries.append(plan)
                return {"answer": {"status": "ok", "count": 25, "unit": "profiles"}, "scope": plan,
                        "receipt": {"release_manifest_sha256": "a" * 64, "result_sha256": "b" * 64}}

        tools = ChatTools(Queries, None, guard, SimpleNamespace(features=(), core=SimpleNamespace(members=(("GSM1", "GSE123", "d1"),))))
        # Include the enabled schema/instruction footprint that exposed the live
        # failure. A count query must never call these unrelated tools.
        unused = SimpleNamespace()
        evidence = create_agent(CountModel(seen=seen), tools, guard, "core_test", "core_test_linear",
                                protocol_tool=unused, training_tool=unused, prediction_tool=unused, analysis_tool=unused)
        app = create_app(evidence, config, chat.sessions, chat.ledger, chat.ledger.store, chat.access,
                         ServiceAuthentication(config.evidence_url, "unused", local_token="offline-context-token"))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        serving = asyncio.create_task(server.serve())
        try:
            for _ in range(100):
                if server.started: break
                await asyncio.sleep(.01)
            principal = Principal("alice", time.time() + 600)
            created = await chat.create(principal, key())
            _, session = await chat.session(principal, created["chat_id"])
            old_run_ids = []
            for index in range(old_turns):
                text = "OLD_UNANSWERED_PROTOCOL: explain the previous RNA extraction. " + "old method detail " * (40 if old_turns > 1 else 1)
                old = chat.ledger.admit("alice", created["chat_id"], key(), text)
                old = chat.ledger.claim("alice", old["run_id"])
                old_run_ids.append(old["run_id"])
                await chat.sessions.append_event(session, Event(id=old["message_id"], author="user", invocation_id=old["run_id"],
                    content=types.Content(role="user", parts=[types.Part(text=text)])))
                await chat.sessions.append_event(session, Event(author="research_coordinator", invocation_id=old["run_id"],
                    content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="transfer_to_agent", args={"agent_name": "evidence_service"}))])))
                chat.ledger.finish("alice", old["run_id"], old["fence"], "cancelled" if index % 2 == 0 else "failed")
            before = chat.ledger.read()
            _, stored_before = await chat.session(principal, created["chat_id"])
            before_events = [e.model_dump_json() for e in stored_before.events]
            chat.executor = CoordinatorExecutor(config, chat.sessions, lambda: CountModel(coordinator=True, seen=seen), guard,
                lambda: "offline-context-token", protocol_enabled=True, training_enabled=True, prediction_enabled=True, analysis_enabled=True)
            question = "How many profiles belong to GSE123?"
            request_key = key()
            result = await chat.send(principal, created["chat_id"], request_key, question)
            assert result["status"] == "completed", chat.ledger.run("alice", result["run_id"])
            assert len(queries) == 1 and queries[0]["metric"] == "sample_count"
            assert len(seen) == 3 and max(r["bytes"] for r in seen) <= 16000
            assert all("OLD_UNANSWERED_PROTOCOL" not in json.dumps(r["contents"]) for r in seen)
            receipt = chat.ledger.run("alice", result["run_id"])
            assert receipt["model_calls"] == 3 and receipt["tool_calls"] == 1
            _, stored_after = await chat.session(principal, created["chat_id"])
            assert [e.model_dump_json() for e in stored_after.events[:len(before_events)]] == before_events
            assert all(chat.ledger.read()["runs"][r] == before["runs"][r] for r in old_run_ids)
            history = await chat.messages(principal, created["chat_id"])
            assert len([m for m in history["messages"] if "OLD_UNANSWERED_PROTOCOL" in m["text"]]) == old_turns
            assert "25 profiles" in history["messages"][-1]["text"]
            # Exact-key replay still performs no model/tool work.
            assert (await chat.send(principal, created["chat_id"], request_key, question))["run_id"] == result["run_id"]
            assert len(seen) == 3 and len(queries) == 1
        finally:
            server.should_exit = True
            await serving
    asyncio.run(run())


def test_execution_view_keeps_current_tool_pairs_and_provider_history_and_state():
    async def run():
        _, chat, _, _ = setup()
        service = chat.sessions
        session = await service.create_session(app_name="copilot", user_id="alice", session_id="test-session",
                                              state={"selected_release_id": "core_test"})
        events = [
            Event(author="user", invocation_id="old", content=types.Content(role="user", parts=[types.Part(text="Old question")])),
            Event(author="user", invocation_id="current", content=types.Content(role="user", parts=[types.Part(text="Current question")])),
            Event(author="research_coordinator", invocation_id="current", content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="query", id="fc-one", args={}))])),
            Event(author="research_coordinator", invocation_id="current", content=types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(name="query", id="fc-one", response={"status": "ok"}))])),
        ]
        for event in events:
            await service.append_event(session, event)
        view = CurrentTurnSessions(service, app_name="copilot", user_id="alice", session_id=session.id, invocation_id="current")
        args = {"app_name": "copilot", "user_id": "alice", "session_id": session.id}
        current = await view.get_session(**args)
        assert [e.model_dump_json() for e in current.events] == [e.model_dump_json() for e in events[1:]]
        current.events[0].content.parts[0].text = "Only mutate the private copy"
        current.state["selected_release_id"] = "private copy"
        original = await service.get_session(**args)
        assert len(original.events) == 4 and original.events[1].content.parts[0].text == "Current question"
        assert original.state["selected_release_id"] == "core_test"
        event = Event(author="research_coordinator", invocation_id="current", content=types.Content(role="model", parts=[types.Part(text="New answer")]))
        await view.append_event(current, event)
        preserved = await service.get_session(**args)
        assert len(preserved.events) == 5 and preserved.events[0].invocation_id == "old"
        assert preserved.events[-1].id == event.id
        with pytest.raises(PermissionError):
            await view.append_event(current, Event(author="user", invocation_id="other"))
        with pytest.raises(PermissionError):
            await view.get_session(**{**args, "user_id": "bob"})
        for operation in (view.create_session, view.list_sessions, view.delete_session):
            with pytest.raises(PermissionError):
                await operation(**args)
        await view.flush()
    asyncio.run(run())
