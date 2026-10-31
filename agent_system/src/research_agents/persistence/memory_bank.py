"""Pinned ADK explicit-memory adapter, with scoped read/delete lifecycle checks."""
import datetime
from google.adk.memory import VertexAiMemoryBankService
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from google.genai.errors import APIError


class PreferenceMemoryBank(VertexAiMemoryBankService):
    def __init__(self, *, project, location, engine_id, app_name, credentials):
        super().__init__(project=project, location=location, agent_engine_id=engine_id, credentials=credentials)
        self.app_name = app_name + '_explicit_preferences_v1'
        self.parent = f'projects/{project}/locations/{location}/reasoningEngines/{engine_id}'

    def _build_api_client(self):
        import agentplatform
        return agentplatform.Client(project=self._project, location=self._location, credentials=self._credentials,
            http_options=types.HttpOptions(timeout=15000, retry_options=types.HttpRetryOptions(attempts=1))).aio

    def _name(self, memory_id):
        import re
        if not re.fullmatch(r'pref-[a-f0-9]{40}', memory_id): raise ValueError('Invalid preference ID')
        return self.parent + '/memories/' + memory_id

    async def _get(self, owner, memory_id):
        memory = await self._get_api_client().memory_banks.memories.get(name=self._name(memory_id))
        if memory.scope != {'app_name': self.app_name, 'user_id': owner}:
            raise PermissionError('Memory scope mismatch')
        return memory

    async def get(self, owner, memory_id):
        return (await self._get(owner, memory_id)).fact

    async def put(self, owner, memory_id, text, expires_at):
        try:
            existing = await self.get(owner, memory_id)
        except APIError as exc:
            if exc.code != 404: raise RuntimeError('Memory lookup unavailable') from None
        else:
            if existing != text: raise RuntimeError('Memory identity collision')
            return
        try:
            await self.add_memory(app_name=self.app_name, user_id=owner,
                memories=[MemoryEntry(id=memory_id, content=types.Content(parts=[types.Part(text=text)]))],
                custom_metadata={'expire_time': datetime.datetime.fromtimestamp(expires_at, datetime.timezone.utc),
                                 'disable_memory_revisions': True, 'wait_for_completion': True})
        except Exception:
            # A create may commit before the response is lost. Read the same ID;
            # never generate another memory or turn a 403 into success.
            if await self.get(owner, memory_id) != text: raise RuntimeError('Memory creation unverified') from None
        if await self.get(owner, memory_id) != text: raise RuntimeError('Memory creation unverified')

    async def delete(self, owner, memory_id):
        try: await self._get(owner, memory_id)
        except APIError as exc:
            if exc.code == 404: return
            raise RuntimeError('Memory deletion could not be verified') from None
        await self._get_api_client().memory_banks.memories.delete(name=self._name(memory_id))
        try: await self._get(owner, memory_id)
        except APIError as exc:
            if exc.code == 404: return
            raise RuntimeError('Memory deletion could not be verified') from None
        raise RuntimeError('Memory deletion is pending; retry forget')
