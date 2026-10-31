"""Owned chat references, idempotency, leases and conservative run admission."""
import copy
import hashlib
import time
import uuid

from .control_store import Conflict, QuotaExceeded, mutate
from ..policies.request_contract import compile_contract

TERMINAL = frozenset({"completed", "failed", "interrupted", "cancelled"})


def new_id(prefix):
    return prefix + "-" + uuid.uuid4().hex


class Ledger:
    def __init__(self, store, workspace, *, clock=time.time, max_runs=20, max_chats=20):
        self.store, self.workspace, self.clock = store, workspace, clock
        self.max_runs, self.max_chats = max_runs, max_chats

    def read(self):
        return self.store.read("ledger.json")[0] or {"schema_version": 1, "chats": {}, "runs": {}}

    def change(self, operation):
        def update(state):
            state = state or {"schema_version": 1, "chats": {}, "runs": {}}
            operation(state)
            return state
        return mutate(self.store, "ledger.json", update)

    def chat(self, owner, chat_id, *, include_deleted=False):
        chat = self.read()["chats"].get(chat_id)
        if not chat or chat["owner"] != owner or (chat["deleted"] and not include_deleted):
            raise FileNotFoundError("Chat not found")
        if chat["expires_at"] <= self.clock() and not include_deleted:
            raise FileNotFoundError("Chat expired")
        return chat

    def create_chat(self, owner, request_key, release_id):
        fingerprint = hashlib.sha256((owner + "\0" + request_key).encode()).hexdigest()
        candidate = new_id("chat")
        def op(state):
            existing = next((c for c in state["chats"].values() if c["request_sha"] == fingerprint), None)
            if existing:
                if existing["release_id"] != release_id or existing["deleted"]:
                    raise Conflict("Create request already used")
                return
            if len(state["chats"]) >= self.max_chats:
                raise QuotaExceeded("Pilot chat allowance reached")
            state["chats"][candidate] = {"chat_id": candidate, "owner": owner, "workspace": self.workspace,
                "request_sha": fingerprint, "release_id": release_id, "session_id": new_id("session"),
                "created_at": self.clock(), "expires_at": self.clock()+30*86400,
                "deleted": False, "ready": False, "active_run": None}
        state = self.change(op)
        return next(c for c in state["chats"].values() if c["request_sha"] == fingerprint)

    def ready(self, owner, chat_id):
        self.chat(owner, chat_id)
        def op(state):
            chat = state["chats"][chat_id]
            if chat["deleted"]: raise Conflict("Chat deleted")
            chat["ready"] = True
        self.change(op)

    def list_chats(self, owner):
        return [c for c in self.read()["chats"].values() if c["owner"] == owner
                and not c["deleted"] and c["expires_at"] > self.clock()]

    def run(self, owner, run_id):
        run = self.read()["runs"].get(run_id)
        if not run or run["owner"] != owner:
            raise FileNotFoundError("Run not found")
        self.chat(owner, run["chat_id"])
        return run

    def admit(self, owner, chat_id, request_key, text, *, identity_expires_at=None):
        self.chat(owner, chat_id)
        request_sha = hashlib.sha256(text.encode()).hexdigest()
        key_sha = hashlib.sha256((owner+"\0"+chat_id+"\0"+request_key).encode()).hexdigest()
        candidate = new_id("run")
        def op(state):
            chat = state["chats"][chat_id]
            if chat["deleted"] or not chat["ready"]: raise Conflict("Chat is not ready")
            old = next((r for r in state["runs"].values() if r["key_sha"] == key_sha), None)
            if old:
                if old["request_sha"] != request_sha: raise Conflict("Idempotency key reused with different text")
                return
            active = state["runs"].get(chat["active_run"])
            if active and active["status"] not in TERMINAL:
                if active["deadline"] > self.clock(): raise Conflict("This chat already has an active turn")
                active["status"] = "interrupted"
                active["events"].append({"seq": len(active["events"])+1, "stage": "interrupted"})
            if len(state["runs"]) >= self.max_runs:
                raise QuotaExceeded("Pilot run allowance reached; review usage before increasing it")
            live = [r for r in state["runs"].values() if r["status"] not in TERMINAL and r["deadline"] > self.clock()]
            if len(live) >= 2: raise QuotaExceeded("Two research requests are already active")
            if sum(r["chat_id"] == chat_id for r in state["runs"].values()) >= 10:
                raise QuotaExceeded("Start a new chat after ten turns")
            state["runs"][candidate] = {"run_id":candidate,"message_id":new_id("message"),"owner":owner,
                "workspace":self.workspace,"chat_id":chat_id,"key_sha":key_sha,"request_sha":request_sha,
                "request_contract":compile_contract(text),
                "status":"admitted","created_at":self.clock(),"deadline":self.clock()+180,
                "identity_expires_at":identity_expires_at or self.clock()+180,
                "fence":uuid.uuid4().hex,"claimed":False,"cancel_requested":False,
                "model_calls":0,"tool_calls":0,"input_bytes":0,"output_tokens":0,"prompt_tokens":0,
                "reserved_model_usd":0.05,"events":[{"seq":1,"stage":"admitted"}],"receipts":[]}
            chat["active_run"] = candidate
        state = self.change(op)
        return next(r for r in state["runs"].values() if r["key_sha"] == key_sha)

    def claim(self, owner, run_id):
        self.run(owner, run_id)
        def op(state):
            r=state["runs"][run_id]
            if r["claimed"] or r["status"] in TERMINAL: raise Conflict("Run already dispatched; do not replay")
            if r["deadline"] <= self.clock(): raise Conflict("Run deadline expired")
            r["claimed"] = True; r["status"] = "running"
        return self.change(op)["runs"][run_id]

    def check(self, owner, run_id, fence):
        run=self.run(owner,run_id)
        if run["fence"] != fence or run["status"] in TERMINAL or run["deadline"] <= self.clock() or run["cancel_requested"]:
            raise Conflict("Run cancelled, expired or no longer active")
        return run

    def reserve(self, owner, run_id, fence, kind, *, input_bytes=0):
        self.check(owner,run_id,fence)
        def op(state):
            r=state["runs"][run_id]
            if r["fence"] != fence or r["status"] in TERMINAL or r["deadline"] <= self.clock() or r["cancel_requested"]:
                raise Conflict("Run is no longer active")
            field="model_calls" if kind=="model" else "tool_calls"
            if r[field] >= (6 if kind=="model" else 4): raise QuotaExceeded("Per-turn call limit reached")
            if kind=="model" and (input_bytes > 16000 or r["input_bytes"]+input_bytes > 96000):
                raise QuotaExceeded("Model context limit reached; ask a smaller question")
            r[field]+=1; r["input_bytes"]+=input_bytes
        return self.change(op)["runs"][run_id]

    def progress(self, owner, run_id, fence, stage, receipt=None):
        self.check(owner,run_id,fence)
        def op(state):
            r=state["runs"][run_id]
            if r["fence"] != fence or r["status"] in TERMINAL or r["cancel_requested"] or r["deadline"]<=self.clock():
                raise Conflict("Stale run update")
            if len(r["events"])>=32:raise QuotaExceeded("Run event limit reached")
            r["events"].append({"seq":len(r["events"])+1,"stage":stage})
            if receipt is not None:r["receipts"].append(copy.deepcopy(receipt))
        self.change(op)

    def usage(self, owner, run_id, fence, prompt_tokens, output_tokens):
        self.check(owner,run_id,fence)
        def op(state):
            r=state['runs'][run_id]
            if r['fence']!=fence or r['status'] in TERMINAL:raise Conflict('Stale usage update')
            r['prompt_tokens']+=max(0,int(prompt_tokens or 0))
            r['output_tokens']+=max(0,int(output_tokens or 0))
        self.change(op)

    def finish(self, owner, run_id, fence, status):
        if status not in TERMINAL:raise ValueError("Invalid terminal state")
        self.run(owner,run_id)
        def op(state):
            r=state["runs"][run_id]
            if r["fence"] != fence or r["status"] in TERMINAL:return
            r["status"]="cancelled" if r["cancel_requested"] else ("interrupted" if r["deadline"]<=self.clock() else status)
            r["events"].append({"seq":len(r["events"])+1,"stage":r["status"]})
            c=state["chats"][r["chat_id"]]
            if c["active_run"]==run_id:c["active_run"]=None
        self.change(op)

    def cancel(self, owner, run_id):
        self.run(owner,run_id)
        def op(state):
            r=state["runs"][run_id]
            if r["status"] not in TERMINAL:r["cancel_requested"]=True
        self.change(op)

    def mark_deleted(self, owner, chat_id):
        self.chat(owner,chat_id,include_deleted=True)
        def op(state):
            c=state["chats"][chat_id]
            active=state["runs"].get(c["active_run"])
            if active and active["status"] not in TERMINAL and active["deadline"]>self.clock():
                raise Conflict("Cancel or finish the active turn before deleting")
            c["deleted"]=True
        self.change(op)
