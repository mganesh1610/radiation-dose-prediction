"""Protected policy configuration; user emails never enter agent prompts."""
import json
from pathlib import Path
from dataclasses import dataclass
from .access import Grant


class PolicyFile:
    """Mounted protected config, re-read to honor revocation without cached grants."""
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        if self.path.stat().st_size > 2_000_000: raise ValueError("Policy too large")
        p = json.loads(self.path.read_text())
        if p.get("schema_version") != 1 or not p.get("revision"):
            raise ValueError("Invalid protected policy")
        return p

    def emails(self):
        return [m["email"] for m in self.read()["members"] if m["enabled"]]

    def grant(self, user, workspace):
        from .access import user_key, IAP_ISSUER
        p = self.read()
        if p["workspace"] != workspace: return None
        for m in p["members"]:
            if m["enabled"] and m.get("subject") and user_key(IAP_ISSUER,m["subject"])==user:
                return Grant(user,workspace,frozenset(p["releases"]),p["revision"])
        return None


class Memberships:
    def __init__(self, policy, store):
        self.policy,self.store=policy,store

    def bind(self, principal, verified_email):
        from ..persistence.control_store import mutate
        if verified_email not in self.policy.emails():raise PermissionError("Membership revoked")
        key="identities/"+principal.user_id+".json"
        old,_=self.store.read(key)
        value={"user_id":principal.user_id,"email":verified_email}
        def binding(record):
            return {k:v for k,v in (record or {}).items() if k != '_control_ops'}
        if binding(old)==value:return
        def update(old):
            if old and binding(old)!=binding(value):raise PermissionError("Identity binding changed; review required")
            return value
        # A binding is an immutable, idempotent value. Keep its original shape
        # compatible with the previous revision's exact binding comparison.
        mutate(self.store,key,update,record_operation=False)

    def grant(self, user, workspace):
        p=self.policy.read()
        if p["workspace"]!=workspace:return None
        binding,_=self.store.read("identities/"+user+".json")
        if not binding or binding.get("user_id")!=user:return None
        if binding.get("email") not in self.policy.emails():return None
        return Grant(user,workspace,frozenset(p["releases"]),p["revision"])


@dataclass(frozen=True)
class ChatConfig:
    workspace: str
    release_id: str
    app_name: str
    origin: str
    evidence_url: str
    model: str = "gemini-3.1-flash-lite"
    max_runs: int = 20
    max_chats: int = 20

    def __post_init__(self):
        # Operator-controlled deployment configuration, never a model/user tool input.
        for value in (self.max_runs, self.max_chats):
            if type(value) is not int or not 1 <= value <= 280:
                raise ValueError("Pilot allowances must be integers between 1 and 280")
