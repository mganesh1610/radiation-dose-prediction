"""Server-side identity and grants; never populate these from tool arguments."""
from dataclasses import dataclass
import hashlib
import time


IAP_ISSUER = "https://cloud.google.com/iap"


@dataclass(frozen=True)
class Principal:
    user_id: str
    expires_at: float


def user_key(issuer: str, subject: str) -> str:
    # Stable opaque mapping scoped to this application; email is not a primary key.
    return "usr_" + hashlib.sha256(("research-copilot-v1\0" + issuer + "\0" + subject).encode()).hexdigest()


class IapAuthenticator:
    """Same signed-IAP boundary as the existing app, without changing that app.

    The deployment adapter will supply protected allowlist/membership providers.
    Verifier injection is for offline tests only, never an HTTP option.
    """
    def __init__(self, audience, allowed_emails, *, verifier=None, clock=time.time, on_authenticated=None,
            assertion_header='X-Goog-IAP-JWT-Assertion'):
        if not audience or not audience.startswith("/projects/"):
            raise ValueError("A pinned IAP audience is required")
        self.audience, self.allowed_emails = audience, allowed_emails
        self.clock = clock
        self.on_authenticated = on_authenticated
        self.assertion_header = assertion_header
        if verifier is None:
            from google.auth.transport.requests import Request
            from google.oauth2.id_token import verify_token
            transport = Request()
            verifier = lambda token, audience: verify_token(token, transport, audience=audience,
                         certs_url="https://www.gstatic.com/iap/verify/public_key")
        self.verifier = verifier

    def authenticate(self, headers):
        token = headers.get(self.assertion_header, "")
        if not isinstance(token, str) or not token or len(token) > 16_384:
            import logging
            logging.getLogger(__name__).warning('Identity check failed at assertion presence')
            raise PermissionError("Verified Google sign-in required")
        stage='signature'
        try:
            claims = self.verifier(token, self.audience)
            stage='claims'
            now = self.clock()
            if (claims.get("iss") != IAP_ISSUER or claims.get("aud") != self.audience
                    or not isinstance(claims.get("sub"), str) or not claims["sub"]
                    or type(claims.get("exp")) not in (int, float) or claims["exp"] <= now
                    or type(claims.get("iat")) not in (int, float) or claims["iat"] > now + 30):
                raise ValueError()
            email = claims.get("email", "").lower()
            if email not in {e.strip().lower() for e in self.allowed_emails()}:
                raise ValueError()
            principal = Principal(user_key(claims["iss"], claims["sub"]), float(claims["exp"]))
            if self.on_authenticated:
                stage='membership_binding'
                self.on_authenticated(principal, email)
            return principal
        except Exception as error:
            import logging
            logging.getLogger(__name__).warning('Identity check failed at %s (%s)',stage,type(error).__name__)
            raise PermissionError("Sign-in assertion is invalid or not authorized") from None


@dataclass(frozen=True)
class Grant:
    user_id: str
    workspace_id: str
    releases: frozenset[str]
    revision: str


class AccessPolicy:
    """Look up current grants on every check so revocation is not cached."""
    def __init__(self, grant_lookup, clock=time.time):
        self.lookup, self.clock = grant_lookup, clock

    def authorize(self, principal, workspace_id, release_id):
        if not isinstance(principal, Principal) or principal.expires_at <= self.clock():
            raise PermissionError("Current verified identity required")
        grant = self.lookup(principal.user_id, workspace_id)
        if (not isinstance(grant, Grant) or grant.user_id != principal.user_id
                or grant.workspace_id != workspace_id or release_id not in grant.releases):
            raise PermissionError("Workspace/release access denied")
        return grant.revision
