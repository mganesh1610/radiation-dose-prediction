"""Browser input never contains identity, sessions, policy or tool arguments."""
from pydantic import Field
from .queries import StrictRecord


class CreateChat(StrictRecord):
    request_id: str = Field(pattern=r"^[a-f0-9-]{32,36}$")


class SendMessage(CreateChat):
    text: str = Field(min_length=1, max_length=2000)


class EmptyBody(StrictRecord):
    pass
