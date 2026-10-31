"""Short conversation labels stored with the managed session, not a transcript copy."""
import unicodedata

TITLE_KEY = 'conversation_title'
TITLE_LIMIT = 72


def normalize_title(value):
    if not isinstance(value, str):
        return None
    # Drop control/bidi formatting characters; render the remaining value as text.
    value = ''.join(' ' if char.isspace() else char for char in value
                    if char.isspace() or not unicodedata.category(char).startswith('C'))
    return ' '.join(value.split())[:TITLE_LIMIT] or None


def first_question_title(session):
    for event in session.events:
        if event.author != 'user' or event.partial or not event.content:
            continue
        parts = [part.text for part in event.content.parts or [] if part.text and not part.thought]
        title = normalize_title(' '.join(parts))
        if title:
            return title
    return None


def session_title(session):
    return normalize_title(session.state.get(TITLE_KEY)) or first_question_title(session)
