"""Explicit, scoped presentation preferences. No conversation mining or scientific facts.

Memory Bank holds the values; the control store holds revision/hash/tombstone
metadata. A pending or deleted revision is never available to an agent.
"""
import asyncio
import hashlib
import json
import time
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from .control_store import mutate, Conflict, QuotaExceeded
from ..policies.run_scope import session_user

CHOICES = {'response_style': ('concise', 'detailed'),
           'display_format': ('prose', 'bullets', 'table')}
TTL_SECONDS = 30 * 86400


class PreferenceChange(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    request_id: str
    expected_revision: int = Field(ge=0)
    value: str


class PreferenceDelete(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    expected_revision: int = Field(ge=0)


def fact(key, value):
    if key not in CHOICES or value not in CHOICES[key]:
        raise ValueError('Unsupported presentation preference')
    return json.dumps({'type': 'explicit_preference_v1', 'key': key, 'value': value}, sort_keys=True)


class Preferences:
    def __init__(self, config, access, store, provider, clock=time.time, allowed_users=None):
        self.config, self.access, self.store, self.provider, self.clock = config, access, store, provider, clock
        self.allowed_users = None if allowed_users is None else frozenset(allowed_users)

    def enabled_for(self, principal):
        return self.allowed_users is None or principal.user_id in self.allowed_users

    def authorize(self, principal):
        self.access.authorize(principal, self.config.workspace, self.config.release_id)
        if not self.enabled_for(principal):raise PermissionError('Preference pilot is not enabled for this account')

    def identity(self, principal, key):
        self.authorize(principal)
        if key not in CHOICES: raise ValueError('Unknown preference')
        owner = session_user(self.config.workspace, principal.user_id)
        return owner, 'preferences/' + owner + '/' + key + '.json'

    async def _read(self, principal, key):
        owner, path = await asyncio.to_thread(self.identity, principal, key)
        record, _ = await asyncio.to_thread(self.store.read, path)
        return owner, path, record

    async def list(self, principal):
        items = []
        for key, choices in CHOICES.items():
            owner, path, record = await self._read(principal, key)
            item = {'key': key, 'choices': choices, 'revision': (record or {}).get('revision', 0),
                    'status': (record or {}).get('status', 'unset'), 'value': None}
            if record and record['status'] == 'active':
                if record['expires_at'] <= self.clock(): item['status'] = 'expired'
                else:
                    text = await self.provider.get(owner, record['memory_id'])
                    if hashlib.sha256(text.encode()).hexdigest() != record['sha256']:
                        raise RuntimeError('Memory integrity check failed')
                    parsed = json.loads(text)
                    if text != fact(key, parsed.get('value')): raise RuntimeError('Memory schema changed')
                    latest, _ = await asyncio.to_thread(self.store.read, path)
                    if latest['revision'] != record['revision'] or latest['status'] != 'active':
                        raise Conflict('Preference changed during retrieval')
                    item.update(value=parsed['value'], expires_at=record['expires_at'])
            items.append(item)
        await asyncio.to_thread(self.authorize,principal)
        return {'preferences': items, 'retention_days': 30, 'automatic_capture': False}

    async def save(self, principal, key, change):
        text = fact(key, change.value)
        UUID(change.request_id)
        owner, path = await asyncio.to_thread(self.identity, principal, key)
        digest = hashlib.sha256(text.encode()).hexdigest()
        memory_id = 'pref-' + hashlib.sha256((owner + key + change.request_id).encode()).hexdigest()[:40]
        now = self.clock()
        def reserve(old):
            old = old or {'revision': 0, 'writes': 0, 'memory_ids': []}
            if old.get('request_id') == change.request_id:
                if old.get('sha256') != digest: raise Conflict('Request already used with another value')
                if old['status'] == 'deleted': raise Conflict('This preference was forgotten; use a new request')
                return old
            if old['revision'] != change.expected_revision: raise Conflict('Preference changed; reload settings')
            if old['writes'] >= 50: raise QuotaExceeded('Preference pilot write limit reached')
            return {'revision': old['revision'] + 1, 'writes': old['writes'] + 1, 'status': 'pending',
                    'request_id': change.request_id, 'memory_id': memory_id, 'sha256': digest,
                    'memory_ids': list(dict.fromkeys(old['memory_ids'] + [memory_id])),
                    'expires_at': now + TTL_SECONDS}
        record = await asyncio.to_thread(mutate, self.store, path, reserve)
        if record['status'] != 'active':
            await self.provider.put(owner, memory_id, text, record['expires_at'])
            await asyncio.to_thread(self.authorize,principal)
            def activate(latest):
                if latest['request_id'] != change.request_id or latest['status'] == 'deleted':
                    raise Conflict('Preference superseded or forgotten')
                latest['status'] = 'active'
                return latest
            try: await asyncio.to_thread(mutate, self.store, path, activate)
            except Conflict:
                await self.provider.delete(owner, memory_id)
                raise
        # Superseded provider values are not retained as searchable personal memories.
        for obsolete in record['memory_ids']:
            if obsolete != memory_id: await self.provider.delete(owner, obsolete)
        return await self.list(principal)

    async def delete(self, principal, key, expected_revision):
        owner, path = await asyncio.to_thread(self.identity, principal, key)
        def tombstone(old):
            if not old: raise FileNotFoundError('No saved preference')
            if old['status'] == 'deleted' and expected_revision in {old['revision'], old['revision'] - 1}: return old
            if old['revision'] != expected_revision: raise Conflict('Preference changed; reload settings')
            old.update(status='deleted', revision=old['revision'] + 1)
            return old
        record = await asyncio.to_thread(mutate, self.store, path, tombstone)
        # Logical forgetting happens before remote deletion, and remains effective on timeout.
        for memory_id in record['memory_ids']: await self.provider.delete(owner, memory_id)
        await asyncio.to_thread(self.authorize,principal)
        return await self.list(principal)

    async def instruction(self, principal):
        snapshot = await self.list(principal)
        current = {x['key']: {'status': x['status'], 'value': x['value']}
                   for x in snapshot['preferences']}
        values = {key: item['value'] for key, item in current.items() if item['value']}
        summary = ('Active saved preferences: ' + '; '.join(f'{key}={value}' for key,value in values.items()) + '.'
                   if values else 'No saved presentation preferences are currently active.')
        style = {'concise': 'Keep answers concise.', 'detailed': 'Give detailed explanations.'}
        display = {'prose': 'Write the answer in paragraphs.',
                   'bullets': 'Write the answer as Markdown bullet points, including a short answer. Do not replace the bullets with a prose paragraph.',
                   'table': 'Use a Markdown table for tabular facts; put necessary limitations immediately below it.'}
        return '\n'.join([
            'Authoritative saved presentation state for this signed-in user, read from the protected preference service now:',
            json.dumps(current, sort_keys=True),
            summary,
            'You CAN read this state. Only discuss saved settings when the user asks about preferences; '
            'do not append a preference-status sentence to scientific answers. For status questions use the actual summary above, '
            'not conversation history. Null values are inactive. Deleted means forgotten; expired means expired; pending means not yet active. '
            'Do not claim memory access is unavailable when this verified state is supplied.',
            style.get(values.get('response_style'), ''), display.get(values.get('display_format'), ''),
            'Apply the selected presentation to the final answer, including answers received from another agent. '
            'The current explicit user formatting request takes precedence. Preserve all source citations, facts and required limitations. '
            'Presentation preferences never alter scientific evidence or access. This snapshot is read-only; '
            'saving, editing and forgetting require the Memory preferences menu. Never claim a chat message changed a preference.'
        ])
