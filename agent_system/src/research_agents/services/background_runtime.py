"""One bounded SDK initialization; control requests do not wait for it.

This prepares services only. It never admits a run, dispatches an agent, retries
an operation or promises progress while Cloud Run has throttled idle CPU.
Actual Session/model requests wait for full preparation on their original RPC.
"""
import asyncio
import time
from .startup_timing import phase


class BackgroundRuntime:
    def __init__(self, loader, *, expected_config=None, wait_seconds=90):
        self.loader = loader
        self.expected_config = expected_config
        self.wait_seconds = wait_seconds
        self.task = None
        self.components = self.service = None
        self.failed = self.closing = False

    def start(self):
        if self.closing:raise RuntimeError('Research services are closing')
        if self.task is None:
            self.task = asyncio.create_task(self._initialize())

    async def _initialize(self):
        started = time.perf_counter()
        try:
            components = await asyncio.to_thread(self.loader)
            if self.expected_config is not None and components.config != self.expected_config:
                raise ValueError('Protected runtime configuration changed')
            await components.sessions.prepare()
            if self.closing:return
            self.components = components
            self.service = components.chat_service()
            phase('execution_ready', started)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failed = True
            phase('execution_failed', started)  # No settings, identities or exception payload.

    def readiness(self):
        return {'execution_ready': self.service is not None,
                'startup_state': 'ready' if self.service is not None else 'failed' if self.failed else 'warming'}

    async def wait(self):
        self.start()
        try:
            await asyncio.wait_for(asyncio.shield(self.task), self.wait_seconds)
        except TimeoutError:
            raise RuntimeError('Research services are still starting; retry the same request') from None
        if self.service is None:raise RuntimeError('Research services did not initialize')
        return self.service

    async def close(self):
        self.closing = True
        if self.task is not None and not self.task.done():
            try:await asyncio.wait_for(asyncio.shield(self.task), 20)
            except TimeoutError:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        if self.service is not None:
            await self.service.executor.aclose()


class DeferredSessions:
    def __init__(self, runtime):self.runtime = runtime

    async def prepare(self):self.runtime.start()

    async def wait_ready(self):await self.runtime.wait()

    async def get_session(self, **kwargs):return await (await self.runtime.wait()).sessions.get_session(**kwargs)

    async def create_session(self, **kwargs):return await (await self.runtime.wait()).sessions.create_session(**kwargs)

    async def list_sessions(self, **kwargs):return await (await self.runtime.wait()).sessions.list_sessions(**kwargs)

    async def append_event(self, session, event):return await (await self.runtime.wait()).sessions.append_event(session, event)

    async def delete_session(self, **kwargs):return await (await self.runtime.wait()).sessions.delete_session(**kwargs)


class DeferredExecutor:
    def __init__(self, runtime):self.runtime = runtime

    def readiness(self):return self.runtime.readiness()

    async def __call__(self, scope, session):return await (await self.runtime.wait()).executor(scope, session)

    async def aclose(self):await self.runtime.close()


class DeferredPreferences:
    def __init__(self, runtime, allowed_users):self.runtime, self.allowed_users = runtime, frozenset(allowed_users)

    def enabled_for(self, principal):return principal.user_id in self.allowed_users

    async def provider(self):
        await self.runtime.wait()
        provider = self.runtime.components.preferences
        if provider is None:raise RuntimeError('Preference memory is not enabled')
        return provider

    async def list(self, principal):return await (await self.provider()).list(principal)

    async def save(self, principal, key, change):return await (await self.provider()).save(principal, key, change)

    async def delete(self, principal, key, revision):return await (await self.provider()).delete(principal, key, revision)


class DeferredArtifacts:
    def __init__(self, runtime):self.runtime = runtime

    async def get(self, service, principal, run_id, artifact_id):
        actual = await self.runtime.wait()
        return await self.runtime.components.artifacts.get(actual, principal, run_id, artifact_id)
