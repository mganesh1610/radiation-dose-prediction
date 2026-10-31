"""SDK-free authenticated controls, with explicitly reported execution readiness."""
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from google.cloud import storage
from ..persistence.credentials import cloud_credentials
from ..persistence.control_store import GcsControlStore
from ..persistence.ledger import Ledger
from ..policies.runtime import ChatConfig, Memberships, PolicyFile
from ..policies.access import AccessPolicy, IapAuthenticator
from .background_runtime import BackgroundRuntime, DeferredSessions, DeferredExecutor, DeferredPreferences, DeferredArtifacts
from .chat_service import ChatService
from .chat_api import create_app
from .startup_timing import phase
from .startup_settings import read_settings


def load_execution(path):
    # Executed in the initializer thread, never on the HTTP control loop.
    from .bootstrap import Components
    components = Components(path)
    async def warm_sdk():
        # Prime lazy provider modules away from the serving event loop. This
        # temporary loop/client is closed without any provider operation.
        try:await components.sessions.prepare()
        finally:
            client = components.sessions._get_api_client()
            await client.aclose()
    asyncio.run(warm_sdk())
    return components


def coordinator_app():
    started = time.perf_counter()
    path = Path(os.environ['COPILOT_CONFIG'])
    p,config,policy,users = read_settings(path)
    credentials = cloud_credentials()
    store = GcsControlStore(p['control_bucket'], 'copilot/v1/' + config.workspace + '/',
                            client=storage.Client(project=p['project'], credentials=credentials))
    ledger = Ledger(store, config.workspace, max_runs=config.max_runs, max_chats=config.max_chats)
    memberships = Memberships(policy, store)
    access = AccessPolicy(memberships.grant)
    auth = IapAuthenticator(p['iap_audience'], policy.emails, on_authenticated=memberships.bind,
                            assertion_header='X-Research-IAP-Assertion')
    runtime = BackgroundRuntime(lambda: load_execution(str(path)), expected_config=config)
    service = ChatService(config, ledger, DeferredSessions(runtime), access, DeferredExecutor(runtime))
    app = create_app(service, auth, DeferredPreferences(runtime, users) if p.get('memory_enabled') else None,
        p.get('protocol_rag_enabled', False), users,
        artifacts=DeferredArtifacts(runtime) if p.get('analysis_enabled') else None,
        prediction_enabled=p.get('prediction_enabled', False), training_enabled=p.get('training_enabled', False))
    phase('controls_factory', started)
    app.state.execution_readiness=runtime.readiness
    app.state.protected_chat_config=config
    return app
