# Research copilot agent runtime

The authenticated research workspace uses a Google ADK Research Coordinator and one
remote Evidence Specialist over A2A. The specialist calls controlled scientific
tools; the language model does not execute arbitrary SQL or calculate a new dose.
The website also retains a separate public structured-answer viewer.

## Supported workflows

- Authorized sample, recorded-dose and profile-count queries for an approved release.
- Bounded expression retrieval with identity, release, representation and mapping checks.
- Source-backed expression plots and private CSV/plot receipts.
- Study-filtered protocol retrieval with passage identifiers and source citations.
- Explicit preference save, edit, expiry and forget through managed memory.
- Owned uploaded-profile QC and explicitly selected frozen-model prediction.
- Read-only status/results for approved bounded Mac training jobs.

The application is a restricted research pilot. Results are research estimates;
recorded laboratory doses and model predictions remain distinct. The scientific
final test has already been consumed and must not be used for tuning.

## Separate environment

Use Python 3.12. Keep the training environment separate.

```sh
python3.12 -m venv .venv-agents
source .venv-agents/bin/activate
python -m pip install -e 'agent_system[dev]'
python -m pip check
python -m pytest agent_system/tests
```

Platform-specific lock files retain the tested dependency versions. The suite
blocks external network access and uses small numerical fixtures and loopback A2A.
Passing it verifies software behavior, not scientific accuracy or deployed capacity.
The SDK starter used by preflight is retained under `tests/fixtures/sdk_starter`;
it is an offline test fixture, not the authenticated production server.

## Runtime entry points

- Coordinator: `research_agents.services.fast_status:coordinator_app`.
- Evidence: `research_agents.services.bootstrap:evidence_app`.
- Website: `app/server.py`, using the protected copilot proxy.

Deployment requires an operator-provided `COPILOT_CONFIG`, protected membership
policy, reviewed release admission, approved object/warehouse resources and
managed Sessions/Memory resources. These files, private memberships and credentials
are deliberately not in the repository. Do not put real configuration in Git.

The configuration uses pinned project/location/resource identifiers, service
identity and IAP audience. Scientific admission pins the release, object generation,
checksums and approved sample/gene mappings. Startup fails closed when these inputs
are missing or inconsistent. A fresh clone is not a provisioned scientific release.

`COPILOT_ROLE` selects Coordinator or Evidence in the container. The optional
operator-only loopback probe additionally requires `COPILOT_OPERATOR_EMAIL`, which
must already be authorized by the protected policy; it does not test browser IAP.
The deployment Dockerfile expects separately provisioned configuration during the
operator build. Do not copy personal project context into a build context.

## Reliability and execution boundaries

The lightweight status entry checks signed identity, current identity binding and
workspace/release grants for every request. It reports warming separately from
execution readiness. Heavy SDK loading is deferred to the first data route.
Managed Sessions retains owned history; generation-conditional object-store
records provide admission, idempotency, cancellation and lease fencing. Expired
claimed turns are interrupted instead of automatically re-executed.

The current interface uses bounded request/response and manual recovery. Streaming,
generated user profiles, selected past-work memory, additional specialist services
and broad inference capacity are separate extensions. No Kafka, Redis or additional
load balancer is required by this implementation.

The cloud recovery fixture is opt-in and only runs in a disposable Cloud Run Job.
It uses the real control/session adapters with a deterministic executor and an
isolated namespace; it adds no crash endpoint to the application. It does not
simulate a real provider outage or establish full HTTP/IAP/A2A failure coverage.
