FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /workspace
COPY agent_system/ /workspace/agent_system/
RUN python -m pip install --no-cache-dir /workspace/agent_system && python -m pip check && python -c "import research_agents.services.bootstrap"
# Immutable image sources permit bytecode validation without filesystem stats.
RUN python -m compileall -q -f --invalidation-mode unchecked-hash /usr/local/lib/python3.12 /workspace/agent_system
COPY config/ /workspace/config/
RUN useradd --uid 10002 --create-home copilot && chmod -R a+rX /workspace/config
USER 10002
ENV COPILOT_CONFIG=/workspace/config/config.json
CMD ["sh","-c","if [ \"$COPILOT_ROLE\" = coordinator ]; then target=research_agents.services.fast_status:coordinator_app; else target=research_agents.services.bootstrap:${COPILOT_ROLE}_app; fi; exec python -m uvicorn \"$target\" --factory --host 0.0.0.0 --port 8080 --timeout-graceful-shutdown 20"]
