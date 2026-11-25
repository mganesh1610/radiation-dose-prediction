FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /workspace
COPY app/requirements.txt /workspace/app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY app/ /workspace/app/
COPY scripts/ /workspace/scripts/
COPY agent_system/src/research_agents/__init__.py /workspace/agent_system/src/research_agents/__init__.py
COPY agent_system/src/research_agents/policies/__init__.py agent_system/src/research_agents/policies/demo_quota.py agent_system/src/research_agents/policies/demo_quota.json /workspace/agent_system/src/research_agents/policies/
COPY runtime/ /workspace/runtime/
RUN useradd --uid 10001 --create-home app
USER 10001
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "30", "app.server:app"]
