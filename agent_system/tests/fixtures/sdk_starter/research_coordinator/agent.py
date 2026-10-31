"""Minimal coordinator for a loopback-only, remote-evidence A2A demo."""
import os

from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import (
    AGENT_CARD_WELL_KNOWN_PATH,
    RemoteA2aAgent,
)

remote_evidence = RemoteA2aAgent(
    name="evidence_service",
    description="Retrieve approved project facts and source references.",
    agent_card=f"http://127.0.0.1:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

root_agent = Agent(
    name="research_coordinator",
    model=os.environ.get("AGENT_MODEL", "gemini-3.1-flash-lite"),
    description="Read-only ASU Biodesign research copilot prototype.",
    instruction=(
        "For factual project questions, delegate to evidence_service. "
        "Preserve the retrieved fact IDs, source links, snapshot scope and limits. "
        "Do not answer project metrics from general knowledge. "
        "If asked to process uploads, train, approve a cohort or predict a dose, "
        "explain that this starter only retrieves recorded evidence and that "
        "the requested numerical workflow is not connected. Never imply work ran."
    ),
    sub_agents=[remote_evidence],
)
