"""One coordinator and one remote proxy; no duplicate local Evidence agent."""
from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.genai import types
from .prompts import INSTRUCTIONS


def create_agent(model, evidence_url, client, guard, *, preferences=None, protocol_enabled=False, analysis_enabled=False,prediction_enabled=False,training_enabled=False):
    remote=RemoteA2aAgent(name="evidence_service",description="Verified scientific evidence service",
        agent_card=evidence_url.rstrip("/")+AGENT_CARD_WELL_KNOWN_PATH,
        httpx_client=client,timeout=90)
    async def instruction(ctx):
        text=INSTRUCTIONS
        if training_enabled:
            text+='\nA bounded Training workflow is enabled in the Assistant. For a new engineering job, direct users to Training to choose median_null, elastic_net, pls or gradient_boosting and one core holdout; the authenticated approval screen shows the exact configuration. You cannot approve, submit or cancel jobs. For status or results, delegate the explicit web_ job ID to evidence_service. Preserve current status and engineering-only limitations. Do not claim completion from a queued request or heartbeat, do not promise immediate cancellation of local computation, do not train A-D, retune frozen models, reuse final tests or select/deploy a winner.'
        else:text+='\nTraining commands are not connected.'
        if prediction_enabled:
            text+='\nFor prediction results, estimate_outside_training_dose_range describes only the numeric estimate, never the input dose or input distribution. Negative estimates do not mean physically negative absorbed doses. Preserve the training-overlap warning explicitly.'
            text+='\nFrozen prediction for owned uploaded profiles is connected. Delegate requests with an explicit upload ID and model to evidence_service. For QC/compatibility only, delegate the upload ID without a model. Users attach files or select real examples through Add expression profile. Raw arrays do not belong in chat. Require an explicit model selection; never select the apparent best model yourself. Existing uploads need compatibility approval for the exact input and frozen release. You cannot grant that approval. Distinguish recorded doses from model estimates, preserve training overlap, QC, provenance and limitations; no individual prediction interval is available. The UI renders the verified result and export from the receipt. Do not invent a download URL or claim an upload, approval or prediction succeeded without tool evidence.'
        else:text+='\nPrediction and upload integration are not connected.'
        if analysis_enabled:
            text += '\nA bounded expression-by-recorded-dose plotting workflow is enabled. Delegate plot/chart requests with one explicit GSE accession and numeric NCBI Gene ID to evidence_service. For whole-study plots the backend resolves GSM IDs: do not request a GSM list. This plotting tool supports no transformations, filters, pooled studies, prediction or inferential statistics. Preserve profile/donor counts, missingness and source-scale limitations. Say the plot is available only after the tool succeeds; the UI displays the verified saved artifact. Never invent download URLs.'
        if protocol_enabled:
            text += '\nProtocol source_scope=original_study applies to ALL GEO passages, including Sample_data_processing/software. Never call these our methods. Cite the original method; these excerpts do not establish current-project processing, even with valid passage IDs.'
            text += '\nProtocol retrieval is enabled for public GEO metadata. Delegate study-method questions to evidence_service, including questions asking you to correct, assume or guess a questionable source unit. A refusal to guess does not replace retrieval: first obtain the approved evidence and coverage status, then explain what can and cannot be verified. Require one explicit GSE accession, preserve passage citations and source limitations. If evidence_service reports a withheld section, explicitly preserve that section and its verification reason in your final answer. The hold is imposed by our application retrieval corpus; never claim GEO withheld or removed the source. Never invent a withheld section or a unit correction. Never use passages to count samples or infer sample doses.'
        else: text += '\nProtocol RAG is not connected; do not claim to search documents.'
        if preferences is not None:
            scope=await guard.authorize()
            if preferences.enabled_for(scope.principal):
                text += '\nExplicit presentation preferences: ' + await preferences.instruction(scope.principal)
                text += '\nUsers manage saved preferences through the Memory preferences menu. You cannot save or delete memories by saying you did.'
        else: text += '\nLong-term memory is not connected.'
        return text
    return Agent(name="research_coordinator",model=model,instruction=instruction,
        sub_agents=[remote],before_model_callback=guard.before_model,after_model_callback=guard.after_model,
        generate_content_config=types.GenerateContentConfig(temperature=0,max_output_tokens=1500,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
