"""Regressions for two observed integrated request failures, without paid calls."""
import pytest
from research_agents.policies.request_contract import compile_contract,bind_plan

@pytest.mark.parametrize('metric',['sample_count','study_group_count'])
def test_explicit_two_units_stay_distinct(metric):
 c=compile_contract('How many profiles and study-local donor groups belong to GSE8917?')
 p={'metric':metric,'filters':{'study_id':'GSE8917','dose_gy':None}}
 bound,error=bind_plan('query',p,c,{'GSE8917'})
 assert error is None and bound['metric']==metric

@pytest.mark.parametrize('text',['How many profiles or study-local donor groups in GSE8917?',
 'How many profiles and donors in GSE8917?'])
def test_ambiguous_units_still_require_clarification(text):
 bound,error=bind_plan('query',{'metric':'sample_count','filters':{'study_id':'GSE8917'}},compile_contract(text),{'GSE8917'})
 assert bound is None and error['status']=='clarification_required'

def test_two_units_do_not_authorize_a_different_study():
 c=compile_contract('How many profiles and study-local donor groups in GSE8917?')
 assert bind_plan('query',{'metric':'sample_count','filters':{'study_id':'GSE102971'}},c,{'GSE8917','GSE102971'})[0] is None

@pytest.mark.parametrize('text',[
 'Use frozen gradient_boosting to calculate a research dose.',
 'Calculate the research dose using frozen gradient_boosting.',
 'Compute dose using frozen model gradient_boosting.'])
def test_explicit_frozen_preset_and_calculate_dose_are_bound(text):
 c=compile_contract(text);assert c['prediction_requested'] and c['prediction_models']==['gradient_boosting']

@pytest.mark.parametrize('text',[
 'Do not calculate dose using frozen gradient_boosting.',
 'Never compute dose using frozen model A.',
 'Do not predict using model A.'])
def test_prediction_negation_remains_binding(text):
 assert not compile_contract(text)['prediction_requested']

def test_no_model_inferred_and_multiple_models_are_preserved():
 assert compile_contract('Calculate the dose.')['prediction_models']==[]
 assert compile_contract('Predict using model A and frozen gradient_boosting.')['prediction_models']==['A','gradient_boosting']
