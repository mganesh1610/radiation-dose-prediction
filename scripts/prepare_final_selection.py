"""Freeze a local research plan using metadata only. Never fits or opens test data."""
import copy,hashlib,json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'context/final-selection/v2'
RELEASE=ROOT/'data/model_ready/development_raw_f6022cfaaca6d1d9'
MODELS=('A','B','C','D','elastic_net','pls','gradient_boosting')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    if (OUT/'protocol-freeze.json').exists():raise RuntimeError('Plan already frozen; create a new version for changes')
    meta=read(RELEASE/'metadata.json');release=read(RELEASE/'release.json')
    expected=next(a for a in release['files'] if a['name']=='metadata.json')
    assert sha(RELEASE/'metadata.json')==expected['sha256'] and (RELEASE/'metadata.json').stat().st_size==expected['bytes']
    records={a['profile_id']:a for a in meta['profiles']};assert len(records)==567
    core=sorted(s for s,a in records.items() if a['role']=='core_development');aux=sorted(set(records)-set(core))
    neural=read(ROOT/'context/model-abcd/v1/config.json');classical=read(ROOT/'context/classical-comparison/v1/config.json')
    candidates=copy.deepcopy(neural['candidates']|classical['candidates'])
    assert set(candidates)==set(MODELS) and len(core)==145 and len(aux)==422
    config=dict(version='final-selection-v2',status='criteria_frozen_not_trained',release=str(RELEASE.relative_to(ROOT)),release_manifest_sha256=sha(RELEASE/'release.json'),metadata_sha256=sha(RELEASE/'metadata.json'),candidates=candidates,models=list(MODELS),untuned_baseline='median_null',features=500,hidden=neural['hidden'],cpu_threads=4,selection_metric='pooled_out_of_fold_donor_balanced_mae_gy',tie_break='smallest_original_candidate_id',primary_seed=20261002,sensitivity_seeds=[20261003,20261004],candidate_fits=280,final_trained_refits=17,median_reference_artifacts=1,locked_test_included=False,locked_test_execution_authorized_in_this_step=False,cloud_execution_enabled=False,app_enforcement_implemented=False)
    folds=[]
    for k in range(5):
        held=set()
        for study in sorted({records[s]['study'] for s in core}):
            donors=sorted({records[s]['donor_group'] for s in core if records[s]['study']==study})
            held.update(donors[k::5])
        val=[s for s in core if records[s]['donor_group'] in held];train=sorted(set(core)-set(val))
        folds.append(dict(fold_id=f'selection-{k}',seed=20261002+k,training=train,validation=val,auxiliary_training=aux))
    splits=dict(core_profiles=core,auxiliary_profiles=aux,profiles=[{key:records[s][key] for key in ['profile_id','study','donor_group','role']} for s in core+aux],folds=folds,full_refit=dict(core_training=core,auxiliary_training=aux,primary_seed=20261002,sensitivity_seeds=[20261003,20261004]))
    jobs=[dict(job_id=f"{f['fold_id']}-{m}-{p['id']}",fold=f['fold_id'],model=m,candidate_id=p['id'],seed=f['seed'],auxiliary_used=m in ('B','C','D'),status='planned') for f in folds for m in MODELS for p in candidates[m]]
    final=[]
    for m in MODELS:
        for i,seed in enumerate([20261002,20261003,20261004] if m in ('A','B','C','D','gradient_boosting') else [20261002]):
            final.append(dict(model=m,seed=seed,seed_role='primary' if i==0 else 'sensitivity',candidate_id=None,candidate_source='resolve_from_completed_five_fold_selection',status='not_trained',required_payloads=['model','preprocessing','gene_ids','target_scaler','training_membership','fit_trace','code_runtime_hashes','raw_replay_checks']))
    final.append(dict(model='median_null',seed=None,seed_role='deterministic',candidate_id=None,status='not_computed',definition='ordinary_median_of_all_core_training_profile_doses',required_payloads=['constant_gy','training_membership','source_hashes']))
    scope=dict(version=1,status='finalized_contract_not_deployed',target=dict(organism='human',tissue='whole_blood',exposure=['ex_vivo_xray','ex_vivo_gamma','matched_control'],time='audited_approximately_24_hours',assay='reviewed_adapter_for_specific_frozen_model'),defaults=dict(new_upload='analysis_only_pending_compatibility_review',training='authenticated_authorized_bounded_job',clinical_use=False,automatic_model_promotion=False),numeric_prediction_requirements=['trusted_input_model_compatibility_record','frozen_model_and_preprocessing_hashes','verified_gene_units_transformation_provenance','reviewed_missingness_and_technical_flags','raw_replay_and_runtime_checks'],display_fields=['research_only','model_seed_version','input_provenance','training_overlap','preprocessing_and_qc','known_transfer_limitations'],no_automatic_numeric_prediction_for=['unknown_provenance','other_species','isolated_or_cultured_lymphocytes','in_vivo_exposure','unvalidated_timepoint','rna_seq_or_counts','unsupported_assay'],numeric_thresholds=dict(missingness=None,out_of_distribution=None,clinically_acceptable_error=None),interval_claim='no_calibrated_individual_prediction_interval',training_dose_coverage_gy=[0,8],coverage_is_validated_accuracy_range=False,cloud_app_changed=False)
    statistical=dict(version=1,comparison_pairs=[['A','B'],['A','C'],['A','D']],unit='donor_mean_absolute_error_gy',difference='A_minus_comparator',null='delta_greater_than_or_equal_zero',alternative='delta_less_than_zero',primary_seed=20261002,primary_test='paired_t_alternative_less',nan_policy='raise',multiplicity='holm',family_size=3,familywise_alpha=.05,paired_difference_ci=dict(method='two_sided_t_bonferroni',confidence_level=1-.05/3),descriptive_bootstrap=dict(method='paired_donor_percentile',confidence_level=.95,resamples=10000,seed=20261003),unestimable_primary=dict(conditions=['fewer_than_two_donors','zero_difference_variance','nonfinite_required_error','unsupported_donor_independence'],bookkeeping_pvalue=1.,is_observed_pvalue=False),primary_completeness='same_predeclared_eligible_profiles_all_models_no_silent_deletion',test_inventory=dict(study='GSE90909',historical_expected_profiles=52,historical_expected_donors=12,eligibility_audit_completed_in_this_step=False),sensitivity_seeds_are_independent_donors=False,locked_test_opened=False,clinical_claim=False)
    save('config.json',config);save('splits.json',splits);save('planned-jobs.json',jobs);save('expected-artifacts.json',final);save('prediction-scope.json',scope);save('statistical-plan.json',statistical)
    evidence=['context/model-abcd/v1/config.json','context/model-abcd/v1/REPORT.md','context/model-abcd/v1/verification.json','context/classical-comparison/v1/config.json','context/classical-comparison/v1/REPORT.md','context/classical-comparison/v1/verification.json','context/control-bias-investigation/v1/REPORT.md','context/transfer-robust/v1/REPORT.md','context/transfer-robust/v1/verification.json','context/extended-models/v1/REPORT.md','context/extended-models/v1/data-recheck.json','context/extended-models/v1/verification.json','context/clustering/v1/REPORT.md','context/clustering/v1/verification.json','context/final-selection/v1/DECISION_PLAN.md','scripts/development_preprocessing.py','scripts/core_baseline.py','scripts/core_transfer.py','scripts/abcd_models.py','scripts/matched_classical.py','scripts/training_contract.py','data/model_ready/development_raw_f6022cfaaca6d1d9/release.json','data/model_ready/development_raw_f6022cfaaca6d1d9/metadata.json']
    originals=read(ROOT/'context/final-documents/manifest.json')['files']
    for item in originals:
        assert sha(ROOT/item['project_path'])==item['sha256'];evidence.append(item['project_path'])
    save('evidence-hashes.json',{n:sha(ROOT/n) for n in evidence})
    outputs=['PROTOCOL.md','STATISTICAL_PLAN.md','PREDICTION_SCOPE.md','config.json','splits.json','planned-jobs.json','expected-artifacts.json','prediction-scope.json','statistical-plan.json','evidence-hashes.json']
    save('protocol-freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='FINALIZED_RULES_NO_TRAINING',sha256={n:sha(OUT/n) for n in outputs},preparation_source_sha256=sha(Path(__file__)),criteria_accepted_by_user=True,new_training_run=False,model_hyperparameters_selected=False,locked_test_opened=False,app_cloud_or_github_changed=False))
    print('Frozen final-selection rules: 5 folds, 280 planned candidate fits, 17 trained artifacts + median planned. No fitting.')

if __name__=='__main__':main()
