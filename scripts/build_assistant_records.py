"""Allowlisted public research facts; structured fallback, not generative inference."""
import json
from pathlib import Path
from frozen_inference import encode,digest
ROOT=Path(__file__).resolve().parents[1]

def build():
 cfg=json.loads((ROOT/'context/final-selection/v2/config.json').read_text());release=ROOT/cfg['release']
 if digest((release/'release.json').read_bytes())!=cfg['release_manifest_sha256']:raise ValueError('Untrusted release')
 manifest=json.loads((release/'release.json').read_text());row=next(r for r in manifest['files'] if r['name']=='metadata.json')
 payload=(release/'metadata.json').read_bytes()
 if digest(payload)!=row['sha256']:raise ValueError('Metadata changed')
 md=json.loads(payload)
 profiles=[{k:r.get(k) for k in ('profile_id','study','dose_gy','dose_status','time_hours','role')} for r in md['profiles']]
 final=json.loads((ROOT/'app/static/research-results.json').read_text())
 def fact(id,question,answer,terms,source='#experiments'):
  return dict(id=id,question=question,answer=answer,terms=terms,sources=[dict(label='Project evidence',url=source)])
 facts=[
 fact('inventory','Did we include 30 million gene expression values?','The source inventory contained 30,053,455 expression values across 607 candidate profiles. After eligibility checks and technical-repeat grouping, development uses 567 profiles: 145 core and 422 auxiliary. Values are not independent training samples.',['30 million','30m','expression values','candidate profiles'],'#datasets'),
 fact('core','How many core profiles and donors are there?','The core cohort contains 145 profiles in three studies and 30 study-local donor groups: GSE102971 (100 profiles), GSE8917 (25), and GSE65292 (20). Cross-study donor identities have not been independently verified.',['core profiles','core cohort','core donors'],'#datasets'),
 fact('auxiliary','Which auxiliary studies were included?','The 422 eligible auxiliary profiles are GSE85570 (400), GSE97000 (12) and the selected normal-gravity GSE20173 subset (10).',['auxiliary','additional studies'],'#datasets'),
 fact('final-cohort','What was the final test cohort?','The consumed final test was GSE90909 photon/control: 52 profiles from 12 donor groups, 0–4 Gy, approximately 24 hours. There were 11 controls and six 4 Gy profiles.',['final test cohort','test profiles','test donors','gse90909'],'#experiments'),
 fact('hypothesis','Did Model A outperform B, C and D?','Model A superiority was not supported. All three one-sided A-versus-B/C/D comparisons had Holm-adjusted p = 1.000. Non-rejection does not establish equivalence.',['superiority','hypothesis','outperform','p value','p-value','holm'],'#experiments'),
 fact('leakage','How was preprocessing protected from leakage?','Development model evaluation used training-only normalization, imputation, scaling and gene selection with donor grouping. The final models and preprocessing were frozen before the one-time test. The descriptive explorer uses full development data and must not be used as training-fold features.',['leakage','training-only','preprocessing'],'#experiments'),
 fact('interval','Are the error bars individual prediction intervals?','No. The final metric intervals are 95% donor-percentile bootstrap intervals from 10,000 shared donor resamples, conditional on the fixed models and dose design. No calibrated individual dose prediction interval is available.',['interval','confidence','bootstrap'],'#experiments'),
 fact('clinical','Can this be used clinically?','No. This is a research prototype, not clinically validated. Supported research inputs require human whole blood, an audited ex vivo photon/control context near 24 hours and model-specific assay/provenance review.',['clinical','patient','diagnosis','treatment','safe dose'],'#predictions'),
 fact('clustering','Did clustering improve prediction?','Clustering exposed study-associated structure, including separation of controls. The core Gaussian-mixture clusters matched study membership. The tested cluster/PCA predictors did not improve on the earlier Elastic Net reference. This does not identify a unique technical or biological cause.',['clustering','clusters','k-means','gaussian mixture'],'#analysis'),
 fact('models','Which other models were tested?','The development comparison contains 20 dose predictors, including A–D neural models, Elastic Net, Lasso, Ridge, linear/RBF SVR, Random Forest, XGBoost, AdaBoost, bounded polynomial regression, PLS, median and cluster/PCA predictors. Three exposure classifiers are reported separately. CNNs and reinforcement learning were discussed but not trained.',['other models','all models','cnn','reinforcement','model families'],'#experiments'),
 fact('training','Where does live training run?','Bounded engineering training runs on the project Mac. The cloud app queues work and stores results. The Mac must be awake, connected and authenticated. Reviewer accounts have two engineering jobs per UTC month within the shared project guard.',['live training','training run','train a model','where train'],'#experiments'),
 fact('upload','Can a new upload receive a dose immediately?','A new upload requires a server-side compatibility decision bound to its content and frozen model release. Unknown or unsupported assays receive QC and an explanation instead of a dose. Curated real development examples have training overlap and only demonstrate functionality.',['upload','compatibility','input format'],'#predictions'),
 fact('outliers','Were outliers automatically removed?','No. PCA, quality flags and clustering support review. Unusual samples are not automatically deleted from the explorer; technical measurements are masked according to the audited preprocessing policy.',['outlier','removed','deleted'],'#analysis'),
 fact('statistics','How are differential expression contrasts calculated?','Each dose is paired with the same donor’s control within a study. Donor repeats are averaged first. Two-sided t tests use at least three complete donor pairs and nonzero variance; Benjamini–Hochberg correction covers all testable genes within each contrast. These are exploratory associations.',['differential expression','contrast','benjamini','dose response'],'#analysis'),
 fact('assistant','Is this a generative AI chatbot?','This version is the proposal’s structured-answer fallback. It looks up approved project records and source links without an LLM or model-token API calls. Unknown facts are reported as not recorded. Independent review of the 30-question acceptance sheet remains required.',['chatbot','generative','language model','ai model','assistant'],'#assistant'),
 ]
 out=dict(schema_version=1,mode='structured_retrieval_no_llm',records=profiles,facts=facts,
          final_models=[dict(id=r['id'],name=r['name'],scores=r['scores']) for r in final['model_cards']],
          version='approved-research-records-v1',final_test_status=final['status'])
 data=encode(out);(ROOT/'app/static/assistant-records.json').write_bytes(data)
 (ROOT/'context/cloud/prediction-workflow-v1/assistant-export.json').write_bytes(encode(dict(bytes=len(data),sha256=digest(data),profiles=len(profiles),facts=len(facts),generative_model=False,human_acceptance_review='pending')))
 print('Assistant records:',len(profiles),len(data),'bytes')
if __name__=='__main__':build()
