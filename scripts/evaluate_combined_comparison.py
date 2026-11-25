"""Replay the frozen combined-cohort models and reconstruct both comparison columns."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from build_development_explorer import NAMES
from final_model_bundle import predict_panel
from locked_evaluation_stats import bootstrap_interval
from research_catalog import MODEL_NAMES
from train_combined_comparison import (BOOST, DEV, FINAL, RELEASE, SEED, initialize, labels,
                                      predict_gy, read, sha, utc, write)

ALIASES = {'A':'pooled_neural_regression','gradient_boosting':'gradient_boosting_wb_wbc'}
COHORT = Path('context/locked-eligibility/v1/eligible-samples.json')
PANELS = Path('data/processed_locked/GSE90909/v1/frozen-panels.npz')
ARCHIVE = Path('output/retrospective_gb_wbc_2026-11-14')


def require_close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)


def csv_read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    if path.exists():
        existing = csv_read(path)
        if len(existing) != len(rows):
            raise ValueError('Existing export has different coverage: ' + path.name)
        for previous, current in zip(existing, rows):
            if set(previous) != set(current):
                raise ValueError('Existing export has different fields: ' + path.name)
            for k,v in current.items():
                if isinstance(v, (int,float)):
                    require_close(float(previous[k]),v)
                elif previous[k] != v:
                    raise ValueError('Existing export has different identifiers: ' + path.name)
        return
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def donor_scores(actual, predicted, donors):
    y, p, groups = np.asarray(actual), np.asarray(predicted), np.asarray(donors)
    assert y.shape == p.shape == groups.shape and np.isfinite(p).all() and np.isfinite(y).all()
    by_donor = [p[groups==g] - y[groups==g] for g in sorted(set(groups))]
    absolute = np.array([np.abs(d).mean() for d in by_donor])
    return dict(mae_gy=float(absolute.mean()),rmse_gy=float(np.sqrt(np.mean([np.mean(d*d) for d in by_donor]))),
                bias_gy=float(np.mean([d.mean() for d in by_donor]))), absolute


def classifier_scores(actual, probabilities, donors):
    y, p, groups = np.asarray(actual), np.asarray(probabilities), np.asarray(donors)
    assert y.shape == p.shape == groups.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    rates = []
    for v in (0,1):
        cohort = y==v
        correct = (p[cohort]>=.5)==v
        rates.append(float(np.mean([correct[groups[cohort]==g].mean() for g in set(groups[cohort])])))
    return dict(specificity=rates[0],sensitivity=rates[1],balanced_accuracy=sum(rates)/2)


def reconstruct_development(root):
    dev = read(root/DEV/'summary.json')
    identity = sha(root/DEV/'protocol.json')
    assert identity == dev['protocol_sha256']
    split = read(root/'context/model-abcd/v1/splits.json')
    lookup = labels(root)
    score_rows = {r['model']:r for r in dev['regression']}
    classifier_rows = {(r['model'],r['study']):r for r in dev['classification']}
    classifiers = {m for m,s in classifier_rows}
    assert (len(score_rows),len(classifiers)) == (20,3)
    out, classification = {}, []
    for model in sorted(set(score_rows)|classifiers):
        all_ids, all_p, study_maes = [], [], []
        for f in split['outer']:
            study = f['study']
            r = read(root/DEV/'outer'/study/(model+'.json'))
            ids, p = r['validation_ids'], r['predictions'][0]
            assert (r['model'],r['study'],r['identity']) == (model,study,identity)
            assert ids == f['validation'] and len(p)==len(ids) and r['training_profiles']==567-len(ids)
            assert not set(ids) & (set(f['training']) | set(split['auxiliary_training']))
            assert len(set(f['training']) | set(split['auxiliary_training']) | set(ids)) == 567
            assert all(lookup[i]['study']==study for i in ids)
            all_ids.extend(ids)
            all_p.extend(p)
            donors = [lookup[i]['donor_group'] for i in ids]
            if model in classifiers:
                values = classifier_scores([lookup[i]['exposure'] for i in ids],p,donors)
                for k,v in values.items(): require_close(v,classifier_rows[model,study][k])
                classification.append(dict(model=model,study=study,profiles=len(ids),**values))
            else:
                scores,_ = donor_scores([lookup[i]['dose_gy'] for i in ids],p,donors)
                for k,v in scores.items(): require_close(v,score_rows[model]['by_study'][study][k.removesuffix('_gy')])
                study_maes.append(scores['mae_gy'])
        assert sorted(all_ids) == sorted(split['core_profiles']) and len(all_p)==145
        donors = [lookup[i]['donor_group'] for i in all_ids]
        assert len(set(donors)) == 30
        if model in classifiers:
            values = classifier_scores([lookup[i]['exposure'] for i in all_ids],all_p,donors)
            for k,v in values.items(): require_close(v,classifier_rows[model,'pooled'][k])
            classification.append(dict(model=model,study='pooled',profiles=145,**values))
        else:
            scores,_ = donor_scores([lookup[i]['dose_gy'] for i in all_ids],all_p,donors)
            scores.update(worst_study_mae_gy=max(study_maes),equal_study_mae_gy=float(np.mean(study_maes)))
            for k,v in scores.items(): require_close(v,score_rows[model][k])
            out[model] = scores
    return out, classification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root, output = args.project.resolve(), args.output.resolve()
    initialize()
    run = root/'data/experiments'/RELEASE
    freeze = read(run/'model-freeze.json')
    protocol = read(run/'protocol.json')
    assert freeze['selection_sha256']==sha(run/'selected.json') and freeze['protocol_sha256']==sha(run/'protocol.json')
    assert set(freeze['models']) == set(protocol['models']) and len(freeze['models'])==20
    for name,digest in protocol['input_hashes'].items():
        assert sha(root/name) == digest, name
    for name,digest in protocol['source_hashes'].items():
        assert sha(Path(__file__).resolve().parent/name) == digest, name
    for a in freeze['models'].values():
        assert sha(root/a['path'])==a['sha256']
    development, classifiers = reconstruct_development(root)
    entry = read(root/'context/locked-eligibility/v1/evaluation-entry-freeze.json')['sha256']
    assert sha(root/COHORT)==entry[str(COHORT)] and sha(root/PANELS)==entry[str(PANELS)]
    cohort = read(root/COHORT)
    ids = [r['gsm'] for r in cohort]
    donors = [r['donor_group'] for r in cohort]
    actual = np.array([r['dose_gy'] for r in cohort])
    assert len(set(ids))==len(ids)==52 and len(set(donors))==12 and actual.min()==0 and actual.max()==4
    assert all(r['eligible'] and r['study']=='GSE90909' for r in cohort)
    train_labels = labels(root)
    assert not set(ids) & (set(train_labels)|{s for r in train_labels.values() for s in r['source_arrays']})
    assert not set(donors) & {r['donor_group'] for r in train_labels.values()}
    start = run/'inference-start.json'
    if start.exists():
        previous = read(start)
        if previous['model_freeze_sha256'] != sha(run/'model-freeze.json') or previous['test_panel_sha256'] != sha(root/PANELS):
            raise ValueError('This evaluation already belongs to a different model or test panel')
    else:
        write(start,dict(utc=utc(),model_freeze_sha256=sha(run/'model-freeze.json'),test_panel_sha256=sha(root/PANELS)))
    with np.load(root/PANELS,allow_pickle=False) as z:
        assert z['sample_ids'].tolist()==ids
        panels = {k:z[k].copy() for k in ('rank','log2')}
    assert all(p.shape==(52,500) and np.isfinite(p).all() for p in panels.values())
    idx = read(root/FINAL/'artifact-index.json')
    info = read(root/FINAL/'folds/full-development/fit.json')
    archive = defaultdict(dict)
    for r in csv_read(root/ARCHIVE/'predictions.csv'):
        if int(r['seed'])==SEED:
            archive[r['model']][r['profile_id']] = r
    old_metrics = {r['model']:r for r in csv_read(root/ARCHIVE/'primary-metrics.csv')}
    ix = np.load(root/'context/locked-evaluation/v1/bootstrap-donor-indices.npy',allow_pickle=False)
    assert np.array_equal(ix,np.random.default_rng(20261003).integers(0,12,(10000,12)))
    table, predictions, final_metrics = [], [], {}
    for model in sorted(development,key=lambda m:development[m]['mae_gy']):
        a = freeze['models'][model]
        path = root/a['path']
        X = panels[a['params']['representation']]
        if model in ('B','C','D'):
            p = predict_panel(path.parent,X,idx['identity'])
        elif model == 'gradient_boosting':
            p = joblib.load(path).predict(X)*info['target_scale']+info['target_mean']
        else:
            p = predict_gy(model,joblib.load(path),X,info)
        scores, donor_mae = donor_scores(actual,p,donors)
        interval = bootstrap_interval(donor_mae,ix)
        scores.update(mae_ci_low_gy=interval['low'],mae_ci_high_gy=interval['high'])
        key = ALIASES.get(model,model)
        if a['source']=='archived':
            record = archive[key]
            assert set(record)==set(ids) and [record[i]['donor_group'] for i in ids]==donors
            require_close(actual,[float(record[i]['actual_gy']) for i in ids])
            require_close(p,[float(record[i]['predicted_gy']) for i in ids])
            for m in ('mae','rmse','bias'): require_close(scores[m+'_gy'],float(old_metrics[key]['donor_'+m]))
            require_close([interval['low'],interval['high']], [float(old_metrics[key][k]) for k in ('mae_ci_low','mae_ci_high')])
        final_metrics[key] = dict(**scores,source=a['source'],representation=a['params']['representation'])
        name = 'Pooled neural dose regression' if model=='A' else MODEL_NAMES.get(key,NAMES[model][0])
        row = dict(model=key,label=name,family=NAMES[model][1],training_profiles=567,development_held_out_profiles=145,
                   development_donors=30,**{'development_'+k:v for k,v in development[model].items()},
                   final_study='GSE90909',final_profiles=52,final_donors=12,**{'final_'+k:v for k,v in scores.items()},fit_source=a['source'])
        table.append(row)
        predictions.extend(dict(model=key,profile_id=i,donor_group=d,actual_gy=float(y),prediction_gy=float(q)) for i,d,y,q in zip(ids,donors,actual,p))
    assert len(predictions)==1040 and len(table)==20 and len(classifiers)==12
    # The whole-blood arm is a separate fixed-settings comparator, outside the 20 combined models.
    control = archive['gradient_boosting']
    assert set(control)==set(ids)
    fixed,_ = donor_scores(actual, [float(control[i]['predicted_gy']) for i in ids],donors)
    require_close(fixed['mae_gy'],float(old_metrics['gradient_boosting']['donor_mae']))
    summary = dict(release=RELEASE,evaluated_at=utc(),protocol_sha256=sha(run/'protocol.json'),model_freeze_sha256=sha(run/'model-freeze.json'),
                   cohort='GSE90909',profiles=52,donors=12,training_profiles=567,primary_seed=SEED,
                   retrospective=True,selection_fits=freeze['selection_fits'],new_final_fits=freeze['new_final_fits'],replayed_final_fits=4,
                   development_prediction_rows=2900,development_classifier_probabilities=435,external_predictions=len(predictions),
                   models=final_metrics,fixed_whole_blood_control=fixed)
    results = run/'results.json'
    if results.exists():
        previous = read(results)
        summary['evaluated_at'] = previous['evaluated_at']
        # Parallel tree summation can vary in the last binary digit.
        for field in ('models','fixed_whole_blood_control'):
            actual_rows = summary[field] if field == 'models' else {'control':summary[field]}
            previous_rows = previous[field] if field == 'models' else {'control':previous[field]}
            if set(actual_rows) != set(previous_rows):
                raise ValueError('The archived model set changed')
            for name, metrics in actual_rows.items():
                if set(metrics) != set(previous_rows[name]):
                    raise ValueError('The archived score fields changed')
                for k,v in metrics.items():
                    old = previous_rows[name][k]
                    if isinstance(v, float):
                        require_close(v,old)
                    elif old != v:
                        raise ValueError('The archived fit metadata changed')
            summary[field] = previous[field]
        if summary != previous:
            raise ValueError('The archived evaluation provenance changed')
    else:
        write(results,summary)
    output.mkdir(exist_ok=True,parents=True)
    csv_write(output/'dose-comparison.csv',table)
    csv_write(output/'gse90909-predictions.csv',predictions)
    csv_write(output/'development-classifiers.csv',classifiers)
    write(output/'verification.json',dict(release=RELEASE, results_sha256=sha(run/'results.json'),
          files={p.name:sha(p) for p in sorted(output.glob('*.csv'))},no_training_test_overlap=True,
          development_table_reconstructed=True,all_archived_predictions_replayed=True,retrospective=True))
    print('Verified 20 development and retrospective rows, 2,900 + 1,040 dose predictions, 435 development classifier probabilities.')
    print('Gradient boosting',final_metrics['gradient_boosting_wb_wbc']['mae_gy'])


if __name__ == '__main__':
    main()
