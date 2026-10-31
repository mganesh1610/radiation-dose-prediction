"""Numeric-only baseline serialization; no unpickling or executable model payloads."""
import json
from pathlib import Path
import numpy as np
from core_transfer import CoreTransfer


def estimator_spec(name, model, y):
    if name=='median_null': return {'kind':name,'value':float(np.median(y))}
    if name=='elastic_net':
        return {'kind':'linear','coef':model.coef_.reshape(-1).tolist(),'intercept':float(model.intercept_)}
    if name=='pls':
        coef=model.coef_.reshape(-1)
        return {'kind':'linear','coef':coef.tolist(), 'intercept':float(np.asarray(model.intercept_).item()-model._x_mean@coef)}
    if name=='gradient_boosting':
        trees=[]
        for estimator in model.estimators_.ravel():
            t=estimator.tree_
            trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),
                'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value.reshape(-1).tolist()})
        return {'kind':'gradient_boosting','initial':float(model.init_.constant_.item()),
                'learning_rate':float(model.learning_rate),'trees':trees}
    raise ValueError('Unsupported estimator')


def numeric_predict(spec, x):
    if spec['kind']=='median_null': return np.full(len(x),spec['value'])
    if spec['kind']=='linear': return x@np.asarray(spec['coef'])+spec['intercept']
    if spec['kind']!='gradient_boosting': raise ValueError('Unknown model representation')
    result=np.full(len(x),spec['initial'])
    for tree in spec['trees']:
        for row in range(len(x)):
            node=0
            while tree['left'][node]!=-1:
                # sklearn decision trees consume float32 input.
                value=np.float32(x[row,tree['feature'][node]])
                node=tree['left'][node] if value<=tree['threshold'][node] else tree['right'][node]
            result[row]+=spec['learning_rate']*tree['value'][node]
    return result


class PortablePredictor:
    def __init__(self, folder):
        folder=Path(folder)
        self.spec=json.loads((folder/'model.json').read_text())
        if self.spec['schema_version']!=1: raise ValueError('Unknown model schema')
        self.transfer=CoreTransfer.load(folder/'reference.npz')
        with np.load(folder/'preparation.npz',allow_pickle=False) as z:
            self.prep={k:z[k] for k in z.files}

    def predict(self, values, probe_ids):
        genes,matrix,_=self.transfer.transform(values,np.isfinite(values),probe_ids)
        if not np.array_equal(genes,self.prep['gene_ids']): raise ValueError('Gene order mismatch')
        selected=self.prep['selected']
        x=matrix.T[:,selected]
        x=np.where(np.isfinite(x),x,self.prep['medians'][selected])
        if self.spec['method']!='gradient_boosting': x=(x-self.prep['mean'])/self.prep['scale']
        result=numeric_predict(self.spec['estimator'],x)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite prediction')
        return result
