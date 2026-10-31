import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from abcd_models import DoseNetwork, balanced_weights, fit_network, tensor, objective, predict, save_model, load_model

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class ABCDTests(unittest.TestCase):
    def batch(self):
        rng=np.random.default_rng(7)
        return {k:tensor(v) for k,v in dict(core_x=rng.normal(size=(8,5)),core_y=np.arange(8)/4,
            core_weights=np.ones(8)/8,core_exposure=[0,1]*4,core_binary_weights=np.ones(8)/8,
            aux_x=rng.normal(size=(6,5)),aux_y=[0,1]*3,aux_weights=np.ones(6)/6,
            aux_exposure=[0,1]*3,aux_binary_weights=np.ones(6)/6).items()}
    def params(self,s):return dict(strategy=s,lr=.001,weight_decay=.0001,steps=6,pretrain_steps=4,classification_lambda=.1,auxiliary_mass=.5)
    def test_weights_balance_sources_donors_classes(self):
        s=np.array(['x']*6+['y']*2);g=np.array(['x:a']*4+['x:b']*2+['y:a']*2)
        w=balanced_weights(s,g);self.assertAlmostEqual(w[:6].sum(),.5);self.assertAlmostEqual(w[:4].sum(),.25)
        cls=np.array([0,0,0,1,0,1,0,1]);w=balanced_weights(s,g,cls)
        for st in ['x','y']:
            for cl in [0,1]:self.assertAlmostEqual(w[(s==st)&(cls==cl)].sum(),.25)
    def test_b_c_never_use_auxiliary_numerical_labels(self):
        for s in ['B','C']:
            a=self.batch();b=copy.deepcopy(a);b['aux_y'][:]=float('nan')
            m,_=fit_network(a,self.params(s),31,hidden=(4,3));n,_=fit_network(b,self.params(s),31,hidden=(4,3))
            for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)
    def test_c_auxiliary_branch_has_no_dose_head_gradient(self):
        batch=self.batch();m=DoseNetwork(5,(4,3));p=self.params('C')
        core=objective(m,batch,'A',p);core.backward();before=m.dose.weight.grad.clone();m.zero_grad()
        objective(m,batch,'C',p).backward();torch.testing.assert_close(before,m.dose.weight.grad,rtol=0,atol=0)
    def test_d_uses_auxiliary_dose(self):
        a=self.batch();b=copy.deepcopy(a);b['aux_y']+=3
        m,_=fit_network(a,self.params('D'),31,hidden=(4,3));n,_=fit_network(b,self.params('D'),31,hidden=(4,3))
        self.assertFalse(torch.equal(m.dose.weight,n.dose.weight))
        b['aux_y'][:]=float('nan')
        with self.assertRaises(ValueError):fit_network(b,self.params('D'),31,hidden=(4,3))
    def test_a_has_no_auxiliary_requirement(self):
        b={k:v for k,v in self.batch().items() if not k.startswith('aux')}
        m,_=fit_network(b,self.params('A'),2,hidden=(4,3));self.assertTrue(torch.isfinite(m.dose.weight).all())
    def test_core_preparation_accepts_empty_auxiliary_fit_membership(self):
        from run_abcd_experiments import fit_core
        from development_preprocessing import AUX
        rng=np.random.default_rng(42);ids=np.array([f'p{i}' for i in range(8)])
        studies=np.array(['GSE8917']*4+['GSE65292']*4);donors=np.array(['GSE8917:a']*2+['GSE8917:b']*2+['GSE65292:a']*2+['GSE65292:b']*2)
        rows=[dict(profile_id=p,study=s,donor_group=d,dose_gy=float(i%2)) for i,(p,s,d) in enumerate(zip(ids,studies,donors))]
        a=dict(values=rng.uniform(1,100,(10,8)),detected=np.ones((10,8),bool),sample_ids=ids,study_ids=studies,split_group_ids=donors,
               probe_ids=np.array([f'probe{i}' for i in range(10)]),gene_keys=np.array([str(i) for i in range(10)]))
        data={'core':a,**{s:dict(profile_ids=np.array([],dtype=str)) for s in AUX}}
        meta=dict(profiles=rows,annotation_gene_ids=list(a['gene_keys']),mapping={})
        prep,genes,mean,scale=fit_core(data,meta,list(ids[:2])+list(ids[4:6]),{r['profile_id']:r for r in rows})
        self.assertEqual(prep.aux_training,());self.assertEqual(prep.aux,{})
        self.assertEqual(len(prep.core_training),4);self.assertGreater(scale,0)
    def test_pretraining_leaves_dose_head_initialization_untouched(self):
        torch.manual_seed(8);m=DoseNetwork(5,(4,3));before=m.dose.weight.detach().clone()
        opt=torch.optim.AdamW(m.parameters(),lr=.01)
        for _ in range(5):
            opt.zero_grad(set_to_none=True);objective(m,self.batch(),'B',self.params('B'),True).backward();opt.step()
        torch.testing.assert_close(before,m.dose.weight,rtol=0,atol=0)
    def test_determinism_and_checkpoint_replay(self):
        b=self.batch();p=self.params('C');m,_=fit_network(b,p,55,hidden=(4,3));n,_=fit_network(b,p,55,hidden=(4,3))
        x=b['core_x'].numpy();expected=predict(m,x,2,3)
        np.testing.assert_array_equal(expected,predict(n,x,2,3))
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'model.pt';save_model(file,m,p,55,2,3,hidden=(4,3));r,meta=load_model(file)
            np.testing.assert_array_equal(expected,predict(r,x,meta['target_mean'],meta['target_scale']))
            np.testing.assert_allclose(expected,np.concatenate([predict(r,row[None],2,3) for row in x]),atol=1e-5,rtol=0)

if __name__=='__main__':unittest.main()
