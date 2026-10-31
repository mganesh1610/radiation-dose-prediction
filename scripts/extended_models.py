"""Additional bounded estimators; all learned transforms use fit inputs only."""
import warnings
from collections import defaultdict
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

REGRESSION = ('ridge','lasso','svr_linear','svr_rbf','random_forest','adaboost','xgboost','polynomial_ridge')
CLASSIFICATION = ('logistic_l1','logistic_l2','gaussian_nb')
STOCHASTIC = ('random_forest','adaboost','xgboost','logistic_l1','logistic_l2')

class WeightedPolynomialRidge:
    """Degree-two model on 5/10 training-fitted weighted principal components."""
    def __init__(self, components, alpha): self.components, self.alpha = components, alpha
    def fit(self,x,y,sample_weight):
        w=np.asarray(sample_weight,dtype=float);w=w/w.sum()
        self.center_=np.average(x,axis=0,weights=w)
        _,_,v=np.linalg.svd((x-self.center_)*np.sqrt(w[:,None]),full_matrices=False)
        self.axes_=v[:self.components]
        self.poly_=PolynomialFeatures(degree=2,include_bias=False)
        p=self.poly_.fit_transform((x-self.center_)@self.axes_.T)
        self.scaler_=StandardScaler().fit(p,sample_weight=w)
        self.model_=Ridge(alpha=self.alpha,solver='svd').fit(self.scaler_.transform(p),y,sample_weight=w*len(w))
        return self
    def predict(self,x):
        p=self.poly_.transform((np.asarray(x)-self.center_)@self.axes_.T)
        return self.model_.predict(self.scaler_.transform(p))

def fit_estimator(p,x,y,weights,seed):
    x=np.asarray(x,dtype=float);y=np.asarray(y);w=np.asarray(weights,dtype=float)
    if x.ndim!=2 or y.shape!=(len(x),) or w.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0): raise ValueError('Invalid fit inputs')
    w=w/w.sum()*len(w);name=p['model']
    if name in CLASSIFICATION and set(y)!={0,1}:raise ValueError('Both binary classes required')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if name=='ridge': model=Ridge(alpha=p['alpha'],solver='svd')
        elif name=='lasso': model=Lasso(alpha=p['alpha'],tol=1e-6,max_iter=100000,selection='cyclic')
        elif name=='svr_linear': model=SVR(kernel='linear',C=p['C'],epsilon=p['epsilon'],tol=1e-5,max_iter=1000000)
        elif name=='svr_rbf': model=SVR(kernel='rbf',C=p['C'],gamma=p['gamma'],epsilon=.1,tol=1e-5,max_iter=1000000)
        elif name=='random_forest': model=RandomForestRegressor(n_estimators=200,max_depth=p['max_depth'],min_samples_leaf=p['min_samples_leaf'],max_features=.25,n_jobs=4,random_state=seed)
        elif name=='adaboost': model=AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=p['max_depth'],min_samples_leaf=3),n_estimators=100,learning_rate=p['learning_rate'],loss='linear',random_state=seed)
        elif name=='xgboost': model=XGBRegressor(n_estimators=150,max_depth=p['max_depth'],reg_lambda=p['reg_lambda'],reg_alpha=0,min_child_weight=3,learning_rate=.05,subsample=.8,colsample_bytree=.5,tree_method='hist',device='cpu',objective='reg:squarederror',n_jobs=4,random_state=seed)
        elif name=='polynomial_ridge': model=WeightedPolynomialRidge(p['components'],p['alpha'])
        elif name in ('logistic_l1','logistic_l2'): model=LogisticRegression(C=p['C'],l1_ratio=1. if name.endswith('l1') else 0.,solver='liblinear' if name.endswith('l1') else 'lbfgs',tol=1e-6,max_iter=20000,random_state=seed)
        elif name=='gaussian_nb': model=GaussianNB(var_smoothing=p['var_smoothing'])
        else: raise ValueError('Unknown model')
        model.fit(x,y,sample_weight=w)
    if caught: raise ValueError('Fit warnings: '+'; '.join(str(r.message) for r in caught))
    if not np.isfinite(model.predict(x)).all():raise ValueError('Nonfinite predictions')
    return model

def binary_metrics(y,prob,donors):
    y=np.asarray(y,dtype=int);prob=np.asarray(prob,dtype=float);d=np.asarray(donors)
    if set(y)!={0,1} or prob.shape!=y.shape or d.shape!=y.shape or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):raise ValueError('Invalid binary evaluation')
    h=prob>=.5; correct=[]
    for cls in (0,1):
        idx=y==cls
        correct.append(float(np.mean([np.mean(h[idx&(d==g)]==cls) for g in sorted(set(d[idx]))])))
    return dict(balanced_accuracy=float(np.mean(correct)),specificity=correct[0],sensitivity=correct[1],false_positive_rate=1-correct[0],brier=float(np.mean([np.mean((prob[d==g]-y[d==g])**2) for g in sorted(set(d))])))
