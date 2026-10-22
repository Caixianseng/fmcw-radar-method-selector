"""Group-separated offline fitting; targets may use clean references, inference may not."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from .api import predict


def train(dataset, destination):
    with np.load(dataset,allow_pickle=False) as z:
        X=z['features'];metrics=z['metrics'];groups=z['groups'];valid=z['reference_valid'].astype(bool)
        failed=z['mp_failed'].astype(bool)
    identities=np.unique(groups)
    if len(identities)<4:raise ValueError('At least four independent sessions are required')
    split=max(1,int(.75*len(identities)))
    train_ids=identities[:split];val_ids=identities[split:]
    fit=np.isin(groups,train_ids)&valid;validation=np.isin(groups,val_ids)
    if not fit.any() or not validation.any():raise ValueError('Training/validation split is empty')
    y=np.c_[-metrics[:,:,2],(metrics[:,:,0]>1).astype(float),metrics[:,:,3]]
    xs=StandardScaler().fit(X[fit]);ys=StandardScaler().fit(y[fit])
    candidates=[]
    for hidden in ((32,16),(64,32)):
        net=MLPRegressor(hidden_layer_sizes=hidden,alpha=.5,random_state=930123,
            activation='relu',solver='adam',learning_rate_init=.001,batch_size=128,
            max_iter=4000,tol=1e-5,n_iter_no_change=30,early_stopping=False)
        net.fit(xs.transform(X[fit]),ys.transform(y[fit]))
        bundle=dict(net=net,x_scaler=xs,y_scaler=ys,columns=tuple(range(32)),
                    penalties=(2.,0.),kind='risk',methods=('MP','C-AR','JSS'),schema_version=1)
        choice=predict(X[validation],bundle)['choice'];m=metrics[validation]
        choice[(choice==0)&failed[validation]]=1
        quality=m[np.arange(len(choice)),choice]
        candidates.append((float(quality[:,2].mean()),bundle,hidden))
    # Exact ties keep the declared enumeration. No holdout data is consulted.
    winner=max(range(len(candidates)),key=lambda i:(candidates[i][0],-i))
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(candidates[winner][1],destination,compress=3)
    return dict(training_sessions=train_ids.tolist(),validation_sessions=val_ids.tolist(),
        training_chirps=int(fit.sum()),validation_chirps=int(validation.sum()),
        selected_hidden=list(candidates[winner][2]),
        validation_mean_sinr_db=candidates[winner][0],new_holdout_used=False,
        note='General reproduction entry; does not overwrite the bundled frozen paper model. Historical development eligibility checks are documented separately.')
