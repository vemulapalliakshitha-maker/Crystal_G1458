"""Standalone training and evaluation for LI_DNN.
Run this file from PS1/C001_Original_Project_Correction.
No import from frozen_models.py required.
"""
"""Frozen C002 'model tuning' checkpoint, NOT the later architecture checkpoint.
Run in PS1/C001_Original_Project_Correction; reads ../remaining_features.csv, ../na_features.csv.
Requires pandas numpy scikit-learn joblib tensorflow.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

SEED = 42
HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / 'data' / 'processed'
OUT = HERE.parent / 'artifacts' / 'frozen_models'
FEATS = [f'feat_{i}' for i in range(1,240)]
FAMILIES = {'Li':['Li'], 'Mixed':['Li','Ca','Mg','Zn','Al','Y']}
# Frozen configs obtained by tracing accepted changes in 28_auto_improve_six_models.py:
CONFIGS = {
 'Li DNN': dict(scaler='standard',pca=80,epochs=120,optimizer='rmsprop',lr=.001,l2=.0001,hidden=[60,30],dropout=[.25,.10]),
 'Li SVR': dict(scaler='minmax',pca=80,C=100,gamma=.02,epsilon=.1),
 'Li KRR': dict(scaler='minmax',pca=80,alpha=.01,gamma=.02),
 'Mixed DNN':dict(scaler='standard',pca=80,epochs=100,optimizer='rmsprop',lr=.001,l2=.0001,hidden=[60,30],dropout=[.25,.10]),
 'Mixed SVR':dict(scaler='minmax',pca=120,C=100,gamma=.01,epsilon=.1),
 'Mixed KRR':dict(scaler='minmax',pca=80,alpha=.01,gamma=.02),
}
# Historical checkpoint numbers are printed separately and never substituted for predictions.
EXPECTED = {
 'Li DNN':[.2975,.3809,.6625,.6384,.6041,.3775],
 'Li SVR':[.2889,.4140,.6354,.6405,.5713,.4411],
 'Li KRR':[.3232,.4179,.6440,.6298,.5862,.4611],
 'Mixed DNN':[.3228,.3856,.8407,.5697,.6786,.4146],
 'Mixed SVR':[.2956,.4126,.8333,.6870,.5714,.4325],
 'Mixed KRR':[.2751,.4130,.8279,.7535,.4915,.4360],
}
METRICS=['Train_MAE','H_MAE','H_R2','Na_MAE','Na_R2','Validation_MAE']

def read_files():
    source_path=SOURCE_DIR/'remaining_features.csv'; na_path=SOURCE_DIR/'na_features.csv'
    for p in [source_path,na_path]:
        if not p.exists(): raise FileNotFoundError(f'Missing {p}. Put this script in C001_Original_Project_Correction, with CSV files in PS1.')
    df=pd.read_csv(source_path); na=pd.read_csv(na_path)
    for title, frame, extra in [('source',df,['battery_id','source_row','working_ion']),('Na',na,['working_ion'])]:
        req=FEATS+['average_voltage']+extra
        missing=set(req)-set(frame.columns)
        if missing: raise ValueError(f'{title} missing: {sorted(missing)}')
        if not np.isfinite(frame[FEATS+['average_voltage']].to_numpy(dtype=float)).all():
            raise ValueError(f'{title} contains nonfinite numeric values')
    if not na.working_ion.astype(str).str.strip().eq('Na').all():
        raise ValueError('na_features.csv must contain only Na rows')
    if df.source_row.duplicated().any(): raise ValueError('Source-row IDs are duplicated')
    return df,na

def make_splits(df):
    split={}
    for family,ions in FAMILIES.items():
        data=df[df.working_ion.isin(ions)].reset_index(drop=True)
        if data.empty or data.battery_id.isna().any(): raise ValueError(f'Invalid {family} dataset')
        outer=GroupShuffleSplit(n_splits=1,test_size=.10,random_state=42)
        tr,ho=next(outer.split(data,groups=data.battery_id))
        train=data.iloc[tr].reset_index(drop=True); hold=data.iloc[ho].reset_index(drop=True)
        inner=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=123)
        it,iv=next(inner.split(train,groups=train.battery_id))
        subtrain=train.iloc[it].reset_index(drop=True); val=train.iloc[iv].reset_index(drop=True)
        assert set(train.battery_id).isdisjoint(set(hold.battery_id))
        assert set(subtrain.battery_id).isdisjoint(set(val.battery_id))
        split[family]=(train,hold,subtrain,val)
        print(f'{family}: train={len(train)}, H={len(hold)}, inner_train={len(subtrain)}, validation={len(val)}, group overlaps=0')
    return split

def preprocessor(cfg):
    scale=StandardScaler() if cfg['scaler']=='standard' else MinMaxScaler(feature_range=(-1,1))
    stages=[('scaler',scale)]
    if cfg['pca'] is not None:
        stages.append(('pca',PCA(n_components=cfg['pca'],svd_solver='randomized',random_state=SEED)))
    return Pipeline(stages)

def fit(kind,cfg,X,y):
    if kind=='SVR':
        model=SVR(kernel='rbf',C=cfg['C'],gamma=cfg['gamma'],epsilon=cfg['epsilon'])
    elif kind=='KRR':
        model=KernelRidge(kernel='rbf',alpha=cfg['alpha'],gamma=cfg['gamma'])
    else:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Input,Dense,Dropout
        from tensorflow.keras.regularizers import l2
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        layers=[Input(shape=(X.shape[1],))]
        for width,drop in zip(cfg['hidden'],cfg['dropout']):
            layers.extend([Dense(width,activation='relu',kernel_regularizer=l2(cfg['l2'])),Dropout(drop)])
        layers.append(Dense(1))
        model=Sequential(layers)
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=cfg['lr']) if cfg['optimizer']=='rmsprop' else tf.keras.optimizers.Adam(learning_rate=cfg['lr'])
        model.compile(optimizer=optimizer,loss='mae')
        model.fit(X.astype('float32'),y.astype('float32'),epochs=cfg['epochs'],batch_size=32,verbose=0)
        return model
    model.fit(X,y)
    return model

def predict(kind,model,X):
    return model.predict(X.astype('float32'),verbose=0).ravel() if kind=='DNN' else model.predict(X).ravel()

def score(family,kind,cfg,split,na):
    train,hold,inner,val=split
    pval=preprocessor(cfg)
    Xinner=pval.fit_transform(inner[FEATS]); Xval=pval.transform(val[FEATS])
    val_model=fit(kind,cfg,Xinner,inner.average_voltage.to_numpy())
    val_mae=mean_absolute_error(val.average_voltage,predict(kind,val_model,Xval))
    del val_model
    pipeline=preprocessor(cfg)
    Xtrain=pipeline.fit_transform(train[FEATS]); Xh=pipeline.transform(hold[FEATS]); Xna=pipeline.transform(na[FEATS])
    model=fit(kind,cfg,Xtrain,train.average_voltage.to_numpy())
    yt=train.average_voltage.to_numpy(); yh=hold.average_voltage.to_numpy(); yn=na.average_voltage.to_numpy()
    pt=predict(kind,model,Xtrain); ph=predict(kind,model,Xh); pn=predict(kind,model,Xna)
    vals=[mean_absolute_error(yt,pt),mean_absolute_error(yh,ph),r2_score(yh,ph),mean_absolute_error(yn,pn),r2_score(yn,pn),val_mae]
    return vals, model,pipeline,(hold,ph,na,pn)

def run(selection):
    OUT.mkdir(exist_ok=True,parents=True)
    source,na=read_files(); splits=make_splits(source)
    results=[]
    for name,cfg in CONFIGS.items():
        if selection!='all' and name.lower().replace(' ','_')!=selection: continue
        family,kind=name.split()
        print(f'\nTRAINING {name}: {json.dumps(cfg)}',flush=True)
        vals,model,pipe,preds=score(family,kind,cfg,splits[family],na)
        results.append({'Model':name,**dict(zip(METRICS,vals))})
        stem=name.lower().replace(' ','_')
        joblib.dump(pipe,OUT/f'{stem}_preprocessor.joblib')
        if kind=='DNN':model.save(OUT/f'{stem}.keras')
        else:joblib.dump(model,OUT/f'{stem}.joblib')
        for set_name,frame,pred in [('H',preds[0],preds[1]),('Na',preds[2],preds[3])]:
            saved=pd.DataFrame({'actual_voltage':frame.average_voltage.to_numpy(),'predicted_voltage':pred,'absolute_error':abs(frame.average_voltage.to_numpy()-pred)})
            for col in ('battery_id','source_row','working_ion'):
                if col in frame:saved[col]=frame[col].to_numpy()
            saved.to_csv(OUT/f'{stem}_{set_name}_predictions.csv',index=False)
        print('Measured:',', '.join(f'{key}={val:.4f}' for key,val in zip(METRICS,vals)),flush=True)
        diff=np.asarray(vals)-np.asarray(EXPECTED[name])
        print('Max absolute difference from historical checkpoint (across all metrics):',round(float(np.max(abs(diff))),5),flush=True)
    frame=pd.DataFrame(results)
    frame.to_csv(OUT/'frozen_six_model_metrics.csv',index=False)
    pd.DataFrame([{'Model':name,**dict(zip(METRICS,vals))} for name,vals in EXPECTED.items()]).to_csv(OUT/'historical_checkpoint_reference.csv',index=False)
    (OUT/'frozen_configurations.json').write_text(json.dumps(CONFIGS,indent=2),encoding='utf8')
    print('\n=== MEASURED RESULTS (NOT HARDCODED) ===\n'+frame.round(4).to_string(index=False),flush=True)
    print('\n=== REFERENCE NUMBERS FROM YOUR C002 RUN (NOT MEASURED THIS RUN) ===\n'+pd.DataFrame([{'Model':k,**dict(zip(METRICS,v))} for k,v in EXPECTED.items()]).round(4).to_string(index=False),flush=True)
    print('\nOutputs saved:',OUT,flush=True)


if __name__ == "__main__":
    run("li_dnn")
