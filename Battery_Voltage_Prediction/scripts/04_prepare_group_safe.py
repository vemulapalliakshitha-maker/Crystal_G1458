"""Fit Li and mixed preprocessors on group-separated training records."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'
OUT=ROOT/'artifacts'/'preprocessing'
OUT.mkdir(parents=True,exist_ok=True)
FEATURES=[f'feat_{i}' for i in range(1,240)]
COLS=[f'PC_{i}' for i in range(1,81)]
df=pd.read_csv(DATA/'remaining_features.csv')
for task, ions in {'li':['Li'],'mixed':['Li','Ca','Mg','Zn','Al','Y']}.items():
    sub=df.loc[df.working_ion.isin(ions)].dropna(subset=FEATURES+['average_voltage','battery_id','source_row']).reset_index(drop=True)
    if sub.empty: raise ValueError(f'No data for {task}')
    if not np.isfinite(sub[FEATURES].to_numpy(dtype=float)).all():raise ValueError('Nonfinite features')
    if sub.source_row.duplicated().any():raise ValueError('Duplicate source rows')
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.10,random_state=42).split(sub,groups=sub.battery_id))
    a,b=sub.iloc[tr],sub.iloc[te]
    assert set(a.battery_id).isdisjoint(set(b.battery_id))
    pipe=Pipeline([('scale',MinMaxScaler(feature_range=(-1,1))),('pca',PCA(n_components=80,random_state=42))])
    for suffix,features,orig in [('train',pipe.fit_transform(a[FEATURES]),a),('holdout',pipe.transform(b[FEATURES]),b)]:
        out=pd.DataFrame(features,columns=COLS)
        for c in ['average_voltage','working_ion','battery_id','source_row']:out[c]=orig[c].to_numpy()
        out.to_csv(DATA/f'{task}_{suffix}_pca_c001.csv',index=False)
    joblib.dump(pipe,OUT/f'{task}_preprocessor_c001.joblib')
    print(f'{task}: training={len(a)} holdout={len(b)} group overlap=0')
na=pd.read_csv(DATA/'na_features.csv')
if not na.working_ion.eq('Na').all():raise ValueError('Na-only evaluation file required')
for task in ('li','mixed'):
    pipe=joblib.load(OUT/f'{task}_preprocessor_c001.joblib')
    out=pd.DataFrame(pipe.transform(na[FEATURES]),columns=COLS)
    out['average_voltage']=na.average_voltage.to_numpy()
    if 'source_row' in na:out['source_row']=na.source_row.to_numpy()
    out.to_csv(DATA/f'na_{task}_pca_c001.csv',index=False)
print('Preprocessing done')
