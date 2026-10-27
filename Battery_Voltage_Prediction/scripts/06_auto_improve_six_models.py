"""C002: cumulative, validation-selected improvements for Li/Mixed DNN, SVR, KRR.
Run from any directory: python 28_auto_improve_six_models.py
Requires: PS1/remaining_features.csv, PS1/na_features.csv.
Never selects a fix using H-set or Na-set. Writes all outputs inside C002_results.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import json
import time
from pathlib import Path
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# Config: set to False to skip repeatedly fitting all six models at each checkpoint.
CHECKPOINT_EACH_STAGE = True
SEED = 42
PCA_COMPONENTS = [40, 80, 120, None]
MIN_REL_IMPROVEMENT = 0.005  # Only accept if validation MAE improves by >=0.5%.
BASE = Path(__file__).resolve().parent
PS1 = BASE.parent / 'data' / 'processed'
OUT = BASE.parent / 'results' / 'measured' / 'C002_optimization'
OUT.mkdir(parents=True, exist_ok=True)
FEATS = [f'feat_{i}' for i in range(1, 240)]
ION_GROUPS = {'Li': ['Li'], 'Mixed': ['Li', 'Ca', 'Mg', 'Zn', 'Al', 'Y']}
OLD_RESULTS = {
 'Li DNN':(.3169,.4160,.7045,.58,.67), 'Li SVR':(.3416,.30,.7059,.67,.58),
 'Li KRR':(.3227,.42,.71,.63,.62), 'Mixed DNN':(.3270,.4240,.7997,.5252,.6683),
 'Mixed SVR':(.2964,.4450,.7636,.5658,.7200), 'Mixed KRR':(.2724,.4874,.7378,.7040,.5856),
}

np.random.seed(SEED)

def safe_path(path):
    if not path.exists(): raise FileNotFoundError(f'Required input missing: {path}')
    return path

def load_data():
    source = pd.read_csv(safe_path(PS1 / 'remaining_features.csv'))
    na = pd.read_csv(safe_path(PS1 / 'na_features.csv'))
    for title, df in [('training', source), ('sodium', na)]:
        needed = FEATS + ['average_voltage', 'working_ion']
        if title == 'training': needed += ['battery_id', 'source_row']
        missing = [x for x in needed if x not in df.columns]
        if missing: raise ValueError(f'{title} missing columns: {missing}')
        if not np.isfinite(df[FEATS + ['average_voltage']].to_numpy(dtype=float)).all():
            raise ValueError(f'{title}: NaN or infinite features/voltage; repair before running')
    if not na['working_ion'].astype(str).str.strip().eq('Na').all():
        raise ValueError('na_features.csv contains non-Na records; external evaluation aborted')
    if source['source_row'].duplicated().any():
        raise ValueError('Duplicated source_row; cannot safely identify features')
    return source, na

def datasets(source):
    output = {}
    for family, ions in ION_GROUPS.items():
        df = source[source['working_ion'].isin(ions)].reset_index(drop=True)
        if df.empty or df['battery_id'].isna().any():
            raise ValueError(f'Empty or missing group ID: {family}')
        # Same deterministic outer group split as prepare_group_safe.py
        outer = GroupShuffleSplit(n_splits=1, test_size=.10, random_state=42)
        train_i, hold_i = next(outer.split(df, groups=df['battery_id']))
        train, hold = df.iloc[train_i].reset_index(drop=True), df.iloc[hold_i].reset_index(drop=True)
        assert set(train.battery_id).isdisjoint(set(hold.battery_id))
        inner = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=123)
        ti, vi = next(inner.split(train, groups=train['battery_id']))
        inner_train, val = train.iloc[ti].reset_index(drop=True), train.iloc[vi].reset_index(drop=True)
        assert set(inner_train.battery_id).isdisjoint(set(val.battery_id))
        output[family] = (train, hold, inner_train, val)
        print(f'{family}: train={len(train)} H={len(hold)} inner_train={len(inner_train)} validation={len(val)}; group overlap=0', flush=True)
    return output

def pipe_for(config, n_train):
    pca = config['pca']
    if pca is not None and pca > min(n_train, len(FEATS)):
        raise ValueError('PCA components exceed sample count')
    scaler = MinMaxScaler(feature_range=(-1,1)) if config['scaler'] == 'minmax' else StandardScaler()
    steps = [('scaler', scaler)]
    if pca is not None:
        steps += [('pca', PCA(n_components=pca, svd_solver='randomized', random_state=SEED))]
    return Pipeline(steps)

def neural_model(inputs, config):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.regularizers import l2
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    hidden = config['hidden']
    layers = [Input(shape=(inputs,))]
    for width, drop in zip(hidden, config['dropout']):
        layers += [Dense(width, activation='relu', kernel_regularizer=l2(config['l2'])), Dropout(drop)]
    layers += [Dense(1)]
    model = Sequential(layers)
    opt_cls = tf.keras.optimizers.Adam if config['optimizer'] == 'adam' else tf.keras.optimizers.RMSprop
    model.compile(optimizer=opt_cls(learning_rate=config['lr']), loss='mae')
    return model

def fit_model(kind, config, x, y):
    if kind == 'SVR':
        model = SVR(kernel='rbf', C=config['C'], gamma=config['gamma'], epsilon=config['epsilon'])
        model.fit(x, y)
    elif kind == 'KRR':
        model = KernelRidge(kernel='rbf', alpha=config['alpha'], gamma=config['gamma'])
        model.fit(x, y)
    else:
        model = neural_model(x.shape[1], config)
        model.fit(x.astype('float32'), y.astype('float32'), epochs=config['epochs'], batch_size=32, verbose=0)
    return model

def predict(kind, model, x):
    if kind == 'DNN': return model.predict(x.astype('float32'), verbose=0).ravel()
    return model.predict(x).ravel()

def validation_score(family, kind, config, split):
    train, hold, inner, val = split
    pipe = pipe_for(config, len(inner))
    xtr = pipe.fit_transform(inner[FEATS]); xva = pipe.transform(val[FEATS])
    ytr = inner.average_voltage.to_numpy(); yva = val.average_voltage.to_numpy()
    model = fit_model(kind, config, xtr, ytr)
    pred = predict(kind, model, xva)
    score = float(mean_absolute_error(yva, pred))
    if not np.isfinite(score): raise ValueError('Nonfinite validation MAE')
    return score, float(r2_score(yva, pred))

def final_evaluate(family, kind, config, split, na, save_models=False):
    train, hold, *_ = split
    pipe = pipe_for(config, len(train))
    xx = pipe.fit_transform(train[FEATS]); hx = pipe.transform(hold[FEATS]); nx = pipe.transform(na[FEATS])
    yt = train.average_voltage.to_numpy(); yh = hold.average_voltage.to_numpy(); yn = na.average_voltage.to_numpy()
    model = fit_model(kind, config, xx, yt)
    pt, ph, pn = [predict(kind, model, x) for x in (xx, hx, nx)]
    if save_models:
        stem = f'{family.lower()}_{kind.lower()}'
        joblib.dump(pipe, OUT / f'{stem}_preprocessor.joblib')
        if kind == 'DNN': model.save(OUT / f'{stem}.keras')
        else: joblib.dump(model, OUT / f'{stem}.joblib')
        for set_name, frame, true, est in [('H', hold, yh, ph), ('Na', na, yn, pn)]:
            result = pd.DataFrame({'actual_voltage':true,'predicted_voltage':est,'abs_error':abs(true-est)})
            for col in ('battery_id', 'working_ion', 'source_row'):
                if col in frame: result[col] = frame[col].to_numpy()
            result.to_csv(OUT / f'{stem}_{set_name}_predictions.csv', index=False)
    return [float(mean_absolute_error(yt,pt)), float(mean_absolute_error(yh,ph)), float(r2_score(yh,ph)),float(mean_absolute_error(yn,pn)),float(r2_score(yn,pn))]

KEYS = ['Train_MAE','H_MAE','H_R2','Na_MAE','Na_R2']
TRIAL_LOG = []
CHECKPOINT_LOG = []

def print_scoreboard(stage, best, splits, na, save_models=False):
    print(f'\n========== {stage}: SIX MODEL RESULTS ==========')
    data = []
    for family in ['Li', 'Mixed']:
        for kind in ['DNN', 'SVR', 'KRR']:
            key = (family,kind)
            scores = final_evaluate(family, kind, best[key]['config'], splits[family], na, save_models)
            name = f'{family} {kind}'
            row = {'Stage':stage,'Model':name,'Validation_MAE':best[key]['mae'],'Validation_R2':best[key]['r2'], **dict(zip(KEYS, scores)), 'Prior_Split_H_MAE':OLD_RESULTS[name][1], 'Prior_Split_H_R2':OLD_RESULTS[name][2], 'Prior_Split_Na_MAE':OLD_RESULTS[name][3], 'Prior_Split_Na_R2':OLD_RESULTS[name][4]}
            data.append(row)
            CHECKPOINT_LOG.append(row)
    pd.DataFrame(data).to_csv(OUT / f'{stage.replace(" ", "_").lower()}_six_models.csv', index=False)
    print(pd.DataFrame(data)[['Model','Train_MAE','H_MAE','H_R2','Na_MAE','Na_R2','Validation_MAE']].round(4).to_string(index=False), flush=True)
    pd.DataFrame(CHECKPOINT_LOG).to_csv(OUT / 'all_stage_six_model_metrics.csv',index=False)


def candidates_for(stage, kind, current):
    # All modifications are tested against the current best config and retained only if better.
    if stage == 'scaling': return [{**current,'scaler':'standard'}] if current['scaler']=='minmax' else []
    if stage == 'PCA': return [{**current,'pca':p} for p in PCA_COMPONENTS if p!=current['pca']]
    if stage == 'model tuning':
        if kind == 'SVR': return [{**current,'C':c,'gamma':g,'epsilon':e} for c,g,e in [(10,.01,.1),(100,.005,.1),(100,.02,.1),(100,.01,.05),(30,'scale',.1)]]
        if kind == 'KRR': return [{**current,'alpha':a,'gamma':g} for a,g in [(.001,.01),(.03,.02),(.1,.01),(.01,.005),(.003,.02)]]
        return [{**current, 'epochs':e} for e in [30,50,120] if e!=current['epochs']]
    if stage == 'DNN architecture':
        if kind != 'DNN': return []
        return [
           {**current, 'optimizer':'adam'},
           {**current, 'hidden':[80,40], 'dropout':[.15,.08], 'optimizer':'adam'},
           {**current, 'hidden':[128,64], 'dropout':[.2,.1], 'optimizer':'adam'},
           {**current, 'hidden':[60,30], 'dropout':[.1,.05], 'optimizer':'adam'},
           {**current, 'l2':.00001,'optimizer':'adam'},
        ]
    return []

def main():
    source, na = load_data()
    split = datasets(source)
    best = {}
    baseline = {'scaler':'minmax','pca':80}
    defaults = {
      'DNN': {'epochs':100,'optimizer':'rmsprop','lr':.001,'l2':.0001,'hidden':[60,30],'dropout':[.25,.10]},
      'SVR': {'C':100,'gamma':.01,'epsilon':.1},
      'KRR': {'alpha':.01,'gamma':.02},
    }
    print('\nOriginal historical results: comparison reference only; DIFFERENT random split, not directly comparable.')
    print(pd.DataFrame([dict(Model=k,**dict(zip(KEYS,v))) for k,v in OLD_RESULTS.items()]).round(4).to_string(index=False),flush=True)
    for family in ION_GROUPS:
        for kind in defaults:
            config = {**baseline,**deepcopy(defaults[kind])}
            m,r = validation_score(family,kind,config,split[family])
            best[family,kind]={'config':config,'mae':m,'r2':r}
            TRIAL_LOG.append({'Stage':'baseline','Model':f'{family} {kind}','Fix':'Original C001 baseline, group-safe','Validation_MAE':m,'Validation_R2':r,'Change_vs_Previous_%':0,'Decision':'BASELINE','Config':json.dumps(config)})
            print(f'Baseline {family} {kind}: validation MAE {m:.4f}, R2 {r:.4f}',flush=True)
    if CHECKPOINT_EACH_STAGE: print_scoreboard('baseline',best,split,na)
    for stage in ['scaling','PCA','model tuning','DNN architecture']:
        print(f'\n===== TESTING FIX: {stage} =====',flush=True)
        for family in ION_GROUPS:
            for kind in defaults:
                key = family, kind
                old = deepcopy(best[key]); winning = deepcopy(old)
                proposals = candidates_for(stage,kind,old['config'])
                for cfg in proposals:
                    try:
                        mae,r2 = validation_score(family,kind,cfg,split[family])
                        improvement = 100*(old['mae']-mae)/old['mae']
                        decision = 'REJECTED'
                        if mae < winning['mae']*(1-MIN_REL_IMPROVEMENT):
                            winning = {'config':cfg,'mae':mae,'r2':r2}
                            decision = 'ACCEPTED_CURRENT_BEST'
                        record = {'Stage':stage,'Model':f'{family} {kind}','Fix':json.dumps(cfg), 'Validation_MAE':mae,'Validation_R2':r2,'Change_vs_Previous_%':improvement,'Decision':decision,'Config':json.dumps(cfg)}
                        TRIAL_LOG.append(record)
                        print(f'{family} {kind} {stage}: {mae:.4f} (vs prior {old["mae"]:.4f}, {improvement:+.2f}%) -> {decision}',flush=True)
                    except Exception as e:
                        TRIAL_LOG.append({'Stage':stage,'Model':f'{family} {kind}','Fix':json.dumps(cfg),'Decision':'ERROR','Error':repr(e)})
                        print(f'ERROR {family} {kind} {stage}: {e}',flush=True)
                    pd.DataFrame(TRIAL_LOG).to_csv(OUT/'all_fix_trials.csv',index=False)
                best[key] = winning
        # Diagnostic only. No choice can depend on H or Na performance.
        if CHECKPOINT_EACH_STAGE:
            print_scoreboard(stage,best,split,na)
    print_scoreboard('FINAL_BEST_VALIDATION',best,split,na,save_models=True)
    pd.DataFrame(TRIAL_LOG).to_csv(OUT/'all_fix_trials.csv',index=False)
    with open(OUT/'winning_configurations.json','w') as f:
        json.dump({f'{fam}_{kind}': data for (fam,kind), data in best.items()},f,indent=2)
    report = ["C002: Selected by lowest group-aware VALIDATION MAE; held-out H and Na are diagnostic only.","Historical numbers used different splits and must NOT be ranked head-to-head.","A rejected candidate means it failed on the development validation split; not proof it is universally poor."]
    for (fam,kind),data in best.items(): report.append(f'{fam} {kind}: validation MAE={data["mae"]:.5f}, R2={data["r2"]:.5f}, config={data["config"]}')
    (OUT/'SUMMARY.txt').write_text('\n'.join(report),encoding='utf-8')
    print('\nALL DONE. Results:',OUT,flush=True)

if __name__ == '__main__': main()
