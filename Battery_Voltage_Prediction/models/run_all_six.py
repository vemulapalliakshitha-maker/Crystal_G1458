"""Run every standalone model; consolidate all six measured metrics."""
import subprocess,sys
from pathlib import Path
import pandas as pd
here=Path(__file__).resolve().parent
output=here.parent/'artifacts'/'frozen_models'
all_rows=[]
for name in ['li_dnn','li_svr','li_krr','mixed_dnn','mixed_svr','mixed_krr']:
    print(f'\n===== {name} =====',flush=True)
    subprocess.run([sys.executable,str(here/f'{name}.py')],check=True)
    current=output/'frozen_six_model_metrics.csv'
    if not current.exists(): raise FileNotFoundError(current)
    df=pd.read_csv(current)
    selected=df[df['Model'].str.lower().str.replace(' ','_',regex=False).eq(name)]
    if len(selected)!=1:raise ValueError(f'Expected one metric row for {name}; got {len(selected)}')
    all_rows.append(selected.iloc[0].to_dict())
    pd.DataFrame(all_rows).to_csv(output/'ALL_SIX_MEASURED_METRICS.csv',index=False)
print('\n=== ALL SIX MEASURED RESULTS ===')
print(pd.DataFrame(all_rows).round(4).to_string(index=False))
print('Saved:',output/'ALL_SIX_MEASURED_METRICS.csv')
