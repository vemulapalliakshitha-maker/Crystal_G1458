"""Filter research CSV to original project ion groups without Na/K leakage."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/raw/dataset/mp_data/data_2019_12_03.csv'
df=pd.read_csv(source)
if 'Unnamed: 0' in df: df=df.drop(columns=['Unnamed: 0'])
df=df.drop_duplicates().reset_index(drop=True)
need={'working_ion','average_voltage','battery_id','id_charge','id_discharge'}
if need-set(df):raise ValueError(f'Missing columns: {need-set(df)}')
train_ions=['Li','Ca','Mg','Zn','Al','Y']
remaining=df[df.working_ion.isin(train_ions)].copy()
na=df[df.working_ion.eq('Na')].copy()
assert not remaining.working_ion.isin(['Na','K']).any()
out=ROOT/'data/interim';out.mkdir(exist_ok=True,parents=True)
remaining.to_csv(out/'remaining_instances.csv',index=False)
na.to_csv(out/'na_instances.csv',index=False)
print(f'Raw={len(df)} training-source={len(remaining)} sodium-source={len(na)}')
print('Training ion counts:',remaining.working_ion.value_counts().to_dict())
