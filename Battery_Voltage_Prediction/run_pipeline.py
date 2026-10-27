"""Run stages: python run_pipeline.py prepare | train | all"""
from pathlib import Path
import subprocess,sys,argparse
p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','all']);a=p.parse_args()
R=Path(__file__).resolve().parent
prep=['scripts/01_prepare_records.py','scripts/02_extract_training_features.py','scripts/03_extract_sodium_features.py','scripts/04_prepare_group_safe.py']
training=['models/run_all_six.py']
for f in (prep if a.stage=='prepare' else training if a.stage=='train' else prep+training):
 print('\nRUN:',f,flush=True);subprocess.run([sys.executable,str(R/f)],cwd=R,check=True)
