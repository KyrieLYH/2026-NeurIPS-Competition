"""Convenience entry for the chosen 150k full-public base recipe.

Uses historical data/sampling/features unchanged. Requires the original public
Measurement Data Bank snapshot, which is NOT bundled. Not executed for this
handoff; exact weight regeneration depends on the dataset/order/environment.
"""
from pathlib import Path
import argparse,sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from scripts.finetune_current import binary_benchmarks,load_slim,attach_features
from predai.data_v2 import balanced_sample,to_pair
from predai.ensemble_v2 import Ensemble,pair_weights
from predai.calibrated_v10 import CalibratedEnsemble

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-data',type=Path,help='Public measurement-db root; reads tables, never traces')
    ap.add_argument('--sample-in',type=Path,help='Previously saved FINAL 150k feature-attached sample, if available')
    ap.add_argument('--sample-out',type=Path,help='Optionally persist the selected sample for future replay')
    ap.add_argument('--output-dir',required=True,type=Path,help='NEW directory only')
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.threads<1:raise ValueError('threads must be positive')
    expected=json.loads((ROOT/'evidence/base_training_metadata.json').read_text(encoding='utf-8'))
    if args.sample_in:
        frame=pd.read_parquet(args.sample_in);audit=None
    else:
        if not args.raw_data:ap.error('--raw-data or --sample-in required')
        benches,audit=binary_benchmarks(args.raw_data)
        if set(benches)!=set(expected['training_benchmarks']):raise ValueError('Snapshot benchmark set differs; do not claim selected-version reproduction')
        slim=load_slim(args.raw_data,benches)
        if len(slim)!=55047908:raise ValueError('Snapshot candidate row count differs')
        sample=balanced_sample(slim,150000,seed=0);del slim
        frame=attach_features(args.raw_data,sample);del sample
    if len(frame)!=150000 or set(frame.benchmark_id)!=set(expected['training_benchmarks']):raise ValueError('Wrong final sample')
    if args.sample_out:
        if args.sample_out.exists():raise FileExistsError(args.sample_out)
        args.sample_out.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(args.sample_out,index=False)
    args.output_dir.mkdir(parents=True)
    # Save sampled IDs; the historical entry did not preserve this manifest.
    ids=[k for k in ['benchmark_id','subject_id','item_id','response_id','trial'] if k in frame]
    frame[ids].to_parquet(args.output_dir/'sample_ids.parquet',index=False)
    pairs=[to_pair(row) for row in frame.to_dict('records')];started=time.perf_counter()
    with threadpool_limits(limits=args.threads):
        base=Ensemble(kappa=2.,n_jobs=args.threads).fit(pairs,frame.label.to_numpy(),pair_weights(frame))
    joblib.dump(base,args.output_dir/'base.joblib',compress=3)
    params=json.loads((ROOT/'runtime/artifacts/training.json').read_text(encoding='utf-8'))['parameters']
    wrapped=CalibratedEnsemble(base,params['a'],params['b'],False)
    joblib.dump(wrapped,args.output_dir/'ensemble.joblib',compress=3)
    meta={'scope':'Recreated fixed recipe; not proof of identical historic model weights',
        'train_rows':len(frame),'training_benchmarks':sorted(frame.benchmark_id.unique()),'sampling_seed':0,
        'tree_seed':42,'fit_seconds':time.perf_counter()-started,'parameters':params,'snapshot_audit':audit}
    (args.output_dir/'training.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='snapshot_audit'},indent=2))
if __name__=='__main__':main()
