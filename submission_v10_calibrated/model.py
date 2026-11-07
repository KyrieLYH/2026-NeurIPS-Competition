from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import joblib
from threadpoolctl import threadpool_limits
import predai.calibrated_v10
_MODEL=None
def predict(input: list[dict],labeled: list | None=None)->float:
    global _MODEL
    with threadpool_limits(limits=1):
        if _MODEL is None:_MODEL=joblib.load(Path(__file__).resolve().parent/'artifacts'/'ensemble.joblib')
        return float(_MODEL.predict_many([input],labeled)[0])
