"""Isolated package checks for preprocessing, acquisition behavior, and parity."""
from pathlib import Path
import sys,zipfile,tempfile,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/robust-v5-20261028'
CHECK=r'''
from pathlib import Path
import sys,copy,math,time,json
sys.path.insert(0,sys.argv[1])
import model
s=dict.fromkeys(('normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra'),'')
s.update(normalized_name='unseen-7B',provider='test')
it={'benchmark_id':'anonymous','item_content':'A question?','item_features':'task=math','interactors':''}
pair=[s,it];other=dict(s,normalized_name='another-7B')
for n in [0,1,3,7,15,31]:
    own=[[[dict(s),dict(it,item_content=f'own {j}')],j%2] for j in range(n)]
    peer=[[[dict(other),dict(it,item_content=f'peer {j}')],1] for j in range(n)]
    evidence=own+peer;before=copy.deepcopy([pair,evidence])
    p=model.predict(pair,evidence)
    assert type(p)is float and math.isfinite(p) and 0<=p<=1
    assert p==model.predict(pair,evidence)
    assert before==[pair,evidence]
    assert model.predict(pair,evidence+[[[other,dict(it,benchmark_id='unrelated')],0]])==p
base=model.predict(pair,[])
assert base==model.predict([s,dict(it,benchmark_id='new-anonymous-id')],[])
prefix='A'+'z'*11999
assert model.predict([s,dict(it,item_content=prefix)],[])==model.predict([s,dict(it,item_content=prefix+' suffix '*10000)],[])
assert model.predict([s,dict(it,item_features='{"task":"math"}')],[])==base
own_positive=[[[dict(s),dict(it,item_content=f'known {j}')],1] for j in range(31)]
own_negative=[[[dict(s),dict(it,item_content=f'known {j}')],0] for j in range(31)]
assert model.predict(pair,own_positive)>model.predict(pair,own_negative)
assert model.predict([{},{}],[])==model.predict([{},{}],[])
for input in [[{},{}],[s,dict(it,benchmark_id='')]]:
    assert math.isfinite(model.predict(input,[]))
start=time.perf_counter()
for _ in range(100):model.predict(pair,evidence)
result={'status':'PASS','checks':['isolated ZIP loading','6 label budgets','determinism','no input mutation','benchmark isolation','benchmark ID not a feature','long-text preprocessing parity','JSON attributes','label adaptation','missing fields'],'100_call_seconds':time.perf_counter()-start}
print(json.dumps(result))
'''

def main():
    archive=OUT/'submission_v5_robust.zip'
    with tempfile.TemporaryDirectory(prefix='verify-',dir=OUT) as tmp:
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert z.testzip() is None
            z.extractall(tmp)
        proc=subprocess.run([sys.executable,'-I','-c',CHECK,tmp],cwd=tmp,check=True,capture_output=True,text=True)
        print(proc.stdout,flush=True)
        result=json.loads(proc.stdout)
        (OUT/'package_check.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
