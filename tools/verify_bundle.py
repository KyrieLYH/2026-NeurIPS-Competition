"""Standard-library-only manifest check. Does not import/unpickle a model."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    failures=[]
    for item in manifest['files']:
        p=ROOT/item['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:failures.append(item['path'])
    if failures:raise AssertionError(failures)
    original=ROOT/'original_submission/submission_v10_calibrated.zip'
    with zipfile.ZipFile(original) as z:
        for n in z.namelist():
            if (ROOT/'runtime'/n).read_bytes()!=z.read(n):raise AssertionError(n)
    print(json.dumps({'status':'PASS','manifest_files':len(manifest['files']),'runtime_members_identical':True}))
if __name__=='__main__':main()
