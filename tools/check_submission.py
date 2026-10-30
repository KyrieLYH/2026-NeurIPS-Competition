"""Run the original isolated inference contract checks on a specified ZIP."""
from pathlib import Path
import argparse,sys,zipfile,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
from scripts.check_v5 import CHECK

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',nargs='?',type=Path,default=ROOT/'original_submission/submission_v10_calibrated.zip')
    a=ap.parse_args()
    # Isolated subprocess only sees the archive, never source/ or historical models.
    with tempfile.TemporaryDirectory(prefix='v10-check-') as tmp:
        with zipfile.ZipFile(a.archive) as z:
            assert z.testzip() is None
            for n in z.namelist():
                if Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n:raise ValueError(n)
            z.extractall(tmp)
        p=subprocess.run([sys.executable,'-I','-c',CHECK,tmp],cwd=tmp,capture_output=True,text=True)
        print(p.stdout,end='')
        if p.stderr:print(p.stderr,file=sys.stderr,end='')
        if p.returncode:raise SystemExit(p.returncode)
if __name__=='__main__':main()
