"""Package the selected runtime directory; no training or source substitution."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]

def package(runtime,output,reference):
    if output.exists():raise FileExistsError(output)
    with zipfile.ZipFile(reference) as z:
        names=z.namelist();members={name:z.read(name) for name in names}
    for name in names:
        p=runtime/name
        if not p.is_file():raise FileNotFoundError(p)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info=zipfile.ZipInfo(name,date_time=(2026,10,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;z.writestr(info,(runtime/name).read_bytes())
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert z.namelist()==names
        changed=[name for name in names if z.read(name)!=members[name]]
    result={'archive':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'same_member_bytes_as_original':not changed,'changed_members':changed,
            'note':'Archive timestamps/compression can differ. Original submitted artifact is in original_submission/.'}
    print(json.dumps(result,ensure_ascii=False,indent=2));return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime',type=Path,default=ROOT/'runtime')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/submission_v10_repacked.zip')
    ap.add_argument('--reference',type=Path,default=ROOT/'original_submission/submission_v10_calibrated.zip')
    a=ap.parse_args();package(a.runtime,a.output,a.reference)
if __name__=='__main__':main()
