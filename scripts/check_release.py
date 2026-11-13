"""Check courses-only scope and checksums; optionally create a clean archive."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
IGNORE={'.git','__pycache__','.venv','.DS_Store'}


def collect():
    allowed={'README.md','CONTRIBUTING.md','SECURITY.md','THIRD_PARTY_NOTICES.md','CHANGELOG.md','LICENSE',
             'VERSION','.gitignore','RELEASE_MANIFEST.json','site','scripts','tests','.github'}
    extra={p.name for p in ROOT.iterdir()}-allowed-IGNORE
    if extra:raise ValueError('Unexpected root entries: '+str(sorted(extra)))
    records={}
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if any(part in IGNORE for part in relative.parts):continue
        if p.is_symlink():raise ValueError('Symlink: '+str(relative))
        if not p.is_file() or p.name=='RELEASE_MANIFEST.json':continue
        if p.suffix not in {'.html','.css','.js','.json','.md','.py','.yml'} and p.name not in {'LICENSE','VERSION','.gitignore','.nojekyll'}:
            raise ValueError('Unexpected file: '+str(relative))
        text=p.read_text()
        if re.search(r'/(?:Users|home)/[A-Za-z0-9._-]+/|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|ghp_[A-Za-z0-9]{30,}',text):
            raise ValueError('Possible private path or credential: '+str(relative))
        records[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',action='store_true')
    parser.add_argument('--zip',type=Path)
    args=parser.parse_args();records=collect();version=(ROOT/'VERSION').read_text().strip()
    manifest=ROOT/'RELEASE_MANIFEST.json'
    if args.write_manifest:
        manifest.write_text(json.dumps({'schema':2,'version':version,'scope':'home-and-five-courses','files':records},indent=2)+'\n')
    else:
        expected=json.loads(manifest.read_text())
        if expected['version']!=version or expected['files']!=records:raise ValueError('Release files or version differ from the reviewed manifest.')
    if args.zip:
        if args.zip.exists():raise ValueError('Choose a new archive filename.')
        with zipfile.ZipFile(args.zip,'w',zipfile.ZIP_DEFLATED) as z:
            for name in [*records,'RELEASE_MANIFEST.json']:
                z.write(ROOT/name,'machine-learning-study-lab/'+name)
    print(json.dumps({'status':'PASS','version':version,'scope':'home-and-five-courses','files':len(records)},indent=2))


if __name__=='__main__':main()
