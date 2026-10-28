"""Stream paper assets to a temporary GitHub Actions runner; never save the tar."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen

URL = 'https://drive.usercontent.google.com/download?id=1KFtL_6QT1Yt32jjCPHJgTcOROGFWf61v&export=download&confirm=t'
EXPECTED_BYTES = 7481497600
PREFIXES = ('data/', 'recons/', 'reference_results/', 'checkpoints/lactos/')


class HashReader:
    def __init__(self, response):
        self.response = response
        self.sha = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        chunk = self.response.read(size)
        self.sha.update(chunk)
        self.bytes += len(chunk)
        return chunk


def download(destination):
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Dataset download is restricted to GitHub Actions; local download is disabled')
    destination.mkdir(parents=True, exist_ok=False)
    files, extracted_bytes = 0, 0
    with urlopen(URL, timeout=120) as response:
        if 'application/octet-stream' not in response.headers.get('Content-Type', ''):
            raise RuntimeError('Asset link did not return the archive')
        reader = HashReader(response)
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            for member in tar:
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'LACToS_assets':
                    raise RuntimeError(f'Unsafe archive path: {member.name}')
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError(f'Unexpected archive member: {member.name}')
                relative = Path(*path.parts[1:])
                if not member.isfile() or not relative.as_posix().startswith(PREFIXES):
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                files += 1
                extracted_bytes += member.size
                if files % 500 == 0:
                    print(f'Extracted {files} selected files ({extracted_bytes / 1e9:.2f} GB)', flush=True)
        while reader.read(8 * 1024 * 1024):
            pass
    if reader.bytes != EXPECTED_BYTES:
        raise RuntimeError(f'Incomplete or changed archive: {reader.bytes} bytes')
    metadata = {'source_url': URL, 'downloaded_bytes': reader.bytes, 'sha256': reader.sha.hexdigest(),
                'sha256_status': 'Computed on GitHub Actions; authors do not publish a checksum in BUNDLE.md',
                'extracted_files': files, 'extracted_bytes': extracted_bytes, 'included_prefixes': PREFIXES,
                'archive_saved': False, 'execution': 'GitHub Actions ephemeral runner'}
    (destination / 'asset_download.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    download(p.parse_args().destination.resolve())
