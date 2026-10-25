"""Fetch pinned model components with restartable byte ranges and SHA256 verification."""

from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_REV = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
IP = "h94/IP-Adapter"
IP_REV = "018e402774aeeddd60609b4ecdb7e298259dc729"
CHUNK = 8 * 1024**2


def snapshot(repo, rev, patterns):
    cache = Path(os.getenv("VISION_MODEL_CACHE", ROOT / ".cache"))
    path = cache / "pinned" / repo.replace("/", "--") / rev
    path.mkdir(parents=True, exist_ok=True)
    mark = path / "verified.json"
    if (
        mark.exists()
        and all(
            (path / k).stat().st_size == v
            for k, v in json.loads(mark.read_text()).items()
            if (path / k).exists()
        )
        and all((path / k).exists() for k in json.loads(mark.read_text()))
    ):
        return path
    url = f"https://huggingface.co/api/models/{repo}/revision/{rev}"
    r = requests.get(url, params={"blobs": "true"}, timeout=30)
    r.raise_for_status()
    files = {}
    for item in r.json()["siblings"]:
        name = item["rfilename"]
        if not any(fnmatch(name, pattern) for pattern in patterns):
            continue
        dest = path / name
        size = item["size"]
        files[name] = size
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == size:
            continue
        u = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
        if "lfs" not in item:
            small = requests.get(u, timeout=40)
            small.raise_for_status()
            if len(small.content) != size:
                raise ValueError("Unexpected config size")
            dest.write_bytes(small.content)
            continue
        digest = item["lfs"]["sha256"]
        hf = (
            Path(os.getenv("HF_HOME", str(ROOT / ".cache" / "huggingface")))
            / "hub"
            / ("models--" + repo.replace("/", "--"))
        )
        cached = hf / "snapshots" / rev / name
        if cached.is_file() and cached.stat().st_size == size:
            with cached.open("rb") as src:
                valid = hashlib.file_digest(src, "sha256").hexdigest() == digest
            if valid:
                try:
                    os.link(cached, dest)
                except OSError:
                    import shutil

                    shutil.copyfile(cached, dest)
                continue
        parts = dest.with_suffix(".ranges")
        parts.mkdir(exist_ok=True)
        partial = hf / "blobs" / (digest + ".incomplete")

        def one(start):
            end = min(start + CHUNK, size) - 1
            file = parts / str(start)
            if file.exists() and file.stat().st_size == end - start + 1:
                return file
            if partial.is_file() and partial.stat().st_size > end:
                with partial.open("rb") as previous:
                    previous.seek(start)
                    buf = previous.read(end - start + 1)
                if len(buf) == end - start + 1:
                    file.write_bytes(buf)
                    return file
            for attempt in range(5):
                try:
                    with requests.get(
                        u,
                        params={"seg": start},
                        headers={"Range": f"bytes={start}-{end}"},
                        timeout=(15, 60),
                        stream=True,
                    ) as resp:
                        resp.raise_for_status()
                        if (
                            resp.headers.get("Content-Range")
                            != f"bytes {start}-{end}/{size}"
                        ):
                            raise IOError("Unexpected range")
                        tmp = file.with_suffix(".partial")
                        with tmp.open("wb") as f:
                            for buf in resp.iter_content(1024**2):
                                f.write(buf)
                    if tmp.stat().st_size != end - start + 1:
                        raise IOError("Incomplete model part")
                    tmp.replace(file)
                    return file
                except (requests.RequestException, IOError):
                    if attempt == 4:
                        raise
                    time.sleep(attempt + 2)

        with ThreadPoolExecutor(max_workers=5) as workers:
            chunks = list(workers.map(one, range(0, size, CHUNK)))
        temp = dest.with_suffix(".assembling")
        sha = hashlib.sha256()
        with temp.open("wb") as out:
            for file in chunks:
                buf = file.read_bytes()
                sha.update(buf)
                out.write(buf)
        if sha.hexdigest() != digest:
            raise IOError("Invalid model hash: " + name)
        temp.replace(dest)
        for file in chunks:
            file.unlink()
        parts.rmdir()
        print(f"Verified {name} ({size:,} bytes)", flush=True)
    mark.write_text(json.dumps(files, indent=2))
    return path


def obtain(manifest_name, override):
    manifest = json.loads((ROOT/"configs"/manifest_name).read_text(encoding="utf-8-sig"))
    files = manifest["files"]
    directory = os.environ.get(override)
    path = Path(directory) if directory else snapshot(manifest["repo"],manifest["revision"],
                                                     [item["rfilename"] for item in files])
    for item in files:
        file = path/item["rfilename"]
        if not file.is_file() or file.stat().st_size != item["size"]:
            raise ValueError(f"Missing or incomplete pinned component {file}")
        if "lfs" in item:
            with file.open("rb") as stream:
                actual = hashlib.file_digest(stream,"sha256").hexdigest()
            expected = item["lfs"]["sha256"]
        else:
            payload = file.read_bytes()
            actual = hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()
            expected = item["blobId"]
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {item['rfilename']}")
    return path


def models():
    ip = obtain("ip-files.json","P07_IP_ROOT")
    sd = obtain("sd15-files.json","P07_SD15_ROOT")
    return sd, ip


if __name__ == "__main__":
    models()
