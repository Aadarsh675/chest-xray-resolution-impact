# tools/vindr_pcxr_fetch.py
"""
Verify and (re)download the ViNDr-PCXR dataset from PhysioNet using SHA256SUMS.txt.
- Skips files that already match the expected hash.
- Redownloads only missing/corrupt files.
- Supports parallel downloads via ProcessPoolExecutor.
- Uses wget with --user/--password (read from CLI args or PHYSIONET_PASSWORD env var).

Example:
  python tools/vindr_pcxr_fetch.py \
      --dest "/content/drive/MyDrive/vindr_pcxr/1.0.0" \
      --user YOUR_USERNAME \
      --password YOUR_PASSWORD \
      --max-workers 8
"""

import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_BASE_URL = "https://physionet.org/files/vindr-pcxr/1.0.0"


def sha256sum(fp: str) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_wget(url: str, out_dir: str, user: str, password: str, quiet: bool = False) -> int:
    """
    Download a single file to out_dir using wget with resume enabled.
    Returns the subprocess return code (0 == success).
    """
    cmd = [
        "wget",
        "-N",              # timestamping (download if newer or missing)
        "-c",              # continue / resume
        "--tries=3",
        "--user", user,
        "--password", password,
        "-P", out_dir,
        url,
    ]
    if quiet:
        cmd.insert(1, "-q")
    env = os.environ.copy()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode


def ensure_sha_file(dest_dir: str, base_url: str, user: str, password: str) -> str:
    """Ensure SHA256SUMS.txt exists locally; download if missing. Return its path."""
    sha_path = os.path.join(dest_dir, "SHA256SUMS.txt")
    if not os.path.exists(sha_path):
        print("📥 Fetching SHA256SUMS.txt …")
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        rc = run_wget(f"{base_url}/SHA256SUMS.txt", dest_dir, user, password, quiet=False)
        if rc != 0 or not os.path.exists(sha_path):
            raise FileNotFoundError("Failed to download SHA256SUMS.txt. Check credentials or connectivity.")
    return sha_path


def parse_sha_file(sha_path: str) -> List[Tuple[str, str]]:
    """Parse SHA256SUMS.txt into a list of (hash_hex, relative_path)."""
    entries: List[Tuple[str, str]] = []
    with open(sha_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or " " not in line:
                continue
            h, rel = line.split(None, 1)
            entries.append((h.strip(), rel.strip()))
    return entries


def identify_needed(entries: List[Tuple[str, str]], dest_dir: str) -> Tuple[List[str], List[str]]:
    """Return (missing_list, corrupt_list) of relative paths."""
    missing, corrupt = [], []
    for h, rel in entries:
        lp = os.path.join(dest_dir, rel)
        if not os.path.exists(lp):
            missing.append(rel)
            continue
        try:
            got = sha256sum(lp)
            if got.lower() != h.lower():
                corrupt.append(rel)
        except Exception:
            corrupt.append(rel)
    return missing, corrupt


# ---------- TOP-LEVEL WORKER (picklable) ----------
def _download_worker(rel: str, base_url: str, dest_dir: str, user: str, password: str, quiet: bool = False) -> Tuple[str, int]:
    """Single-file download worker for ProcessPoolExecutor."""
    url = f"{base_url}/{rel}"
    out_dir = os.path.join(dest_dir, os.path.dirname(rel))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rc = run_wget(url, out_dir, user, password, quiet=quiet)
    return rel, rc
# --------------------------------------------------


def parallel_download(
    rel_paths: List[str],
    base_url: str,
    dest_dir: str,
    user: str,
    password: str,
    max_workers: int,
) -> Dict[str, bool]:
    """Download the given relative paths in parallel. Returns rel_path -> success(bool)."""
    results: Dict[str, bool] = {}
    if not rel_paths:
        return results

    print(f"⬇️ Parallel downloading {len(rel_paths)} files with {max_workers} workers…")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_download_worker, rel, base_url, dest_dir, user, password, False)
                for rel in rel_paths]
        for fut in as_completed(futs):
            rel, rc = fut.result()
            ok = (rc == 0)
            results[rel] = ok
            print(f"   {'✅' if ok else '❌'} {rel}")
    return results


def reverify(entries: List[Tuple[str, str]], dest_dir: str, subset: List[str] = None) -> List[str]:
    """Re-verify a subset (or all) entries. Returns list of rel paths still bad."""
    still_bad: List[str] = []
    target = subset if subset is not None else [rel for _, rel in entries]
    expected = {rel: h for h, rel in entries}
    for rel in target:
        lp = os.path.join(dest_dir, rel)
        if not os.path.exists(lp):
            still_bad.append(rel)
            continue
        try:
            got = sha256sum(lp)
            if got.lower() != expected[rel].lower():
                still_bad.append(rel)
        except Exception:
            still_bad.append(rel)
    return still_bad


def main():
    ap = argparse.ArgumentParser(description="Verify / resume ViNDr-PCXR download using SHA256SUMS.txt (parallel).")
    ap.add_argument("--dest", type=str, required=True,
                    help="Local destination directory (e.g., /content/drive/MyDrive/vindr_pcxr/1.0.0)")
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                    help=f"Base URL (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--user", type=str, required=True, help="PhysioNet username")
    ap.add_argument("--password", type=str, default=os.environ.get("PHYSIONET_PASSWORD", ""),
                    help="PhysioNet password (or set env PHYSIONET_PASSWORD)")
    ap.add_argument("--max-workers", type=int, default=8, help="Parallel download workers")
    ap.add_argument("--verify-only", action="store_true", help="Only verify files; do not download")
    args = ap.parse_args()

    dest_dir = args.dest
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    if not args.password:
        print("❌ No password provided. Use --password or set PHYSIONET_PASSWORD env var.", file=sys.stderr)
        sys.exit(2)

    # Ensure checksum file exists
    sha_path = ensure_sha_file(dest_dir, args.base_url, args.user, args.password)

    # Parse entries
    entries = parse_sha_file(sha_path)
    print(f"🧾 Found {len(entries)} entries in SHA256SUMS.txt")

    # Identify missing/corrupt
    missing, corrupt = identify_needed(entries, dest_dir)
    ok_count = len(entries) - len(missing) - len(corrupt)
    print(f"📦 Status: {ok_count} OK, {len(missing)} missing, {len(corrupt)} corrupt")

    if args.verify_only:
        print("✅ Verification complete (no downloads requested).")
        sys.exit(0)

    # Delete corrupt files before re-download to avoid wget -N skip
    for rel in corrupt:
        lp = os.path.join(dest_dir, rel)
        try:
            os.remove(lp)
        except FileNotFoundError:
            pass

    need = missing + corrupt
    if not need:
        print("✅ Dataset already complete — no download needed.")
        sys.exit(0)

    # Download in parallel
    results = parallel_download(
        rel_paths=need,
        base_url=args.base_url,
        dest_dir=dest_dir,
        user=args.user,
        password=args.password,
        max_workers=args.max_workers,
    )

    # Re-verify only the ones we attempted
    to_check = [rel for rel, ok in results.items() if ok]
    still_bad = reverify(entries, dest_dir, subset=to_check)

    failed = [rel for rel, ok in results.items() if not ok]
    if still_bad:
        print(f"⚠️ {len(still_bad)} file(s) still mismatched after download:")
        for rel in still_bad[:20]:
            print("   -", rel)
        if len(still_bad) > 20:
            print("   …")
    if failed:
        print(f"❌ {len(failed)} file(s) failed to download:")
        for rel in failed[:20]:
            print("   -", rel)
        if len(failed) > 20:
            print("   …")

    if not still_bad and not failed:
        print("🎉 All requested files verified successfully.")
    else:
        print("ℹ️ You can re-run this script to resume any remaining files.")


if __name__ == "__main__":
    main()
