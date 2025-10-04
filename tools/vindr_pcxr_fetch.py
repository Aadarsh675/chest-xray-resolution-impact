#!/usr/bin/env python3
# tools/vindr_pcxr_fetch.py
"""
Fast ViNDr-PCXR fetcher with verification and resume.

- Reads SHA256SUMS.txt (downloading if needed).
- Verifies which files are missing/corrupt.
- Uses a single aria2c job list for high-throughput parallel + segmented downloads.
- Optionally stages to local disk, then rsyncs to Google Drive (faster).
- Safe to re-run; already-verified files are skipped.

Example:
  python tools/vindr_pcxr_fetch.py \
    --dest "/content/drive/MyDrive/vindr_pcxr/1.0.0" \
    --user <USER> --password <PASS> \
    --max-jobs 32 --per-file-conns 16 --segments 16 \
    --stage-dir "/content/vindr_staging"
"""

import argparse
import hashlib
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

DEFAULT_BASE_URL = "https://physionet.org/files/vindr-pcxr/1.0.0"


def sha256sum(fp: str) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_wget(url: str, out_dir: str, user: str, password: str) -> int:
    """Small helper just to get SHA256SUMS.txt if needed."""
    cmd = [
        "wget", "-q", "-N", "-c",
        "--user", user, "--password", password,
        "-P", out_dir, url,
    ]
    return subprocess.run(cmd).returncode


def ensure_sha_file(dest_dir: str, base_url: str, user: str, password: str) -> str:
    sha_path = os.path.join(dest_dir, "SHA256SUMS.txt")
    if not os.path.exists(sha_path):
        print("📥 Fetching SHA256SUMS.txt …")
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        rc = run_wget(f"{base_url}/SHA256SUMS.txt", dest_dir, user, password)
        if rc != 0 or not os.path.exists(sha_path):
            raise FileNotFoundError("Failed to download SHA256SUMS.txt. Check credentials/connectivity.")
    return sha_path


def parse_sha_file(sha_path: str) -> List[Tuple[str, str]]:
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
    missing, corrupt = [], []
    for h, rel in entries:
        local = os.path.join(dest_dir, rel)
        if not os.path.exists(local):
            missing.append(rel)
            continue
        try:
            if sha256sum(local).lower() != h.lower():
                corrupt.append(rel)
        except Exception:
            corrupt.append(rel)
    return missing, corrupt


def write_aria2_list(
    rel_paths: List[str],
    base_url: str,
    out_dir: str,
    user: str,
    password: str,
    list_path: str
) -> None:
    """
    Write an aria2 input list where each file has:
      URL
       out=<filename>
       dir=<directory>
    """
    lines: List[str] = []
    for rel in rel_paths:
        url = f"{base_url}/{rel}"
        subdir = os.path.dirname(rel)
        fname = os.path.basename(rel)
        target_dir = os.path.join(out_dir, subdir)
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        lines.append(url)
        lines.append(f"  out={fname}")
        lines.append(f"  dir={target_dir}")
    with open(list_path, "w") as f:
        f.write("\n".join(lines))
    # Restrict permissions (contains structure but not creds)
    os.chmod(list_path, 0o600)


def run_aria2c(
    list_file: str,
    user: str,
    password: str,
    max_jobs: int,
    per_file_conns: int,
    segments: int
) -> int:
    """
    Run aria2c once over the whole job list.
    Notes:
      - --continue, --auto-file-renaming=false to keep names stable
      - --http-user/--http-passwd for authenticated endpoints
    """
    cmd = [
        "aria2c",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        "--enable-rpc=false",
        "--file-allocation=none",
        "--http-user", user,
        "--http-passwd", password,
        "-j", str(max_jobs),         # concurrent downloads
        "-x", str(per_file_conns),   # per-file connections
        "-s", str(segments),         # split each file into N segments
        "-i", list_file,
    ]
    print("🚀 Launching aria2c …")
    print("   ", " ".join(cmd))
    return subprocess.run(cmd).returncode


def rsync_stage_to_dest(stage_dir: str, dest_dir: str) -> int:
    """
    Sync local staged files to final destination. rsync is efficient for Drive writes.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-a", "--info=progress2",
        os.path.join(stage_dir, ""),  # trailing slash to copy contents
        os.path.join(dest_dir, ""),
    ]
    print("📦 Syncing staged files to destination …")
    return subprocess.run(cmd).returncode


def reverify(entries: List[Tuple[str, str]], dest_dir: str, subset: List[str] = None) -> List[str]:
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
    ap = argparse.ArgumentParser(description="Fast verify/resume ViNDr-PCXR download with aria2c.")
    ap.add_argument("--dest", required=True, type=str,
                    help="Final dataset directory (e.g., /content/drive/MyDrive/vindr_pcxr/1.0.0)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, type=str)
    ap.add_argument("--user", required=True, type=str, help="PhysioNet username")
    ap.add_argument("--password", default=os.environ.get("PHYSIONET_PASSWORD", ""), type=str,
                    help="PhysioNet password or env PHYSIONET_PASSWORD")
    ap.add_argument("--max-jobs", default=32, type=int, help="Global concurrent downloads")
    ap.add_argument("--per-file-conns", default=16, type=int, help="Connections per file (-x)")
    ap.add_argument("--segments", default=16, type=int, help="Splits per file (-s)")
    ap.add_argument("--verify-only", action="store_true", help="Only verify files; do not download")
    ap.add_argument("--stage-dir", default="", type=str,
                    help="Optional local staging dir (e.g., /content/vindr_staging). If empty, download directly to --dest.")
    args = ap.parse_args()

    if not args.password:
        print("❌ No password provided. Use --password or set PHYSIONET_PASSWORD.", file=sys.stderr)
        sys.exit(2)

    dest_dir = args.dest
    stage_dir = args.stage_dir.strip() or dest_dir  # default: no staging (direct to dest)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    Path(stage_dir).mkdir(parents=True, exist_ok=True)

    # Ensure SHA file in DEST (checksum reference lives next to final dataset)
    sha_path = ensure_sha_file(dest_dir, args.base_url, args.user, args.password)

    # Parse + verify current state (in DEST)
    entries = parse_sha_file(sha_path)
    print(f"🧾 Found {len(entries)} entries in SHA256SUMS.txt")

    missing, corrupt = identify_needed(entries, dest_dir)
    ok_count = len(entries) - len(missing) - len(corrupt)
    print(f"📦 Status: {ok_count} OK, {len(missing)} missing, {len(corrupt)} corrupt")

    if args.verify_only:
        print("✅ Verification complete (no downloads requested).")
        sys.exit(0)

    # Remove corrupt files in DEST so they will be redownloaded
    for rel in corrupt:
        try:
            os.remove(os.path.join(dest_dir, rel))
        except FileNotFoundError:
            pass

    need = missing + corrupt
    if not need:
        print("✅ Dataset already complete — nothing to download.")
        sys.exit(0)

    # Build aria2 list file AGAINST STAGE DIR (fast local disk recommended)
    list_path = os.path.join(stage_dir, "_aria2_joblist.txt")
    write_aria2_list(need, args.base_url, stage_dir, args.user, args.password, list_path)

    # Download to stage (or direct dest if stage==dest)
    rc = run_aria2c(list_path, args.user, args.password, args.max_jobs, args.per_file_conns, args.segments)
    if rc != 0:
        print("⚠️ aria2c exited with non-zero status. You can re-run to resume.", file=sys.stderr)

    # If we staged to local, sync to final destination
    if os.path.abspath(stage_dir) != os.path.abspath(dest_dir):
        rc_sync = rsync_stage_to_dest(stage_dir, dest_dir)
        if rc_sync != 0:
            print("⚠️ rsync reported errors while syncing to destination.", file=sys.stderr)

    # Re-verify only the files we attempted
    still_bad = reverify(entries, dest_dir, subset=need)

    if still_bad:
        print(f"⚠️ {len(still_bad)} file(s) still mismatched after download:")
        for rel in still_bad[:20]:
            print("   -", rel)
        if len(still_bad) > 20:
            print("   …")
        print("ℹ️ You can re-run this script to resume the remaining files.")
    else:
        print("🎉 All requested files verified successfully.")


if __name__ == "__main__":
    main()
