#!/usr/bin/env python3
"""
fast_download.py — Fast, proxy-bypassing multi-connection HTTP Range downloader.
Bypasses local slow proxies (strips HTTP_PROXY/HTTPS_PROXY/ALL_PROXY, sets NO_PROXY=*),
downloads using 4-6 concurrent Range threads with 30s chunk timeout, 5s flushed
progress heartbeats, and a 20s <150 KB/s stall watchdog so URLs fail over rapidly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

for _k in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FastDownloader/1.1"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class StallAbort(RuntimeError):
    """Raised when download speed stays below threshold too long."""


class _SharedProgress:
    def __init__(self, label: str, total_bytes: Optional[int], stall_sec: float = 20.0, min_rate: float = 150 * 1024):
        self.label = label
        self.total_bytes = total_bytes
        self.stall_sec = stall_sec
        self.min_rate = min_rate
        self.lock = threading.Lock()
        self.downloaded = 0
        self.aborted = False
        self.start_t = time.time()
        self.last_log_t = self.start_t
        self.win_start_t = self.start_t
        self.win_bytes = 0

    def add(self, n: int) -> None:
        now = time.time()
        with self.lock:
            self.downloaded += n
            self.win_bytes += n
            win_dt = now - self.win_start_t
            if self.total_bytes and self.total_bytes >= 8 * 1024 * 1024 and win_dt >= self.stall_sec:
                rate = self.win_bytes / win_dt
                if rate < self.min_rate:
                    self.aborted = True
                    raise StallAbort(
                        f"Stall detected on {self.label}: {rate/1024:.1f} KB/s over {win_dt:.0f}s (<{self.min_rate/1024:.0f} KB/s)"
                    )
                self.win_start_t = now
                self.win_bytes = 0
            if now - self.last_log_t >= 5.0 and self.total_bytes and self.total_bytes >= 8 * 1024 * 1024:
                dt = max(0.01, now - self.start_t)
                mb = self.downloaded / (1024 * 1024)
                tot_mb = self.total_bytes / (1024 * 1024)
                spd = mb / dt
                print(f"[prog] {self.label}: {mb:.1f}/{tot_mb:.1f} MB ({spd:.2f} MB/s)", flush=True)
                self.last_log_t = now


def _probe_url(url: str, timeout: int = 25) -> tuple[str, Optional[int], bool, str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        method="HEAD",
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            cl = resp.headers.get("Content-Length")
            ar = (resp.headers.get("Accept-Ranges") or "").lower()
            ct = (resp.headers.get("Content-Type") or "").lower()
            size = int(cl) if cl and cl.isdigit() else None
            if size is not None:
                return final_url, size, ("bytes" in ar), ct
    except Exception:
        pass

    req2 = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Range": "bytes=0-0",
            "Accept-Encoding": "identity",
        },
        method="GET",
    )
    with _OPENER.open(req2, timeout=timeout) as resp:
        final_url = resp.geturl()
        status = getattr(resp, "status", 200)
        ct = (resp.headers.get("Content-Type") or "").lower()
        cr = resp.headers.get("Content-Range") or ""
        resp.read(1)
        if status == 206 and "/" in cr:
            total_str = cr.split("/")[-1].strip()
            size = int(total_str) if total_str.isdigit() else None
            return final_url, size, True, ct
        return final_url, None, False, ct


def _looks_like_html(head_bytes: bytes, content_type: str) -> bool:
    if "text/html" in content_type or "application/xhtml" in content_type:
        return True
    s = head_bytes.lstrip()[:256].lower()
    return s.startswith(b"<!doctype html") or s.startswith(b"<html")


def _download_range_chunk(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    timeout: int,
    retries: int,
    progress: _SharedProgress,
) -> int:
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        progress.add(expected)
        return expected

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        if progress.aborted:
            raise StallAbort("Aborted due to stall watchdog")
        try:
            existing = part_path.stat().st_size if part_path.exists() else 0
            if existing > expected:
                part_path.unlink(missing_ok=True)
                existing = 0
            elif existing > 0:
                progress.add(existing)
            req_start = start + existing
            if req_start > end:
                return expected
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Range": f"bytes={req_start}-{end}",
                    "Accept-Encoding": "identity",
                },
            )
            with _OPENER.open(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                ct = (resp.headers.get("Content-Type") or "").lower()
                if status not in (200, 206):
                    raise RuntimeError(f"HTTP {status} for range {req_start}-{end}")
                mode = "ab" if (existing > 0 and status == 206) else "wb"
                if mode == "wb" and req_start != start:
                    part_path.unlink(missing_ok=True)
                    raise RuntimeError("Server returned HTTP 200 instead of 206 for Range")
                first_chunk = True
                with open(part_path, mode) as f:
                    while True:
                        if progress.aborted:
                            raise StallAbort("Aborted due to stall watchdog")
                        buf = resp.read(256 * 1024)
                        if not buf:
                            break
                        if first_chunk and start == 0 and existing == 0:
                            if _looks_like_html(buf, ct):
                                raise ValueError("Downloaded payload looks like an HTML error page")
                        first_chunk = False
                        f.write(buf)
                        f.flush()
                        progress.add(len(buf))
            got = part_path.stat().st_size
            if got != expected:
                raise IOError(f"Chunk size mismatch: got {got}, expected {expected}")
            return got
        except StallAbort:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(min(1.5 * attempt, 4.0))
    raise RuntimeError(f"Chunk {start}-{end} failed after {retries} attempts: {last_err}")


def _download_single(
    url: str,
    tmp_path: Path,
    expected_size: Optional[int],
    ct_hint: str,
    timeout: int,
    retries: int,
    allow_html: bool,
    progress: _SharedProgress,
) -> int:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            tmp_path.unlink(missing_ok=True)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
            )
            total = 0
            first_chunk = True
            with _OPENER.open(req, timeout=timeout) as resp:
                ct = (resp.headers.get("Content-Type") or ct_hint).lower()
                cl = resp.headers.get("Content-Length")
                hdr_size = int(cl) if cl and cl.isdigit() else expected_size
                with open(tmp_path, "wb") as f:
                    while True:
                        buf = resp.read(256 * 1024)
                        if not buf:
                            break
                        if first_chunk and not allow_html and _looks_like_html(buf, ct):
                            raise ValueError("Payload is an HTML error page instead of target asset")
                        first_chunk = False
                        f.write(buf)
                        f.flush()
                        progress.add(len(buf))
                        total += len(buf)
                if hdr_size is not None and total != hdr_size:
                    raise IOError(f"Size mismatch: got {total}, header expected {hdr_size}")
            return total
        except StallAbort:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(min(1.5 * attempt, 4.0))
    raise RuntimeError(f"Single-stream download failed after {retries} attempts: {last_err}")


def download_file(
    url: str,
    dest: Path,
    workers: int = 6,
    timeout: int = 30,
    retries: int = 4,
    min_bytes: int = 1,
    allow_html: bool = False,
) -> dict:
    t0 = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)

    final_url, total_size, accept_ranges, ct = _probe_url(url, timeout=timeout)
    if not allow_html and ("text/html" in ct):
        raise ValueError(f"Refusing HTML response from {final_url} (Content-Type={ct})")

    if dest.exists() and total_size is not None and dest.stat().st_size == total_size and total_size >= min_bytes:
        return {
            "url": final_url,
            "dest": str(dest),
            "bytes": total_size,
            "elapsed_s": round(time.time() - t0, 2),
            "cached": True,
        }

    tmp_path = dest.with_name(dest.name + ".tmp")
    progress = _SharedProgress(dest.name, total_size)

    if accept_ranges and total_size is not None and total_size >= 8 * 1024 * 1024 and workers > 1:
        n_workers = max(1, min(workers, math.ceil(total_size / (4 * 1024 * 1024))))
        part_size = math.ceil(total_size / n_workers)
        ranges = []
        for idx in range(n_workers):
            start = idx * part_size
            end = min(total_size - 1, (idx + 1) * part_size - 1)
            if start <= end:
                part_path = dest.with_name(f"{dest.name}.part{idx}")
                ranges.append((idx, start, end, part_path))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as pool:
                futs = [
                    pool.submit(
                        _download_range_chunk,
                        final_url,
                        part_path,
                        start,
                        end,
                        timeout,
                        retries,
                        progress,
                    )
                    for _, start, end, part_path in ranges
                ]
                for fut in futs:
                    fut.result()

            tmp_path.unlink(missing_ok=True)
            with open(tmp_path, "wb") as out_f:
                for _, _, _, part_path in ranges:
                    with open(part_path, "rb") as pf:
                        while True:
                            chunk = pf.read(1024 * 1024)
                            if not chunk:
                                break
                            out_f.write(chunk)
            for _, _, _, part_path in ranges:
                part_path.unlink(missing_ok=True)
        except StallAbort:
            for _, _, _, part_path in ranges:
                part_path.unlink(missing_ok=True)
            raise
        except Exception as e:
            for _, _, _, part_path in ranges:
                part_path.unlink(missing_ok=True)
            print(f"[warn] Parallel Range failed ({e}); falling back to single stream...", file=sys.stderr, flush=True)
            _download_single(final_url, tmp_path, total_size, ct, timeout, retries, allow_html, _SharedProgress(dest.name, total_size))
    else:
        _download_single(final_url, tmp_path, total_size, ct, timeout, retries, allow_html, progress)

    got_size = tmp_path.stat().st_size
    if total_size is not None and got_size != total_size:
        raise IOError(f"Final file size mismatch for {dest}: got {got_size}, expected {total_size}")
    if got_size < min_bytes:
        raise IOError(f"Downloaded file too small ({got_size} < {min_bytes} bytes): {dest}")
    if not allow_html:
        with open(tmp_path, "rb") as f:
            head = f.read(512)
        if _looks_like_html(head, ct):
            raise ValueError(f"Downloaded file starts with HTML error content: {dest}")

    os.replace(tmp_path, dest)
    elapsed = max(0.01, time.time() - t0)
    mb_s = (got_size / (1024 * 1024)) / elapsed
    return {
        "url": final_url,
        "dest": str(dest),
        "bytes": got_size,
        "elapsed_s": round(elapsed, 2),
        "speed_mb_s": round(mb_s, 2),
        "cached": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Proxy-bypassing parallel HTTP Range downloader")
    ap.add_argument("url", help="Primary direct URL")
    ap.add_argument("dest", help="Output local file path")
    ap.add_argument("--fallback-url", action="append", default=[], help="Fallback URL(s) if primary fails")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent range workers (default: 6)")
    ap.add_argument("--timeout", type=int, default=30, help="Per-request socket timeout seconds")
    ap.add_argument("--retries", type=int, default=4, help="Retries per chunk/request")
    ap.add_argument("--min-bytes", type=int, default=16, help="Minimum expected file size")
    ap.add_argument("--allow-html", action="store_true", help="Allow HTML files")
    args = ap.parse_args()

    urls = [args.url] + list(args.fallback_url)
    dest = Path(args.dest)
    for u in urls:
        try:
            res = download_file(
                u,
                dest,
                workers=args.workers,
                timeout=args.timeout,
                retries=args.retries,
                min_bytes=args.min_bytes,
                allow_html=args.allow_html,
            )
            mb = res["bytes"] / (1024 * 1024)
            tag = "CACHED" if res["cached"] else f"{res['speed_mb_s']} MB/s"
            print(f"[OK] {dest.name} ({mb:.2f} MB, {res['elapsed_s']}s, {tag})", flush=True)
            return 0
        except Exception as e:
            print(f"[ERR] {u} -> {e}", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
