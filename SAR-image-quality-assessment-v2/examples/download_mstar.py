#!/usr/bin/env python3
"""下载真实 MSTAR 数据（分片并行 + 重试 + 大小校验）。

来源：GitHub 仓库 jwcalder/MSTAR-Active-Learning 的 Data/SAR10{a,b,c}.npz
（SDMS 官方公开 MSTAR mixed-targets 的预处理结果，88×88 幅度+相位 float 切片）。

本脚本针对国内访问 GitHub 慢 / 断连的问题，用多线程分片（Range 请求）并行下载，
每片带重试，下载完按字节数校验。输出到 mstar_raw/。

运行：
    python examples/download_mstar.py
"""
import concurrent.futures as cf
import os
import time
import urllib.request

BASE = "https://raw.githubusercontent.com/jwcalder/MSTAR-Active-Learning/main/Data/"
# 文件名 -> 期望字节数（来自 GitHub API 的 size 字段）
FILES = {
    "SAR10a.npz": 89657078,
    "SAR10b.npz": 88463056,
    "SAR10c.npz": 89112182,
}
CHUNKS = 10
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mstar_raw")


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end, path, retries=8):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            return len(data)
        except Exception as exc:  # 断连重试
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def download_one(name, expected):
    url = BASE + name
    try:
        size = head_size(url)
    except Exception:
        size = expected
    chunk = (size + CHUNKS - 1) // CHUNKS
    parts = []
    print(f"[{name}] 大小 {size} 字节，{CHUNKS} 片并行下载...", flush=True)
    with cf.ThreadPoolExecutor(max_workers=CHUNKS) as ex:
        futs = []
        for i in range(CHUNKS):
            start = i * chunk
            end = min((i + 1) * chunk - 1, size - 1)
            p = os.path.join(OUT, f"{name}.part{i}")
            parts.append(p)
            futs.append(ex.submit(fetch_range, url, start, end, p))
        for f in futs:
            f.result()  # 任一失败会抛异常
    # 合并
    final = os.path.join(OUT, name)
    with open(final, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                out.write(f.read())
    for p in parts:
        os.remove(p)
    actual = os.path.getsize(final)
    ok = actual == size
    print(f"[{name}] {actual} / {size} 字节 {'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, size in FILES.items():
        final = os.path.join(OUT, name)
        if os.path.exists(final) and os.path.getsize(final) == size:
            print(f"[{name}] 已存在且大小正确，跳过")
            continue
        download_one(name, size)


if __name__ == "__main__":
    main()
