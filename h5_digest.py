#!/usr/bin/env python3
"""Content digest of a replay_scaled.py HDF5 export, for the S0 shim gate.

A plain file hash never matches across runs: the export writes
``created_utc`` and gzip output can differ. This hashes what matters
instead: every dataset's name, dtype, shape and values, and every attribute
except the volatile ones, walked in sorted order.

    python h5_digest.py run_A.h5 run_B.h5        # one digest per file
    python h5_digest.py --diff run_A.h5 run_B.h5 # list differing objects
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import h5py
import numpy as np

VOLATILE_ATTRS = {"created_utc"}


def _items(h5: h5py.File) -> dict[str, bytes]:
    """Map each object path (and path@attr) to a byte digest of its content."""
    out: dict[str, bytes] = {}

    def attrs(path: str, obj) -> None:
        for k in sorted(obj.attrs):
            if k in VOLATILE_ATTRS:
                continue
            v = np.asarray(obj.attrs[k])
            out[f"{path}@{k}"] = hashlib.sha256(
                repr(v.dtype).encode() + repr(v.shape).encode() + v.tobytes()).digest()

    attrs("/", h5)

    def visit(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            v = obj[()]
            v = np.asarray(v)
            data = (b"\0".join(x if isinstance(x, bytes) else str(x).encode() for x in v.ravel())
                    if v.dtype == object else v.tobytes())
            out[name] = hashlib.sha256(
                repr(v.dtype).encode() + repr(v.shape).encode() + data).digest()
        attrs(name, obj)

    h5.visititems(visit)
    return out


def digest(path: str) -> str:
    with h5py.File(path, "r") as h5:
        items = _items(h5)
    h = hashlib.sha256()
    for k in sorted(items):
        h.update(k.encode() + b"\0" + items[k])
    return h.hexdigest()


def diff(a: str, b: str) -> list[str]:
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        ia, ib = _items(fa), _items(fb)
    return sorted(k for k in ia.keys() | ib.keys() if ia.get(k) != ib.get(k))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--diff", action="store_true", help="compare exactly two files")
    args = ap.parse_args()
    if args.diff:
        if len(args.files) != 2:
            ap.error("--diff takes two files")
        d = diff(*args.files)
        print("\n".join(d) if d else "identical")
        return 1 if d else 0
    for f in args.files:
        print(f"{digest(f)}  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
