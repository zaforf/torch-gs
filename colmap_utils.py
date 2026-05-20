import os
import struct

import numpy as np


def _read_points3d_bin(path):
    points = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            pid             = struct.unpack('<Q',   f.read(8))[0]
            xyz             = struct.unpack('<ddd', f.read(24))
            rgb             = struct.unpack('<BBB', f.read(3))
            _error          = struct.unpack('<d',   f.read(8))[0]
            track_len       = struct.unpack('<Q',   f.read(8))[0]
            f.read(8 * track_len)                               # skip track
            points[pid] = (xyz, rgb)
    xyz = np.array([v[0] for v in points.values()], dtype=np.float32)
    rgb = np.array([v[1] for v in points.values()], dtype=np.uint8)
    return xyz, rgb


def _read_points3d_txt(path):
    xyzs, rgbs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            xyzs.append([float(p[1]), float(p[2]), float(p[3])])
            rgbs.append([int(p[4]),   int(p[5]),   int(p[6])])
    return np.array(xyzs, dtype=np.float32), np.array(rgbs, dtype=np.uint8)


def read_colmap_points3d(sparse_dir):
    """
    Read points3D from a COLMAP sparse directory.
    Handles both binary and text formats, and the optional /0 subdirectory.
    Returns xyz [N, 3] float32 and rgb [N, 3] uint8.
    """
    for subdir in [sparse_dir, os.path.join(sparse_dir, '0')]:
        for name, reader in [('points3D.bin', _read_points3d_bin),
                              ('points3D.txt', _read_points3d_txt)]:
            path = os.path.join(subdir, name)
            if os.path.exists(path):
                xyz, rgb = reader(path)
                print(f"Loaded {len(xyz):,} COLMAP points from {path}")
                return xyz, rgb
    raise FileNotFoundError(f"No points3D.bin/txt found under {sparse_dir}")
