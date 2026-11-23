"""Check the CUDA retrieval kernels against an independent exhaustive bucket implementation."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    device_count = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    device_count = 0
pytestmark = pytest.mark.skipif(device_count == 0, reason="CUDA device required")

from afp.gpu_retrieval import GPUAngularLSH, gpu_exact


def test_bucket_union_matches_across_devices_and_query_batch_sizes():
    """Verify all four probe settings, visits with multiplicity and consistent window tie breaking."""
    rng = np.random.default_rng(606)
    refs = rng.standard_normal((100, 256), dtype=np.float32)
    refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    refs[1] = refs[0]
    tracks = np.arange(500, 600, dtype=np.int32)
    queries = np.stack([refs[0], refs[14], -refs[61]])
    index = GPUAngularLSH(cp.asarray(refs), cp.asarray(tracks), bits=5, tables=8)
    planes = index.planes.get()
    weights = 1 << np.arange(index.bits)
    signatures = ((refs @ planes).reshape(-1, index.tables, index.bits) >= 0) @ weights
    for probes in range(4):
        expected = []
        for query in queries:
            projections = (query @ planes).reshape(index.tables, index.bits)
            keys = (projections >= 0) @ weights
            nearby = np.argsort(np.abs(projections), axis=1)
            windows = []
            for table, key in enumerate(keys):
                first, second = weights[nearby[table, :2]]
                for mask in [0, first, second, first | second][: probes + 1]:
                    windows.extend(np.flatnonzero(signatures[:, table] == (key ^ mask)).tolist())
            rows = np.unique(windows)
            if len(rows):
                scores = refs[rows] @ query
                expected.append((tracks[rows[scores.argmax()]], scores.max(), len(windows)))
            else:
                expected.append((-1, -np.inf, 0))
        for start, stop in [(0, 3), (0, 1), (1, 2), (2, 3)]:
            got = index.search(cp.asarray(queries[start:stop]), probes=probes)
            for column, array in enumerate(got):
                np.testing.assert_allclose(array.get(), np.asarray(expected)[start:stop, column], atol=2e-6)
    exact_track, exact_score = gpu_exact(cp.asarray(refs), cp.asarray(tracks), cp.asarray(queries))
    dense = queries @ refs.T
    np.testing.assert_array_equal(exact_track.get(), tracks[np.argmax(dense, axis=1)])
    np.testing.assert_allclose(exact_score.get(), dense.max(axis=1), atol=2e-6)


def test_empty_cuda_buckets_return_missing_track():
    """An antipodal query has no matching signatures at zero extra probes."""
    references = cp.zeros((1, 256), dtype=cp.float32)
    references[0, 0] = 1
    index = GPUAngularLSH(references, cp.array([7]), bits=8, tables=4)
    track, score, visits = index.search(-references)
    assert track.item() == -1 and score.item() == -np.inf and visits.item() == 0
