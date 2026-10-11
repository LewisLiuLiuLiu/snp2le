# SPDX-FileCopyrightText: 2026 Simon Dorrer
# SPDX-License-Identifier: Apache-2.0
"""test_blas_threads.py - a conversion runs its linear algebra on one BLAS thread.

A fit's matrices are small, so the BLAS thread pool costs more than it saves.  engine.convert
limits the loaded BLAS libraries to one thread with threadpoolctl, gives them their count back
afterwards, and leaves them alone when the user set OPENBLAS_NUM_THREADS or MKL_NUM_THREADS.
The tests start from 2 threads, so the limit shows even when pytest itself runs on one.
Run with:  pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest                                                   # noqa: E402

pytest.importorskip("threadpoolctl")
from threadpoolctl import threadpool_info, threadpool_limits    # noqa: E402

from snp2le.core import engine, io                              # noqa: E402
from snp2le.core.state import ConverterState                    # noqa: E402

_BPF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "snp2le", "examples", "bpf_ihp-sg13g2.s2p")


def _blas_threads():
    return sorted({i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"})


def _threads_seen_while_converting():
    """The BLAS thread counts reported from inside a conversion, at every progress step."""
    seen = set()
    res = engine.convert(ConverterState(mode="universal", max_order=6, thermal_noise=True),
                         io.load_touchstone(_BPF),
                         progress=lambda frac, msg: seen.add(tuple(_blas_threads())))
    assert res.ok and res.noise.ok, res.error or res.noise.message
    return seen


@pytest.fixture()
def two_threads():
    import scipy.linalg                     # noqa: F401  load scipy's own BLAS copy too
    with threadpool_limits(limits=2, user_api="blas"):
        if _blas_threads() != [2]:
            pytest.skip(f"cannot set 2 BLAS threads here (found {_blas_threads()})")
        yield


def test_a_conversion_runs_on_one_blas_thread_and_gives_the_count_back(two_threads,
                                                                         monkeypatch):
    for name in engine._BLAS_THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    assert _threads_seen_while_converting() == {(1,)}
    assert _blas_threads() == [2]


def test_a_thread_count_the_user_set_is_left_alone(two_threads, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    assert _threads_seen_while_converting() == {(2,)}
    assert _blas_threads() == [2]
