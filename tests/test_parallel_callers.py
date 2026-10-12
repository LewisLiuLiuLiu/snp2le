# SPDX-FileCopyrightText: 2026 Simon Dorrer
# SPDX-License-Identifier: Apache-2.0
"""test_parallel_callers.py - fits that overlap in threads give the process its streams back.

engine.convert is also a library call, and a caller may run several at once in threads.  A fit
silences sys.stdout and sys.stderr, which belong to the whole process.  The overlap below is
forced rather than left to timing: both fits are inside the silenced section, then the first
one leaves before the second.  With a save-and-restore per call the second one would put back
the first one's buffer, and the process would stay silent for good.
Run with:  pytest -q
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snp2le.core import io, universal       # noqa: E402

_BPF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "snp2le", "examples", "bpf_ihp-sg13g2.s2p")


def test_overlapping_fits_give_the_streams_back(monkeypatch):
    out, err = sys.stdout, sys.stderr
    both_inside = threading.Barrier(2, timeout=120)
    first_left = threading.Event()
    silenced = []
    real = universal._auto_fit

    def held(*args, **kwargs):
        vf = real(*args, **kwargs)
        both_inside.wait()
        silenced.append(sys.stdout is not out)
        if threading.current_thread().name == "second":
            assert first_left.wait(timeout=120)
            silenced.append(sys.stdout is not out)    # the first fit has left, still silent
        return vf

    monkeypatch.setattr(universal, "_auto_fit", held)
    net = io.without_dc(io.load_touchstone(_BPF))
    errors = []

    def fit(done=None):
        try:
            universal.fit_universal(net, max_order=6)
        except Exception as exc:                   # noqa: BLE001
            errors.append(exc)
        if done is not None:
            done.set()

    first = threading.Thread(target=fit, args=(first_left,), name="first")
    second = threading.Thread(target=fit, name="second")
    first.start()
    second.start()
    first.join()
    second.join()
    assert not errors, errors
    assert silenced == [True, True, True]
    assert sys.stdout is out and sys.stderr is err
