# SPDX-FileCopyrightText: 2026 Simon Dorrer
# SPDX-License-Identifier: Apache-2.0
"""test_dc_point.py - the 0 Hz point of a universal fit (no Qt).

A file can carry a 0 Hz sample, which the fit drops and extrapolates to.  Passivity
enforcement tries a refit that includes it when residue perturbation cannot reach the ceiling,
as one more candidate next to the lower-order refit.  Run with:  pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
import skrf                                          # noqa: E402

from snp2le.core import io, engine                   # noqa: E402
from snp2le.core.state import ConverterState         # noqa: E402

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "snp2le", "examples")


def _example(name):
    return io.load_touchstone(os.path.join(_EXAMPLES, name))


def _with_dc_point(net):
    """`net` with a 0 Hz point taken from its lowest sample, the S-parameters made real, as
    an EM solver's extrapolated DC point would be."""
    f = np.concatenate(([0.0], net.f))
    s = np.concatenate((np.real(net.s[:1]), net.s))
    return skrf.Network(frequency=skrf.Frequency.from_f(f, unit="Hz"), s=s, z0=net.z0[0, 0])


def _convert(net, **kw):
    return engine.convert(ConverterState(mode="universal", **kw), net)


def test_a_refit_with_the_dc_point_rescues_a_runaway_constant_term():
    """The 500 pH coil at order 6 stays near-passive: its constant term D sits at 2.18,
    residue perturbation cannot move it, and the lower-order refit misses too.  Given a
    0 Hz point, enforcement also tries a refit that includes it, which lands on another
    model, and that one comes out strictly passive."""
    net = _example("ind_500pH_ihp-sg13cmos5l.s2p")
    plain = _convert(net, max_order=6)
    assert plain.passivity_path == "near-passive" and not plain.passive
    res = _convert(_with_dc_point(net), max_order=6)
    assert res.ok and res.passive and res.passivity_path == "dc refit"
    assert res.sigma_max <= 1.0
    assert any("refit with the 0 Hz point" in m for m in res.messages)
    assert not any("not needed" in m for m in res.messages)


def test_the_refit_with_the_dc_point_only_runs_where_it_can_help():
    """A file whose fit perturbs to passivity never sees it, so its result is the one it
    always had, and a fit range that starts above 0 Hz leaves no DC point to refit with."""
    net = _with_dc_point(_example("wpd_ihp-sg13g2.s3p"))
    res = _convert(net, max_order=6)
    assert res.passivity_path == "perturbed"
    assert res.ngspice == _convert(io.without_dc(net), max_order=6).ngspice
    coil = _with_dc_point(_example("ind_500pH_ihp-sg13cmos5l.s2p"))
    above = _convert(coil, max_order=6, f_min=1e9)
    assert above.passivity_path == "near-passive"


def test_a_dc_point_above_passivity_is_made_passive():
    """An extrapolated DC point can come out a hair above passivity, and a fit that includes
    it would inherit a negative DC conductance.  Singular values above 1 are clipped, the
    ones below stay, so a coil's DC resistance keeps its value."""
    net = _example("ind_d20_w7_sp3_nw1_r10_ihp-sg13g2.s2p")
    u, sv, vh = np.linalg.svd(net.s[0])
    hot = net.copy()
    hot.s[0] = u @ np.diag(sv * [1.0 + 1e-5, 1.0]) @ vh
    fixed, before = io.passive_dc(hot)
    after = np.linalg.svd(fixed.s[0], compute_uv=False)
    assert before > 1.0 and after[0] <= 1.0 + 1e-12
    assert abs(after[1] - sv[1]) < 1e-12
    assert hot.s[0] is not fixed.s[0]                  # the caller's network is untouched
    plain, nan = io.passive_dc(_example("bpf_ihp-sg13g2.s2p"))
    assert nan != nan and plain.f[0] > 0


def test_a_dc_point_that_is_not_finite_is_ignored():
    """A 0 Hz row of NaN is dropped like any 0 Hz row and never refitted, so the result is
    the one the file gives without it."""
    net = _example("ind_500pH_ihp-sg13cmos5l.s2p")
    bad = _with_dc_point(net)
    bad.s[0] = np.nan
    same, nan = io.passive_dc(bad)
    assert same is bad and nan != nan
    res = _convert(bad, max_order=6)
    assert res.ok and res.passivity_path == "near-passive"
    assert res.ngspice == _convert(net, max_order=6).ngspice
