# SPDX-FileCopyrightText: 2026 Simon Dorrer
# SPDX-License-Identifier: Apache-2.0
"""test_dc_point.py - the 0 Hz point of a universal fit (no Qt).

A file can carry a 0 Hz sample, which the fit drops and extrapolates to.  Passivity
enforcement tries a refit that includes it when residue perturbation cannot reach the ceiling,
as one more candidate next to the lower-order refit.  The DC pin (dcpin.py) then makes the
model meet the sample exactly.  The pin checks solve the emitted circuit itself, at 0 Hz and
in the band, so a sign or a node gone wrong in the netlist fails here.  Run with:  pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
import skrf                                          # noqa: E402

from snp2le.core import io, engine, dcpin            # noqa: E402
from snp2le.core.state import ConverterState         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_noise import _port_noise                   # noqa: E402

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


def _with_series_dc(net, r):
    """`net` with the exact 0 Hz point of a series resistor `r` between its two 50 ohm
    ports, whose common mode is open, a singular value of exactly 1."""
    a = r / (r + 100.0)
    s0 = np.array([[[a, 1.0 - a], [1.0 - a, a]]])
    f = np.concatenate(([0.0], net.f))
    return skrf.Network(frequency=skrf.Frequency.from_f(f, unit="Hz"),
                        s=np.concatenate((s0, net.s)), z0=50.0)


def _convert(net, **kw):
    return engine.convert(ConverterState(mode="universal", **kw), net)


def _r_dc(s, z0=50.0):
    """Resistance between the two ports of a 2-port at DC, from its S-matrix."""
    y = (np.eye(2) - s) @ np.linalg.inv(np.eye(2) + s) / z0
    return float(-1.0 / np.real(y[0, 1]))


_COIL = "ind_d20_w7_sp3_nw1_r10_ihp-sg13g2.s2p"             # starts at 0 Hz


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


def test_the_refit_stays_strictly_passive_so_it_can_carry_noise():
    """A coil's exact 0 Hz point has an open common mode, a singular value of exactly 1.  A
    refit that met it would be passive but not strictly, with no thermal noise, so the refit
    gets the point 1e-6 below it."""
    coil = _with_series_dc(_example("ind_500pH_ihp-sg13cmos5l.s2p"), 0.5)
    res = _convert(coil, max_order=6, thermal_noise=True)
    assert res.passivity_path == "dc refit" and res.passive
    assert res.noise.ok, res.noise.message


def test_the_pin_gives_the_coil_its_dc_resistance():
    """The bundled coil's file reads 1.0964 ohm at 0 Hz, its fit at order 13 1.29 ohm.  The
    pinned netlist, solved as a circuit, reads the file's value."""
    net = _example(_COIL)
    want = _r_dc(np.real(net.s[0]))
    res = _convert(net, max_order=13)
    assert res.dc_pin.ok and res.dc_pin.n_sources == 6, res.dc_pin.message
    assert abs(_r_dc(np.real(_port_noise(res.ir, 0.0)[1])) / want - 1.0) < 1e-9
    plain = _convert(net, max_order=13, dc_pin=False)
    assert plain.dc_pin is None
    assert _r_dc(np.real(_port_noise(plain.ir, 0.0)[1])) > 1.15 * want
    assert any(m.startswith("DC pinned to the file's 0 Hz point") for m in res.messages)


def test_the_pin_leaves_the_band_and_the_noise_alone():
    """In the band the pinned netlist is the fit to 1e-7, the overlay shows the netlist,
    the noise generator and the passivity result are those of the fit, and the netlist is
    the unpinned one plus the pin's lines."""
    net = _example(_COIL)
    on = _convert(net, max_order=13, thermal_noise=True)
    off = _convert(net, max_order=13, thermal_noise=True, dc_pin=False)
    assert on.dc_pin.ok and on.noise.ok and off.noise.ok and on.passive == off.passive
    assert np.max(np.abs(on.model_s - off.model_s)) == on.dc_pin.rf_change < 1e-7
    for k in (0, len(on.freq) // 2, len(on.freq) - 1):
        circuit = _port_noise(on.ir, on.freq[k])[1]
        assert np.max(np.abs(circuit - on.model_s[k])) < 1e-9
    for a, b in ((on.ngspice, off.ngspice), (on.vacask, off.vacask)):
        added = [ln for ln in a.splitlines() if ln not in set(b.splitlines())]
        assert [ln for ln in a.splitlines() if ln not in added] == b.splitlines()
        assert added and all("dcp" in ln for ln in added)
    assert "Rdcp_1 dcp_d1 0 1.0 noisy=0" in on.ngspice
    assert "Gdcp_v1 (GND dcp_d1 p1 GND) vccs gain=1.0" in on.vacask


def test_the_pin_only_acts_on_a_universal_fit_of_a_file_with_a_0_hz_point():
    bpf = _example("bpf_ihp-sg13g2.s2p")
    res = _convert(bpf, max_order=6)
    assert res.dc_pin is None and res.ngspice == _convert(bpf, max_order=6, dc_pin=False).ngspice
    st = ConverterState(mode="structure", structure_key="inductor-pi")
    assert engine.convert(st, _example(_COIL)).dc_pin is None
    bad = _with_dc_point(_example("ind_500pH_ihp-sg13cmos5l.s2p"))
    bad.s[0] = np.nan
    assert _convert(bad, max_order=13).dc_pin is None


def test_the_pin_uses_the_0_hz_point_below_a_fit_range():
    """A fit range above the file's first sample still has the file's 0 Hz point to pin to."""
    net = _example(_COIL)
    res = _convert(net, max_order=13, f_min=float(net.f[5]))
    assert res.band_limited and res.dc_pin.ok
    assert abs(_r_dc(np.real(_port_noise(res.ir, 0.0)[1])) / _r_dc(np.real(net.s[0])) - 1) < 1e-9


def test_a_pin_that_would_make_the_dc_point_singular_comes_back_out():
    """The WPD fitted at order 6 with a 0 Hz point prepended nearly shorts a mode at DC, 566 S,
    which the pin would have to cancel.  That leaves the DC matrix singular, so the netlist
    stays the fit's."""
    net = _with_dc_point(_example("wpd_ihp-sg13g2.s3p"))
    res = _convert(net, max_order=6)
    assert not res.dc_pin.ok and "singular" in res.dc_pin.message
    assert res.dc.ok and res.ngspice == _convert(net, max_order=6, dc_pin=False).ngspice


def test_the_island_leak_reaches_only_the_floating_mode():
    """The coil's common mode has no DC path.  Without a leak it keeps the file's 1e-10 S,
    and with one it gets that conductance on top.  The DC resistance stays."""
    net = _example(_COIL)
    want = _r_dc(np.real(net.s[0]))
    g_file = np.linalg.eigvalsh(dcpin.dc_target(net.s[0], np.full(2, 50.0))[0])[0]
    assert 0 < g_file < 1e-9
    for leak in (0.0, 1e-6):
        g_cm = g_file + leak
        res = _convert(net, max_order=13, dc_island_leak=leak)
        assert res.dc_pin.ok and res.dc_pin.n_islands == 1
        s = np.real(_port_noise(res.ir, 0.0)[1])
        y = (np.eye(2) - s) @ np.linalg.inv(np.eye(2) + s) / 50.0
        assert abs(np.linalg.eigvalsh(0.5 * (y + y.T))[0] / g_cm - 1.0) < 1e-3
        assert abs(_r_dc(s) / want - 1.0) < 1e-5


def test_a_bad_setting_in_a_loaded_design_skips_the_pin():
    net = _example(_COIL)
    for kw in ({"dc_pin_fc": 0.0}, {"dc_pin_fc": "fast"}, {"dc_island_leak": -1.0}):
        res = _convert(net, max_order=13, **kw)
        assert res.ok and not res.dc_pin.ok and "DC pin skipped" in res.dc_pin.message
        assert "dcp" not in res.ngspice


def test_the_pin_matches_its_closed_form():
    """pinned_sparams is Y_fit + dY / (1 + jf/fc) written without (I + S)^-1."""
    rng = np.random.default_rng(1)
    s = 0.3 * (rng.standard_normal((4, 3, 3)) + 1j * rng.standard_normal((4, 3, 3)))
    dy = rng.standard_normal((3, 3)) * 0.01
    f = np.array([0.0, 50.0, 1e3, 1e9])
    got = dcpin.pinned_sparams(s, f, dy, np.full(3, 50.0), 160.0)
    for k in range(len(f)):
        y = (np.eye(3) - s[k]) @ np.linalg.inv(np.eye(3) + s[k]) / 50.0
        y = y + dy / (1.0 + 1j * f[k] / 160.0)
        want = np.linalg.solve(np.eye(3) + 50.0 * y, np.eye(3) - 50.0 * y)
        assert np.max(np.abs(got[k] - want)) < 1e-12


def test_the_pin_options_survive_a_design_round_trip():
    st = ConverterState(dc_pin=False, dc_pin_fc=1e3, dc_island_leak=1e-6)
    back = ConverterState.from_json(st.to_json())
    assert (back.dc_pin, back.dc_pin_fc, back.dc_island_leak) == (False, 1e3, 1e-6)
    # a design saved before the pin existed opens with it on, at its defaults
    old = ConverterState.from_json('{"mode": "universal"}')
    assert (old.dc_pin, old.dc_pin_fc, old.dc_island_leak) == (True, 160.0, 0.0)


def test_cli_dc_pin_flags(capsys):
    import pytest
    from snp2le import cli
    p = cli.build_parser()
    a = p.parse_args(["convert", "x.s2p"])
    assert (a.dc_pin, a.dc_pin_fc, a.dc_island_leak) == (None, None, None)
    assert p.parse_args(["convert", "x.s2p", "--no-dc-pin"]).dc_pin is False
    assert p.parse_args(["convert", "x.s2p", "--dc-pin-fc", "1kHz"]).dc_pin_fc == 1e3
    assert p.parse_args(["convert", "x.s2p", "--dc-island-leak", "1e-6"]).dc_island_leak == 1e-6
    for bad in (["--dc-pin-fc", "0"], ["--dc-island-leak", "-1"], ["--dc-island-leak", "inf"]):
        with pytest.raises(SystemExit):
            p.parse_args(["convert", "x.s2p", *bad])
    capsys.readouterr()


def test_cli_writes_the_pinned_netlist(tmp_path, capsys):
    from snp2le import cli
    src = os.path.join(_EXAMPLES, _COIL)
    out = tmp_path / "coil.spice"
    assert cli.main(["convert", src, "--order", "13", "-o", str(out), "--no-progress"]) == 0
    assert "Gdcp_y1_2" in out.read_text() and "dc_pinned" in capsys.readouterr().out
    assert cli.main(["convert", src, "--order", "13", "--no-dc-pin", "-o", str(out),
                     "--quiet"]) == 0
    assert "dcp" not in out.read_text()
    assert cli.main(["convert", src, "--mode", "structure", "--dc-pin-fc", "1k", "-o",
                     str(tmp_path / "s.spice"), "--quiet"]) == 0
    assert "--dc-pin-fc ignored in structure mode" in capsys.readouterr().err
