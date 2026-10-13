# SPDX-FileCopyrightText: 2026 Simon Dorrer
# SPDX-License-Identifier: Apache-2.0
"""dcpin.py - the DC pin: a universal model that meets the file's 0 Hz point (no Qt).

A universal fit drops the file's 0 Hz sample and extrapolates to DC from the samples above it.
A coil's DC resistance sits in a small departure of S21 from 1 (0.3 % for 0.3 ohm between two
50 ohm ports), so a fit error of 1e-3 in S puts it tens of percent off.  The pin leaves the fit
as it is and adds the difference at DC through a first-order low-pass:

    Y(s) = Y_fit(s) + dY / (1 + s / wc),    dY = Y_dc - Y_fit(0),

so the model meets the file's DC admittance exactly and moves by about |dY| fc / f well above
the corner fc.  Per port j a VCCS copies V(p_j) into a node d_j loaded by 1 ohm and 1/wc farad,
and per nonzero dY_ij a VCCS from p_i to ground reads V(d_j).  Its resistor only sets the
low-pass and is noiseless.

The target is the file's S(0) made real and symmetric, with its eigenvalues clipped to
+-DC_SIGMA, since an EM solver's extrapolated DC point can come out a hair above passivity.
Below a few kHz Y_fit(jw) is close to Y_fit(0), so the real part of the pinned Y is close to a
convex combination of the fit's and the target's DC conductance, both passive.  That is no
proof at every frequency, so the pinned model is checked on a grid, and the pin is skipped if
it would take a frequency where the fit is passive to sigma >= 1, or move the fitted band by
more than the fit's own RMS error.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .ir import Element

# Largest singular value of the 0 Hz point the pin aims at, and that the refit in passivity
# enforcement fits (engine.py).  A coil's open common mode sits at exactly 1 there, and a
# model that met it would be passive but not strictly, so it could not carry thermal noise
# (5 of 20 ORCA coils at order 40 lost it to the refit).  A mode with no DC path keeps
# (1 - DC_SIGMA) / (2 z0), 1e-8 S at 50 ohm.
DC_SIGMA = 1.0 - 1e-6
# Gain of the copy sources, the read gains scaled up by its inverse.  VACASK, which does not
# scale its matrix, solved the pinned coil 4e-7 off at 1 THz with a gain of 1 (a 3-port ORCA
# coil 2e-7), and all seven cases tried to 1e-15 with gains from 1e-4 to 1e-9.  Ngspice is
# exact either way.
_COPY = 1e-6
# A DC mode below this many times the clip's floor, (1 - DC_SIGMA) / (2 z0), has no DC path to
# ground and gets the optional leak: 1e-7 S at 50 ohm.
_ISLAND = 10.0
# dY entries below this fraction of the largest target entry are dropped.
_ZERO = 1e-12
# Check grid below the band: log-spaced from this many decades under the corner.
_GRID_DECADES = 3.0
_GRID_POINTS = 300
# Prefix of every node and element the pin adds, distinct from scikit-rf's names and from the
# noise generator's nz_.
_TAG = "dcp"


@dataclass
class DCPinHealth:
    ok: bool                            # the correction is in the netlist
    message: str
    fc: float = float("nan")            # corner of the low-pass [Hz]
    dc_move: float = float("nan")       # largest change of the model's S at 0 Hz
    rf_change: float = float("nan")     # largest change of the model's S in the fitted band
    dy_max: float = float("nan")        # largest |dY| entry [S]
    sigma_dc: float = float("nan")      # sigma_max of the file's S(0) before clipping
    n_islands: int = 0                  # DC modes without a DC path to ground
    n_sources: int = 0                  # controlled sources in the netlist, 0 when skipped


def _y_from_s(s, rz):
    """Y of a real S-matrix with real reference resistances `rz`, or None when I + S is
    singular (a port shorted at DC)."""
    n = len(s)
    a = np.eye(n) + s
    if np.linalg.cond(a) > 1e12:
        return None
    g = np.linalg.solve(a.T, (np.eye(n) - s).T).T          # (I - S)(I + S)^-1
    r = 1.0 / np.sqrt(rz)
    return r[:, None] * g * r[None, :]


def dc_target(s0, rz, leak=0.0):
    """(Y_dc, sigma_dc, n_islands, n_shorts, made_real) of the file's 0 Hz S-matrix `s0`.

    The S-matrix is made real and symmetric (`made_real` is how far that moved it) and its
    eigenvalues are clipped to +-DC_SIGMA before the conversion.  `n_shorts` counts modes at
    -DC_SIGMA or below, ports shorted at DC, which no finite conductance meets.  With `leak`
    above 0, each DC mode without a DC path (see _ISLAND) gets that conductance to ground,
    through the projector on its eigenvector."""
    s0 = np.asarray(s0)
    sigma = float(np.linalg.svd(s0, compute_uv=False)[0])
    s = 0.5 * (np.real(s0) + np.real(s0).T)
    made_real = float(np.max(np.abs(s0 - s)))
    lam, q = np.linalg.eigh(s)
    n_shorts = int(np.count_nonzero(lam <= -DC_SIGMA))
    lam = np.clip(lam, -DC_SIGMA, DC_SIGMA)
    y = _y_from_s((q * lam) @ q.T, rz)
    y = 0.5 * (y + y.T)
    g, v = np.linalg.eigh(y)
    floor = (1.0 - DC_SIGMA) / ((1.0 + DC_SIGMA) * float(np.min(rz)))
    isl = v[:, g < _ISLAND * floor]
    if leak > 0 and isl.shape[1]:
        y = y + leak * isl @ isl.T
    return y, sigma, int(isl.shape[1]), n_shorts, made_real


def pinned_sparams(s_fit, f, dy, rz, fc):
    """S[f] of the pinned model from the fit's S[f]: Y_fit + dY h(f), h = 1 / (1 + jf/fc).

    Written without (I + S)^-1, so a frequency where the fit shorts a port stays finite:
    with M = h R^1/2 dY R^1/2, S_pin = (2I + (I + S) M)^-1 (2S - (I + S) M)."""
    s_fit = np.asarray(s_fit, dtype=complex)
    n = s_fit.shape[-1]
    r = np.sqrt(rz)
    m0 = r[:, None] * dy * r[None, :]
    h = 1.0 / (1.0 + 1j * np.asarray(f, dtype=float) / fc)
    eye = np.eye(n)
    ips = eye + s_fit
    ipm = ips @ (h[:, None, None] * m0)
    return np.linalg.solve(2.0 * eye + ipm, 2.0 * s_fit - ipm)


def pin_elements(ports, dy, fc, ground="0"):
    """The correction as IR elements: a low-pass copy of each port voltage that some dY
    entry reads, and a VCCS per nonzero dY entry."""
    els = []
    cols = [j for j in range(len(ports)) if np.any(dy[:, j] != 0.0)]
    for j in cols:
        d = f"{_TAG}_d{j + 1}"
        els.append(Element("G", f"G{_TAG}_v{j + 1}", (ground, d), _COPY,
                           ctrl=(ports[j], ground)))
        els.append(Element("R", f"R{_TAG}_{j + 1}", (d, ground), 1.0, noisy=False))
        els.append(Element("C", f"C{_TAG}_{j + 1}", (d, ground), 1.0 / (2.0 * np.pi * fc)))
    for i in range(len(ports)):
        for j in cols:
            if dy[i, j] != 0.0:
                els.append(Element("G", f"G{_TAG}_y{i + 1}_{j + 1}", (ports[i], ground),
                                   float(dy[i, j]) / _COPY, ctrl=(f"{_TAG}_d{j + 1}", ground)))
    return els


def _sigma(s):
    return np.linalg.svd(s, compute_uv=False)[..., 0]


def _as_float(value):
    """A loaded design can carry anything here. A value that is no number becomes NaN, which
    its own check then names."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def add_dc_pin(ir, vf, s0, rz, freq, model_s, rms, fc, leak=0.0):
    """Pin the realisation `ir` of the fit `vf` to the 0 Hz S-matrix `s0`, in place.

    `rz` are the real port reference resistances, `freq` and `model_s` the fitted band and
    the fit's S there, `rms` its RMS error.  Returns (DCPinHealth, the pinned model's S on
    `freq`).  When `ok` is False nothing was added, the S comes back as it was, and
    `message` says why."""
    from .units import format_eng
    from .universal import model_sparams
    skip = "DC pin skipped: "
    fc, leak = _as_float(fc), _as_float(leak)
    if not (np.isfinite(fc) and fc > 0):
        return (DCPinHealth(False, skip + "the corner frequency must be finite and above 0 Hz"),
                model_s)
    if not (np.isfinite(leak) and leak >= 0):
        return DCPinHealth(False, skip + "the leak must be 0 S or more"), model_s
    if any(n.startswith(_TAG + "_") for e in ir.elements for n in e.nodes):
        return DCPinHealth(False, skip + f"the netlist already uses '{_TAG}_' node names"), model_s
    rz = np.asarray(rz, dtype=float)
    y_dc, sigma_dc, n_isl, n_shorts, made_real = dc_target(s0, rz, leak)
    s_fit0 = np.real(model_sparams(vf, [0.0])[0])
    y_fit0 = _y_from_s(s_fit0, rz)
    if n_shorts or y_fit0 is None:
        where = "the file's 0 Hz point" if n_shorts else "the fit"
        return DCPinHealth(False, skip + f"{where} shorts a port at DC, which no finite "
                                         f"conductance meets", fc=fc, sigma_dc=sigma_dc,
                           n_islands=n_isl), model_s
    dy = y_dc - y_fit0
    dy[np.abs(dy) < _ZERO * max(1.0, float(np.max(np.abs(y_dc))))] = 0.0
    health = DCPinHealth(False, "", fc=fc, dy_max=float(np.max(np.abs(dy))),
                         sigma_dc=sigma_dc, n_islands=n_isl)
    if not np.any(dy):
        health.message = "DC pin: the model already meets the file's 0 Hz point"
        health.dc_move = health.rf_change = 0.0
        return health, model_s

    # the fitted band, and a grid from below the corner up to the band
    freq = np.asarray(freq, dtype=float)
    s_band = pinned_sparams(model_s, freq, dy, rz, fc)
    health.rf_change = float(np.max(np.abs(s_band - model_s)))
    top = float(freq[-1]) if freq.size else fc * 1e6
    grid = np.concatenate(([0.0], np.logspace(np.log10(fc) - _GRID_DECADES, np.log10(top),
                                              _GRID_POINTS)))
    s_grid = model_sparams(vf, grid)
    s_pin = pinned_sparams(s_grid, grid, dy, rz, fc)
    health.dc_move = float(np.max(np.abs(s_pin[0] - s_fit0)))

    if not health.rf_change <= rms:
        health.message = (skip + f"it would move the fitted band by {health.rf_change:.1e}, "
                                 f"more than the fit's RMS error {rms:.1e}. A lower corner "
                                 f"frequency (CLI --dc-pin-fc) moves it less")
        return health, model_s
    all_f = np.concatenate((grid, freq))
    sig_fit = np.concatenate((_sigma(s_grid), _sigma(model_s)))
    sig_pin = np.concatenate((_sigma(s_pin), _sigma(s_band)))
    worse = (sig_fit < 1.0) & (sig_pin >= 1.0)
    if np.any(worse):
        k = int(np.argmax(np.where(worse, sig_pin, -np.inf)))
        health.message = (skip + f"it would take sigma_max to {sig_pin[k]:.7f} at "
                                 f"{format_eng(all_f[k], 'Hz')}, where the fit is passive")
        return health, model_s

    els = pin_elements(list(ir.ports), dy, fc)
    for el in els:
        ir.add(el)
    health.ok = True
    health.n_sources = sum(1 for e in els if e.kind == "G")
    ir.notes += [
        f"DC pin: Y = Y_fit + dY / (1 + s/wc), fc = {fc:g} Hz, so the model meets the file's "
        f"0 Hz point. Nodes {_TAG}_d*, controlled sources G{_TAG}_*, noiseless"]
    extra = []
    if made_real > 1e-6:
        extra.append(f"the file's 0 Hz point was made real and symmetric, a change of "
                     f"{made_real:.1e}")
    if sigma_dc > 1.0:
        extra.append(f"the file's 0 Hz point had sigma_max {sigma_dc:.7f} and was clipped "
                     f"below 1")
    if leak > 0 and n_isl:
        extra.append(f"{n_isl} DC mode{'' if n_isl == 1 else 's'} without a DC path got "
                     f"{leak:.1e} S to ground")
    health.message = (f"DC pinned to the file's 0 Hz point: S at 0 Hz moved by "
                      f"{health.dc_move:.1e}, the fitted band by at most "
                      f"{health.rf_change:.1e}" + "".join(f", {x}" for x in extra))
    return health, s_band
