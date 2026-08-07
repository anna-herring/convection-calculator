"""X-ray mass attenuation coefficients, log-log interpolated in energy.

Values are the NIST tabulations (Hubbell & Seltzer, NIST SRD 126):
  Xe    physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z54.html
  water physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
Only tabulated grid points are stored -- no hand-entered intermediate
values -- so everything between them is honest log-log interpolation.

Xenon has a K-edge at 34.5614 keV. The table is split into below-edge and
above-edge branches so interpolation never crosses the discontinuity; that
jump (a factor of 5.4) is the main lever on Xe contrast at a beamline.
"""
import numpy as np

XE_K_EDGE_KEV = 34.5614

# (energy keV, mu/rho cm2/g) -- xenon, BELOW the K edge
_XE_BELOW = [
    (10.0, 169.0), (15.0, 57.43), (20.0, 26.52), (30.0, 8.930),
    (34.5614, 6.129),
]
# xenon, ABOVE the K edge
_XE_ABOVE = [
    (34.5614, 33.16), (40.0, 22.70), (50.0, 12.72), (60.0, 7.825),
    (80.0, 3.633), (100.0, 2.011), (150.0, 0.7202), (200.0, 0.3760),
]
# liquid water
_WATER = [
    (10.0, 5.329), (15.0, 1.673), (20.0, 0.8096), (30.0, 0.3756),
    (40.0, 0.2683), (50.0, 0.2269), (60.0, 0.2059), (80.0, 0.1837),
    (100.0, 0.1707), (150.0, 0.1505), (200.0, 0.1370),
]


def _loglog(table, E_keV):
    e = np.array([t[0] for t in table])
    m = np.array([t[1] for t in table])
    E = float(np.clip(E_keV, e[0], e[-1]))
    return float(np.exp(np.interp(np.log(E), np.log(e), np.log(m))))


def mu_rho_xenon(E_keV, above_edge=None):
    """Xe mass attenuation coefficient [cm2/g]. At the edge, above_edge picks branch."""
    if above_edge is None:
        above_edge = E_keV >= XE_K_EDGE_KEV
    return _loglog(_XE_ABOVE if above_edge else _XE_BELOW, E_keV)


def mu_rho_water(E_keV):
    """Water mass attenuation coefficient [cm2/g]."""
    return _loglog(_WATER, E_keV)


def edge_jump():
    """Ratio of Xe mu/rho just above vs just below the K edge."""
    return mu_rho_xenon(XE_K_EDGE_KEV, True) / mu_rho_xenon(XE_K_EDGE_KEV, False)
