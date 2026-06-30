"""
Radionuclide transport regime calculator.

Determines dominant transport regime (diffusion / mixed / advection) for
radionuclides in porous media as a function of T, P, and interstitial velocity.

Modes
-----
Single-phase  : aqueous transport with speciation, Kd, retardation, Péclet number.
Multiphase    : as above, plus gas-phase fluid properties (CO2, CH4, hydrocarbons,
                H2S, N2) and saturation effects on effective transport parameters.

Speciation is rule-based on Eh–pH stability fields (Langmuir 1997; Bruno et al.;
SKB SR-Site 2011).  Kd values are geometric-mean literature values from:
  SKB SR-Site (2011) TR-10-40  — granite/fracture fill
  Bradbury & Baeyens (2003) J Contam Hydrol — bentonite/shale
  EPA NCRP Report 152 (2007)  — sandstone, carbonate
"""

import numpy as np
from iapws import IAPWS97 as _IAPWS97

# ── Physical constants ────────────────────────────────────────────────────────
R    = 8.314          # J/(mol·K)
kB   = 1.380649e-23   # J/K
F    = 96485.0        # C/mol
Mw_H2O  = 0.018015   # kg/mol
Mw_NaCl = 0.05844    # kg/mol

# ── Gas-phase fluids (NW phase for multiphase mode) ───────────────────────────
# Extends existing NW_FLUIDS with hydrocarbons and H2S
GAS_FLUIDS = {
    "CO2": {
        "coolprop": "CO2",      "Mw": 0.04401, "Vmp": 35.1e-6,
        "D25": 2.2e-9,  "NIST_Hcp": 0.035,    "H_dT": 2400,
        "label": "CO₂",  "color": "#e74c3c",  "group": "Common gases",
        "Tc": 304.13,   "Pc": 7.377e6, "omega":  0.2239, "mu25_uPas": 14.9,
    },
    "CH4": {
        "coolprop": "Methane",  "Mw": 0.01604, "Vmp": 37.7e-6,
        "D25": 1.88e-9, "NIST_Hcp": 0.000014, "H_dT": 1600,
        "label": "CH₄",  "color": "#f39c12",  "group": "Hydrocarbons",
        "Tc": 190.56,   "Pc": 4.599e6, "omega":  0.0115, "mu25_uPas": 11.1,
    },
    "C2H6": {
        "coolprop": "Ethane",   "Mw": 0.03007, "Vmp": 55.0e-6,
        "D25": 1.52e-9, "NIST_Hcp": 0.0000185, "H_dT": 2100,
        "label": "C₂H₆", "color": "#e67e22",  "group": "Hydrocarbons",
        "Tc": 305.32,   "Pc": 4.872e6, "omega":  0.0995, "mu25_uPas":  9.4,
    },
    "C3H8": {
        "coolprop": "Propane",  "Mw": 0.04410, "Vmp": 74.0e-6,
        "D25": 1.21e-9, "NIST_Hcp": 0.0000153, "H_dT": 2700,
        "label": "C₃H₈", "color": "#d35400",  "group": "Hydrocarbons",
        "Tc": 369.89,   "Pc": 4.252e6, "omega":  0.1521, "mu25_uPas":  8.2,
    },
    "nC4H10": {
        "coolprop": "n-Butane", "Mw": 0.05812, "Vmp": 96.5e-6,
        "D25": 1.02e-9, "NIST_Hcp": 0.0000124, "H_dT": 3200,
        "label": "n-C₄H₁₀", "color": "#c0392b", "group": "Hydrocarbons",
        "Tc": 425.13,   "Pc": 3.796e6, "omega":  0.2002, "mu25_uPas":  7.4,
    },
    "H2S": {
        "coolprop": "H2S",      "Mw": 0.03408, "Vmp": 37.0e-6,
        "D25": 1.73e-9, "NIST_Hcp": 0.001,    "H_dT": 2100,
        "label": "H₂S",  "color": "#95a5a6",  "group": "Common gases",
        "Tc": 373.10,   "Pc": 9.000e6, "omega":  0.0943, "mu25_uPas": 12.6,
    },
    "H2": {
        "coolprop": "Hydrogen", "Mw": 0.00202, "Vmp": 26.7e-6,
        "D25": 5.11e-9, "NIST_Hcp": 0.000078, "H_dT": 495,
        "label": "H₂",   "color": "#2980b9",  "group": "Common gases",
        "Tc":  33.15,   "Pc": 1.296e6, "omega": -0.2160, "mu25_uPas":  8.9,
    },
    "N2": {
        "coolprop": "Nitrogen", "Mw": 0.02802, "Vmp": 34.05e-6,
        "D25": 2.61e-9, "NIST_Hcp": 0.000600, "H_dT": 1300,
        "label": "N₂",   "color": "#27ae60",  "group": "Common gases",
        "Tc": 126.19,   "Pc": 3.396e6, "omega":  0.0372, "mu25_uPas": 17.8,
    },
}

# ── Radionuclide speciation database ─────────────────────────────────────────
# For each radionuclide, speciation rules return the dominant aqueous species
# based on Eh (mV) and pH.  Rules are evaluated in order; first match wins.
# References: Langmuir (1997); SKB TR-10-40; OECD-NEA TDB series.
#
# Each rule: (Eh_min, Eh_max, pH_min, pH_max, species_key, description)
# Eh in mV vs SHE; None = no bound.

SPECIATION_RULES = {
    "U": {
        "label": "Uranium", "Z": 92, "halflife": "4.47e9 y",
        "rules": [
            (-500, -100, None, None,  "U4",          "U⁴⁺ / UO₂(s)"),
            (-100,  100, None,  7.5,  "UO2_2p",      "UO₂²⁺ (uranyl)"),
            (-100,  100,  7.5, None,  "UO2CO3_4m",   "UO₂(CO₃)₃⁴⁻"),
            ( 100, None, None,  6.0,  "UO2_2p",      "UO₂²⁺ (uranyl)"),
            ( 100, None,  6.0, None,  "UO2CO3_4m",   "UO₂(CO₃)₃⁴⁻"),
        ],
    },
    "Tc": {
        "label": "Technetium-99", "Z": 43, "halflife": "2.13e5 y",
        "rules": [
            (None, -50, None, None, "Tc4_ox",  "TcO₂(s) / Tc(OH)₄"),
            ( -50, None, None, None, "TcO4m",  "TcO₄⁻ (pertechnetate)"),
        ],
    },
    "I": {
        "label": "Iodine-129", "Z": 53, "halflife": "1.57e7 y",
        "rules": [
            (None, 200, None, None, "Im",   "I⁻ (iodide)"),
            ( 200, None, None, None, "IO3m", "IO₃⁻ (iodate)"),
        ],
    },
    "Np": {
        "label": "Neptunium-237", "Z": 93, "halflife": "2.14e6 y",
        "rules": [
            (None, -100, None, None,  "Np4",    "Np⁴⁺"),
            (-100,  400, None, None,  "NpO2p",  "NpO₂⁺ (neptunyl)"),
            ( 400, None, None, None,  "NpO2_2p","NpO₂²⁺"),
        ],
    },
    "Pu": {
        "label": "Plutonium-239", "Z": 94, "halflife": "2.41e4 y",
        "rules": [
            (None, -200, None, None,  "Pu3",     "Pu³⁺"),
            (-200,  100, None, None,  "Pu4_ox",  "Pu(OH)₄(s) / PuO₂"),
            ( 100,  600, None, None,  "PuO2p",   "PuO₂⁺"),
            ( 600, None, None, None,  "PuO2_2p", "PuO₂²⁺"),
        ],
    },
    "Cs": {
        "label": "Cesium-137", "Z": 55, "halflife": "30.17 y",
        "rules": [
            (None, None, None, None, "Csp", "Cs⁺"),
        ],
    },
    "Sr": {
        "label": "Strontium-90", "Z": 38, "halflife": "28.8 y",
        "rules": [
            (None, None, None, None, "Sr2p", "Sr²⁺"),
        ],
    },
    "Ra": {
        "label": "Radium-226", "Z": 88, "halflife": "1.6e3 y",
        "rules": [
            (None, None, None, None, "Ra2p", "Ra²⁺"),
        ],
    },
    "Th": {
        "label": "Thorium-232", "Z": 90, "halflife": "1.41e10 y",
        "rules": [
            (None, None, None, None, "Th4", "Th(OH)₄(aq) / ThO₂(s)"),
        ],
    },
    "Se": {
        "label": "Selenium-79", "Z": 34, "halflife": "3.27e5 y",
        "rules": [
            (None, -200, None, None,  "HSem",   "HSe⁻ / Se²⁻"),
            (-200,  200, None, None,  "SeO3_2m","SeO₃²⁻ (selenite)"),
            ( 200, None, None, None,  "SeO4_2m","SeO₄²⁻ (selenate)"),
        ],
    },
    "Rn": {
        "label": "Radon-222", "Z": 86, "halflife": "3.82 d",
        "rules": [
            (None, None, None, None, "Rn_aq", "Rn(aq) / Rn(g)"),
        ],
    },
}

# ── Aqueous diffusivity at 25 °C [m²/s] and charge for each species ──────────
# Sources: CRC Handbook; Rard & Miller (1988); estimated from Stokes-Einstein
# for large complexes (ionic radius from Shannon 1976 or estimated).
SPECIES_PROPS = {
    "Csp":       {"D25": 2.06e-9, "charge": +1, "label": "Cs⁺"},
    "Sr2p":      {"D25": 0.79e-9, "charge": +2, "label": "Sr²⁺"},
    "Ra2p":      {"D25": 0.89e-9, "charge": +2, "label": "Ra²⁺"},
    "Im":        {"D25": 2.05e-9, "charge": -1, "label": "I⁻"},
    "IO3m":      {"D25": 1.08e-9, "charge": -1, "label": "IO₃⁻"},
    "TcO4m":     {"D25": 1.47e-9, "charge": -1, "label": "TcO₄⁻"},
    "Tc4_ox":    {"D25": 0.30e-9, "charge":  0, "label": "TcO₂·nH₂O"},
    "UO2_2p":    {"D25": 0.42e-9, "charge": +2, "label": "UO₂²⁺"},
    "UO2CO3_4m": {"D25": 0.35e-9, "charge": -4, "label": "UO₂(CO₃)₃⁴⁻"},
    "U4":        {"D25": 0.40e-9, "charge": +4, "label": "U⁴⁺"},
    "NpO2p":     {"D25": 0.60e-9, "charge": +1, "label": "NpO₂⁺"},
    "NpO2_2p":   {"D25": 0.50e-9, "charge": +2, "label": "NpO₂²⁺"},
    "Np4":       {"D25": 0.40e-9, "charge": +4, "label": "Np⁴⁺"},
    "PuO2p":     {"D25": 0.50e-9, "charge": +1, "label": "PuO₂⁺"},
    "PuO2_2p":   {"D25": 0.40e-9, "charge": +2, "label": "PuO₂²⁺"},
    "Pu4_ox":    {"D25": 0.30e-9, "charge":  0, "label": "Pu(OH)₄"},
    "Pu3":       {"D25": 0.45e-9, "charge": +3, "label": "Pu³⁺"},
    "Th4":       {"D25": 0.35e-9, "charge": +4, "label": "Th(OH)₄"},
    "SeO4_2m":   {"D25": 1.08e-9, "charge": -2, "label": "SeO₄²⁻"},
    "SeO3_2m":   {"D25": 0.96e-9, "charge": -2, "label": "SeO₃²⁻"},
    "HSem":      {"D25": 1.65e-9, "charge": -1, "label": "HSe⁻"},
    "Rn_aq":     {"D25": 1.1e-9,  "charge":  0, "label": "Rn(aq)"},
}

# ── Kd database (L/kg) ────────────────────────────────────────────────────────
# Three scenarios per (species, rock type):
#   "mean"         — geometric mean of published literature ranges
#   "conservative" — ~10th-percentile low value (fast transport; worst case for
#                    migration / safety assessment screening)
#   "generous"     — ~90th-percentile high value (slow transport; best case for
#                    containment / optimistic assessment)
#
# Cation sorption uncertainty is ~1 order of magnitude each side of mean.
# Anionic / near-zero Kd species (TcO4-, I-, UO2(CO3)3^4-) have lower bound
# of 0 (effectively non-sorbing) since literature minima approach zero.
# Strongly-sorbing reduced species (Th, Pu(IV), U(IV)) span ~2 orders of mag.
#
# Sources: SKB TR-10-40 (2011); Bradbury & Baeyens (2003) J Contam Hydrol;
#          EPA NCRP-152 (2007); Tochiyama et al. (1995); Yui et al. (1999);
#          NEA/OECD TDB reviews (2003–2020).
#
# Structure: KD[species_key][rock_type] = {"low": x, "mean": y, "high": z}

def _kd(lo, mean, hi):
    return {"low": lo, "mean": mean, "high": hi}

KD = {
    "Csp": {
        "granite":   _kd(5e-3,  5e-2,  5e-1),
        "sandstone": _kd(2e-3,  2e-2,  2e-1),
        "shale":     _kd(5e-2,  5e-1,  5.0),
        "carbonate": _kd(1e-3,  1e-2,  1e-1),
        "bentonite": _kd(5.0,   50.0,  500.0),
    },
    "Sr2p": {
        "granite":   _kd(1e-4,  1e-3,  1e-2),
        "sandstone": _kd(5e-5,  5e-4,  5e-3),
        "shale":     _kd(5e-4,  5e-3,  5e-2),
        "carbonate": _kd(5e-4,  5e-3,  5e-2),
        "bentonite": _kd(1e-2,  1e-1,  1.0),
    },
    "Ra2p": {
        "granite":   _kd(1e-3,  1e-2,  1e-1),
        "sandstone": _kd(5e-4,  5e-3,  5e-2),
        "shale":     _kd(1e-2,  1e-1,  1.0),
        "carbonate": _kd(1e-2,  1e-1,  1.0),
        "bentonite": _kd(5e-1,  5.0,   50.0),
    },
    "Im": {
        "granite":   _kd(0.0,   1e-4,  5e-3),
        "sandstone": _kd(0.0,   5e-5,  2e-3),
        "shale":     _kd(0.0,   1e-4,  5e-3),
        "carbonate": _kd(0.0,   5e-5,  2e-3),
        "bentonite": _kd(0.0,   1e-3,  1e-2),
    },
    "IO3m": {
        "granite":   _kd(1e-4,  1e-3,  1e-2),
        "sandstone": _kd(5e-5,  5e-4,  5e-3),
        "shale":     _kd(2e-4,  2e-3,  2e-2),
        "carbonate": _kd(1e-4,  1e-3,  1e-2),
        "bentonite": _kd(5e-4,  5e-3,  5e-2),
    },
    "TcO4m": {
        "granite":   _kd(0.0,   1e-4,  2e-3),
        "sandstone": _kd(0.0,   1e-4,  2e-3),
        "shale":     _kd(0.0,   5e-4,  5e-3),
        "carbonate": _kd(0.0,   1e-4,  2e-3),
        "bentonite": _kd(0.0,   1e-3,  1e-2),
    },
    "Tc4_ox": {
        "granite":   _kd(1e-3,  1e-2,  1e-1),
        "sandstone": _kd(5e-4,  5e-3,  5e-2),
        "shale":     _kd(1e-2,  1e-1,  1.0),
        "carbonate": _kd(5e-4,  5e-3,  5e-2),
        "bentonite": _kd(1e-1,  1.0,   10.0),
    },
    "UO2_2p": {
        "granite":   _kd(1e-3,  1e-2,  1e-1),
        "sandstone": _kd(5e-4,  5e-3,  5e-2),
        "shale":     _kd(5e-3,  5e-2,  5e-1),
        "carbonate": _kd(5e-4,  5e-3,  5e-2),
        "bentonite": _kd(5e-2,  5e-1,  5.0),
    },
    "UO2CO3_4m": {
        "granite":   _kd(0.0,   1e-4,  2e-3),
        "sandstone": _kd(0.0,   5e-5,  1e-3),
        "shale":     _kd(0.0,   5e-4,  5e-3),
        "carbonate": _kd(0.0,   1e-4,  2e-3),
        "bentonite": _kd(0.0,   1e-3,  1e-2),
    },
    "U4": {
        "granite":   _kd(2e-2,  1e-1,  5e-1),
        "sandstone": _kd(1e-2,  5e-2,  2e-1),
        "shale":     _kd(2e-1,  1.0,   5.0),
        "carbonate": _kd(1e-2,  5e-2,  2e-1),
        "bentonite": _kd(2.0,   10.0,  50.0),
    },
    "NpO2p": {
        "granite":   _kd(1e-3,  1e-2,  1e-1),
        "sandstone": _kd(5e-4,  5e-3,  5e-2),
        "shale":     _kd(5e-3,  5e-2,  5e-1),
        "carbonate": _kd(5e-4,  5e-3,  5e-2),
        "bentonite": _kd(5e-2,  5e-1,  5.0),
    },
    "NpO2_2p": {
        "granite":   _kd(1e-3,  1e-2,  1e-1),
        "sandstone": _kd(5e-4,  5e-3,  5e-2),
        "shale":     _kd(5e-3,  5e-2,  5e-1),
        "carbonate": _kd(5e-4,  5e-3,  5e-2),
        "bentonite": _kd(5e-2,  5e-1,  5.0),
    },
    "Np4": {
        "granite":   _kd(1e-1,  1.0,   10.0),
        "sandstone": _kd(5e-2,  5e-1,  5.0),
        "shale":     _kd(1.0,   10.0,  100.0),
        "carbonate": _kd(5e-2,  5e-1,  5.0),
        "bentonite": _kd(5.0,   50.0,  500.0),
    },
    "PuO2p": {
        "granite":   _kd(5e-3,  5e-2,  5e-1),
        "sandstone": _kd(2e-3,  2e-2,  2e-1),
        "shale":     _kd(5e-2,  5e-1,  5.0),
        "carbonate": _kd(2e-3,  2e-2,  2e-1),
        "bentonite": _kd(5e-1,  5.0,   50.0),
    },
    "PuO2_2p": {
        "granite":   _kd(5e-3,  5e-2,  5e-1),
        "sandstone": _kd(2e-3,  2e-2,  2e-1),
        "shale":     _kd(5e-2,  5e-1,  5.0),
        "carbonate": _kd(2e-3,  2e-2,  2e-1),
        "bentonite": _kd(5e-1,  5.0,   50.0),
    },
    "Pu4_ox": {
        "granite":   _kd(2.0,   10.0,  50.0),
        "sandstone": _kd(1.0,   5.0,   25.0),
        "shale":     _kd(10.0,  50.0,  250.0),
        "carbonate": _kd(1.0,   5.0,   25.0),
        "bentonite": _kd(100.0, 500.0, 2000.0),
    },
    "Pu3": {
        "granite":   _kd(5e-2,  5e-1,  5.0),
        "sandstone": _kd(2e-2,  2e-1,  2.0),
        "shale":     _kd(5e-1,  5.0,   50.0),
        "carbonate": _kd(2e-2,  2e-1,  2.0),
        "bentonite": _kd(2.0,   20.0,  200.0),
    },
    "Th4": {
        "granite":   _kd(2.0,   10.0,  50.0),
        "sandstone": _kd(1.0,   5.0,   25.0),
        "shale":     _kd(10.0,  50.0,  250.0),
        "carbonate": _kd(1.0,   5.0,   25.0),
        "bentonite": _kd(100.0, 500.0, 2000.0),
    },
    "SeO4_2m": {
        "granite":   _kd(0.0,   1e-4,  2e-3),
        "sandstone": _kd(0.0,   5e-4,  5e-3),
        "shale":     _kd(0.0,   1e-3,  1e-2),
        "carbonate": _kd(0.0,   1e-3,  1e-2),
        "bentonite": _kd(0.0,   1e-2,  1e-1),
    },
    "SeO3_2m": {
        "granite":   _kd(1e-2,  1e-1,  1.0),
        "sandstone": _kd(5e-3,  5e-2,  5e-1),
        "shale":     _kd(5e-2,  5e-1,  5.0),
        "carbonate": _kd(5e-3,  5e-2,  5e-1),
        "bentonite": _kd(5e-1,  5.0,   50.0),
    },
    "HSem": {
        "granite":   _kd(1e-2,  1e-1,  1.0),
        "sandstone": _kd(5e-3,  5e-2,  5e-1),
        "shale":     _kd(5e-2,  5e-1,  5.0),
        "carbonate": _kd(5e-3,  5e-2,  5e-1),
        "bentonite": _kd(5e-1,  5.0,   50.0),
    },
    "Rn_aq": {
        "granite":   _kd(0.0, 0.0, 0.0),
        "sandstone": _kd(0.0, 0.0, 0.0),
        "shale":     _kd(0.0, 0.0, 0.0),
        "carbonate": _kd(0.0, 0.0, 0.0),
        "bentonite": _kd(0.0, 0.0, 0.0),
    },
}

# Grain density [kg/m³] by rock type
GRAIN_DENSITY = {
    "granite": 2650, "sandstone": 2650, "shale": 2650,
    "carbonate": 2710, "bentonite": 2750,
}

# ── Rock thermal properties ───────────────────────────────────────────────────
# lambda_rock [W/(m·K)], rho_cp_rock [J/(m³·K)] = ρ·cp
# Sources: Clauser & Huenges (1995) Rock Physics Handbook;
#          Robertson (1988) Geol Soc Spec Pub
ROCK_THERMAL = {
    "granite":   {"lambda_rock": 3.0, "rho_cp_rock": 2.0e6},
    "sandstone": {"lambda_rock": 2.5, "rho_cp_rock": 2.1e6},
    "shale":     {"lambda_rock": 1.5, "rho_cp_rock": 2.2e6},
    "carbonate": {"lambda_rock": 2.8, "rho_cp_rock": 2.1e6},
    "bentonite": {"lambda_rock": 0.5, "rho_cp_rock": 1.5e6},
}

# ── Half-lives in seconds ─────────────────────────────────────────────────────
# Used for Damköhler number: Da = λ_decay · L · R / v_aq
# λ_decay = ln(2) / t_{1/2}
_YR_S = 3.15576e7   # seconds per year
HALFLIFE_S = {
    "U":   4.47e9  * _YR_S,
    "Tc":  2.13e5  * _YR_S,
    "I":   1.57e7  * _YR_S,
    "Np":  2.14e6  * _YR_S,
    "Pu":  2.41e4  * _YR_S,
    "Cs":  30.17   * _YR_S,
    "Sr":  28.8    * _YR_S,
    "Ra":  1.600e3 * _YR_S,
    "Th":  1.41e10 * _YR_S,
    "Se":  3.27e5  * _YR_S,
    "Rn":  3.82    * 86400,   # 3.82 days
}

# ── Pure-Python fluid property helpers (replaces CoolProp) ───────────────────

def _iapws_state(T_K: float, P_Pa: float):
    """Return IAPWS97 water state; None on failure."""
    try:
        return _IAPWS97(T=max(274.0, min(1073.0, T_K)),
                        P=max(1e-4, min(100.0, P_Pa * 1e-6)))
    except Exception:
        return None

def _pr_eos(T_K: float, P_Pa: float, Tc: float, Pc_Pa: float,
            omega: float, Mw: float):
    """Peng-Robinson EOS. Returns (rho_kg_m3, Z, fugacity_Pa)."""
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    alpha = (1 + kappa * (1 - (T_K / Tc) ** 0.5)) ** 2
    a = 0.45724 * R ** 2 * Tc ** 2 / Pc_Pa * alpha
    b = 0.07780 * R * Tc / Pc_Pa
    A = a * P_Pa / (R * T_K) ** 2
    B = b * P_Pa / (R * T_K)
    coeffs = [1, -(1 - B), A - 3 * B ** 2 - 2 * B, -(A * B - B ** 2 - B ** 3)]
    roots = np.roots(coeffs)
    reals = sorted([r.real for r in roots if abs(r.imag) < 1e-8 and r.real > B])
    Z = reals[-1] if reals else 1.0
    V = Z * R * T_K / P_Pa
    rho = Mw / V
    ln_phi = (Z - 1 - np.log(max(Z - B, 1e-12))
              - A / (2 * 2 ** 0.5 * B)
              * np.log((Z + (1 + 2 ** 0.5) * B) / max(Z + (1 - 2 ** 0.5) * B, 1e-12)))
    fug = P_Pa * np.exp(np.clip(ln_phi, -20, 20))
    return _safe(rho), Z, _safe(fug)

def _gas_viscosity_uPas(T_K: float, P_Pa: float,
                        Tc: float, Pc_Pa: float, mu25: float) -> float:
    """Gas viscosity [μPa·s]: T^0.7 scaling + pressure correction."""
    mu0 = mu25 * (T_K / 298.15) ** 0.7
    Pr = P_Pa / Pc_Pa
    Tr = T_K / Tc
    correction = 1.0 + max(0.0, Pr / (5.0 * max(Tr, 0.5))) * 0.5
    return mu0 * correction

def _water_vapor_pressure_Pa(T_K: float) -> float:
    """Water saturation pressure [Pa] via IAPWS-97."""
    try:
        sat = _IAPWS97(T=max(274.0, min(646.0, T_K)), x=0)
        return sat.P * 1e6
    except Exception:
        t = T_K - 273.15
        return 10 ** (8.07131 - 1730.63 / (233.426 + t)) * 133.322

# ── Utility helpers ───────────────────────────────────────────────────────────

def _safe(v):
    return float(v) if np.isfinite(v) else None


def _water_viscosity(T_K, P_Pa=1e5):
    """Liquid water dynamic viscosity [Pa·s] via IAPWS-97."""
    st = _iapws_state(T_K, P_Pa)
    if st is not None and st.mu and np.isfinite(st.mu):
        return st.mu
    t = T_K - 273.15
    log_r = ((20 - t) / (t + 96)) * (
        1.2378 - 1.303e-3 * (20 - t) + 3.06e-6 * (20 - t)**2
        + 2.55e-8 * (20 - t)**3
    )
    return 1.002e-3 * 10**log_r


def _brine_viscosity(T_K, P_Pa, S_g_L):
    """Brine viscosity [Pa·s] via Mao & Duan (2009) correction."""
    mu_w = _water_viscosity(T_K, P_Pa)
    m = S_g_L / (Mw_NaCl * 1000)
    A, B = 1.9741e-2, -4.22e-5
    return mu_w * np.exp(m * (A + B * T_K))


# ── Speciation ────────────────────────────────────────────────────────────────

def speciate(rn_id: str, Eh_mV: float, pH: float) -> str:
    """Return dominant species key for radionuclide at given Eh/pH."""
    rules = SPECIATION_RULES[rn_id]["rules"]
    for (Eh_lo, Eh_hi, pH_lo, pH_hi, species, _desc) in rules:
        if (Eh_lo is None or Eh_mV >= Eh_lo) and \
           (Eh_hi is None or Eh_mV <  Eh_hi) and \
           (pH_lo is None or pH    >= pH_lo) and \
           (pH_hi is None or pH    <  pH_hi):
            return species
    return rules[-1][4]   # fallback: last rule


# ── Aqueous molecular diffusivity ─────────────────────────────────────────────

def aqueous_diffusivity(species_key: str, T_K: float, mu_w_Pas: float) -> float:
    """
    Stokes-Einstein temperature/viscosity scaling from D_25 reference.
        D(T) = D_25 × (T / 298.15) × (μ_25 / μ(T))
    Returns D_m [m²/s].
    """
    sp = SPECIES_PROPS[species_key]
    mu25 = _water_viscosity(298.15, 1e5)
    if mu_w_Pas <= 0 or not np.isfinite(mu_w_Pas):
        mu_w_Pas = mu25
    return sp["D25"] * (T_K / 298.15) * (mu25 / mu_w_Pas)


# ── Effective diffusivity with tortuosity ────────────────────────────────────

def effective_diffusivity(D_m: float, porosity: float, tortuosity: float,
                           v_m_s: float, dispersivity_m: float,
                           S_w: float = 1.0) -> dict:
    """
    D_eff = D_mol + D_mech
    D_mol = D_m * φ * S_w / τ          (Millington-Quirk-style; τ = user tortuosity)
    D_mech = α_L * v                   (longitudinal mechanical dispersion)
    In multiphase: saturation S_w reduces the diffusive term.
    """
    D_mol  = D_m * porosity * S_w / tortuosity
    D_mech = dispersivity_m * v_m_s
    D_eff  = D_mol + D_mech
    return {
        "D_mol_m2s":  _safe(D_mol),
        "D_mech_m2s": _safe(D_mech),
        "D_eff_m2s":  _safe(D_eff),
    }


# ── Retardation factor ────────────────────────────────────────────────────────

def retardation_factor(species_key: str, rock_type: str,
                        porosity: float,
                        kd_scenario: str = "mean") -> dict:
    """
    R = 1 + (ρ_b / φ) * Kd
    ρ_b = (1 − φ) * ρ_grain   [dry bulk density, kg/m³]
    Kd in m³/kg (converted from L/kg).

    kd_scenario : "mean"         — geometric-mean literature value
                  "conservative" — ~10th-percentile low Kd (fast transport,
                                   worst case for migration)
                  "generous"     — ~90th-percentile high Kd (slow transport,
                                   best case for containment)
    """
    kd_entry = KD.get(species_key, {}).get(rock_type, {"low": 0.0, "mean": 0.0, "high": 0.0})
    key_map  = {"conservative": "low", "mean": "mean", "generous": "high"}
    Kd_L_kg  = kd_entry.get(key_map.get(kd_scenario, "mean"), 0.0)
    Kd_m3_kg = Kd_L_kg * 1e-3
    rho_grain = GRAIN_DENSITY.get(rock_type, 2650)
    rho_b = (1.0 - porosity) * rho_grain
    R = 1.0 + (rho_b / porosity) * Kd_m3_kg
    return {
        "Kd_L_kg":       _safe(Kd_L_kg),
        "Kd_low_L_kg":   _safe(kd_entry.get("low", 0.0)),
        "Kd_mean_L_kg":  _safe(kd_entry.get("mean", 0.0)),
        "Kd_high_L_kg":  _safe(kd_entry.get("high", 0.0)),
        "kd_scenario":   kd_scenario,
        "rho_b_kg_m3":   _safe(rho_b),
        "R":             _safe(R),
    }


# ── Péclet number & regime ────────────────────────────────────────────────────

def peclet_number(v_m_s: float, L_m: float, D_eff_m2s: float) -> float:
    """Pe = v · L / D_eff  (pore-scale or field-scale depending on L)."""
    if D_eff_m2s <= 0:
        return np.inf
    return v_m_s * L_m / D_eff_m2s


_REGIME_THRESHOLDS = [
    (0.01,   "Diffusion dominated",   "Pe < 0.01 — molecular diffusion controls",   "#2980b9"),
    (1.0,    "Transitional (diff.)",  "0.01 ≤ Pe < 1 — diffusion dominant",         "#27ae60"),
    (10.0,   "Mixed",                 "1 ≤ Pe < 10 — comparable diffusion & advection", "#f39c12"),
    (100.0,  "Transitional (adv.)",   "10 ≤ Pe < 100 — advection dominant",         "#e67e22"),
    (np.inf, "Advection dominated",   "Pe ≥ 100 — mechanical dispersion / plug flow","#e74c3c"),
]

def transport_regime(Pe: float) -> dict:
    """Return regime label, description, and color given Péclet number."""
    for threshold, label, desc, color in _REGIME_THRESHOLDS:
        if Pe < threshold:
            return {"regime": label, "description": desc, "color": color, "Pe": _safe(Pe)}
    return {"regime": "Advection dominated", "description": "Pe ≥ 100",
            "color": "#e74c3c", "Pe": _safe(Pe)}


# ── Gas phase properties ──────────────────────────────────────────────────────

def gas_phase_props(fluid_id: str, T_K: float, P_Pa: float) -> dict:
    """Density, viscosity, compressibility, and phase via Peng-Robinson EOS."""
    gf = GAS_FLUIDS[fluid_id]
    rho, Z, _ = _pr_eos(T_K, P_Pa, gf["Tc"], gf["Pc"], gf["omega"], gf["Mw"])
    mu  = _gas_viscosity_uPas(T_K, P_Pa, gf["Tc"], gf["Pc"], gf["mu25_uPas"])
    Ct_GPa = _safe(1.0 / (max(Z, 0.1) * max(P_Pa, 1e4)) * 1e9)
    phase = ("Supercritical" if T_K > gf["Tc"] and P_Pa > gf["Pc"]
             else "Gas" if T_K > gf["Tc"] or P_Pa < gf["Pc"]
             else "Liquid-like")
    return {"density_kg_m3": rho, "viscosity_uPas": _safe(mu),
            "compressibility_1_GPa": Ct_GPa, "phase": phase}


def gas_water_partition(fluid_id: str, T_K: float, P_MPa: float) -> dict:
    """
    Henry's law + KK correction for gas solubility in water.
    Returns mole fraction in water and dimensionless Henry constant K_H.
    """
    gf = GAS_FLUIDS[fluid_id]
    H_ref = 0.101325 / (gf["NIST_Hcp"] * Mw_H2O)
    H = H_ref / np.exp(gf["H_dT"] * (1.0 / T_K - 1.0 / 298.15))  # MPa
    Pv_MPa = _water_vapor_pressure_Pa(T_K) * 1e-6
    KK = H * np.exp(gf["Vmp"] * (P_MPa - Pv_MPa) * 1e6 / (R * T_K))
    _, _, fug_Pa = _pr_eos(T_K, P_MPa * 1e6, gf["Tc"], gf["Pc"], gf["omega"], gf["Mw"])
    fug_MPa = (fug_Pa or P_MPa * 1e6) * 1e-6
    xs = fug_MPa / KK
    mg_L = xs * (1.0 / Mw_H2O) * gf["Mw"] * 1e6
    return {
        "henry_MPa":     _safe(H),
        "kk_henry_MPa":  _safe(KK),
        "solubility_mol_frac": _safe(xs),
        "solubility_mg_L":     _safe(mg_L),
    }


# ── Rn-specific gas partitioning ─────────────────────────────────────────────
# Rn Henry constant from Sander (2015) Atm Chem Phys
_Rn_Hcp_298 = 9.3e-6   # mol/(m³·Pa) at 25°C → K_H^cc dimensionless = Hcp*R*T
_Rn_H_dT    = 2600     # K

def rn_partition_coefficient(T_K: float) -> float:
    """Dimensionless Rn gas/water partition coefficient (Henry volatility)."""
    Hcp = _Rn_Hcp_298 * np.exp(-_Rn_H_dT * (1.0 / T_K - 1.0 / 298.15))
    K_H_cc = 1.0 / (Hcp * R * T_K)   # Cgas / Cwater (dimensionless)
    return _safe(K_H_cc)


# ── CO2 pH effect ─────────────────────────────────────────────────────────────

def co2_acidification(T_K: float, P_MPa: float, P_CO2_frac: float = 1.0) -> dict:
    """
    Approximate pH depression from dissolved CO2.
    Uses Henry's law solubility + carbonate equilibrium (Ka1).
    P_CO2_frac = mole fraction of CO2 in gas phase (1.0 for pure CO2).
    """
    P_CO2_MPa = P_MPa * P_CO2_frac
    # Solubility: [CO2]_aq ≈ Kh * P_CO2
    H_ref = 0.101325 / (0.035 * Mw_H2O)  # MPa
    H = H_ref / np.exp(2400 * (1.0 / T_K - 1.0 / 298.15))
    CO2_aq_mol_L = (P_CO2_MPa / H) / Mw_H2O   # very rough

    # Ka1 at 25°C ≈ 4.3e-7; T correction ~ exp(-2000*(1/T-1/298))
    Ka1 = 4.3e-7 * np.exp(-2000 * (1.0 / T_K - 1.0 / 298.15))
    H_conc = np.sqrt(Ka1 * max(CO2_aq_mol_L, 1e-20))
    pH = -np.log10(max(H_conc, 1e-15))
    return {
        "CO2_aq_mol_L": _safe(CO2_aq_mol_L),
        "approx_pH_from_CO2": _safe(pH),
    }


# ── Single-phase transport analysis ──────────────────────────────────────────

# ── Damköhler number ─────────────────────────────────────────────────────────

def damkohler_number(rn_id: str, v_eff_m_s: float, L_m: float) -> dict:
    """
    Da_I = λ_decay × t_transport = λ_decay × (L / v_eff)

    Compares radioactive decay rate to advective transport rate over length L.
      Da << 1 : transport dominates — nuclide migrates distance L before
                significant decay (migration concern)
      Da >> 1 : decay dominates — nuclide decays in place before reaching L
                (containment by decay)

    Note: v_eff already incorporates retardation (v_eff = v/R), so Da reflects
    the competition between decay and *retarded* migration.
    """
    t12 = HALFLIFE_S.get(rn_id, None)
    if t12 is None or t12 <= 0:
        return {"Da": None, "lambda_decay_1_s": None, "t_transport_s": None,
                "interpretation": "Unknown half-life"}
    lam = np.log(2) / t12
    if v_eff_m_s is None or v_eff_m_s <= 0:
        return {"Da": None, "lambda_decay_1_s": _safe(lam), "t_transport_s": None,
                "interpretation": "Zero velocity"}
    t_trans = L_m / v_eff_m_s
    Da = lam * t_trans

    if Da < 0.01:
        interp = "Transport >> decay: migration concern"
    elif Da < 1:
        interp = "Transport > decay: migrates before significant decay"
    elif Da < 100:
        interp = "Decay ≈ transport: partial decay en route"
    else:
        interp = "Decay >> transport: decays before reaching L"

    return {
        "Da":               _safe(Da),
        "lambda_decay_1_s": _safe(lam),
        "t_transport_s":    _safe(t_trans),
        "t_transport_yr":   _safe(t_trans / _YR_S),
        "halflife_yr":      _safe(t12 / _YR_S),
        "interpretation":   interp,
    }


# ── Thermal convection (Rayleigh-Darcy) ──────────────────────────────────────

def calculate_thermal_convection(
    T_C: float, P_MPa: float,
    dT_dz_C_m: float,      # thermal gradient [°C/m], positive = warming downward
    H_m: float,            # domain height / vertical extent of convection cell [m]
    aperture_um: float,    # hydraulic fracture aperture [μm]; k = b²/12
    salinity_g_L: float,
    rock_type: str = "granite",
) -> dict:
    """
    Rayleigh-Darcy number for thermal convection in a single fracture.

    Ra_D = k_f · ρ_f · g · β_T · (dT/dz) · H / (μ_f · α_rock)

    where
      k_f    = b² / 12                       fracture (parallel-plate) permeability
      β_T    = isobaric thermal expansion of brine  [1/K]
      α_rock = λ_rock / (ρ·c_p)_rock        thermal diffusivity of host rock [m²/s]

    Using rock thermal diffusivity (not fluid) follows the Horton-Rogers-Lapwood
    convention for porous media where heat storage is dominated by the solid matrix.

    Critical value (horizontal layer, HRL theory): Ra_c = 4π² ≈ 39.5
    For a dipping or vertical fracture onset is lower; Ra_c ≈ 4π² is conservative.

    Buoyancy velocity estimate (order of magnitude):
      v_buoy ~ k_f · ρ_f · g · β_T · ΔT / μ_f    [m/s]

    Nuclear decay heat context (see note dict below):
      Near-field repository thermal gradient varies from ~10–50 °C/m (0–10 yr
      post-emplacement) to ~0.03 °C/m (>10 000 yr, approaches geothermal).
      References: NAGRA NTB 02-05; SKB TR-10-67; Tsang et al. (2015).
    """
    T_K  = T_C + 273.15
    P_Pa = P_MPa * 1e6

    # ── Fluid properties via IAPWS-97 ─────────────────────────────────────────
    _ws    = _iapws_state(T_K, P_Pa)
    rho_f  = (_ws.rho   if _ws else None) or 1000.0
    cp_f   = ((_ws.cp * 1e3) if _ws else None) or 4200.0   # kJ→J/(kg·K)
    lam_f  = (_ws.k     if _ws else None) or 0.6
    beta_T = (_ws.alfav if _ws else None) or 2.1e-4

    mu_f = _brine_viscosity(T_K, P_Pa, salinity_g_L)

    # Fluid thermal diffusivity
    alpha_f = lam_f / (rho_f * cp_f)   # m²/s

    # ── Rock thermal properties ───────────────────────────────────────────────
    rt = ROCK_THERMAL.get(rock_type, ROCK_THERMAL["granite"])
    alpha_rock = rt["lambda_rock"] / rt["rho_cp_rock"]   # m²/s

    # ── Fracture permeability ─────────────────────────────────────────────────
    b_m = aperture_um * 1e-6          # aperture in metres
    k_f = b_m**2 / 12.0              # parallel-plate permeability [m²]

    # ── Rayleigh-Darcy number (rock diffusivity basis) ────────────────────────
    Ra_rock = (k_f * rho_f * 9.81 * abs(beta_T) * abs(dT_dz_C_m) * H_m
               / (mu_f * alpha_rock))

    # Also report fluid-diffusivity basis for comparison
    Ra_fluid = (k_f * rho_f * 9.81 * abs(beta_T) * abs(dT_dz_C_m) * H_m
                / (mu_f * alpha_f))

    Ra_c = 4.0 * np.pi**2   # ≈ 39.478

    # ── Buoyancy velocity estimate ────────────────────────────────────────────
    delta_T   = abs(dT_dz_C_m) * H_m
    v_buoy    = k_f * rho_f * 9.81 * abs(beta_T) * delta_T / mu_f   # m/s
    v_buoy_yr = v_buoy * _YR_S

    # ── Nusselt number (simple Nusselt-Rayleigh correlation) ─────────────────
    # For Ra < Ra_c: Nu = 1 (conduction only)
    # For Ra > Ra_c: Nu ≈ (Ra / Ra_c)^0.5  (basic scaling, Nield & Bejan 2006)
    Ra_ref = Ra_rock
    Nu = 1.0 if Ra_ref <= Ra_c else (Ra_ref / Ra_c)**0.5

    # ── Convective heat flux estimate ─────────────────────────────────────────
    # q_conv ≈ Nu · λ_rock · (dT/dz)  [W/m²]
    q_conv = Nu * rt["lambda_rock"] * abs(dT_dz_C_m)

    # ── Status ────────────────────────────────────────────────────────────────
    ratio = Ra_ref / Ra_c
    if ratio < 0.5:
        status = "Conduction dominated — no convection"
        status_color = "#2980b9"
    elif ratio < 1.0:
        status = "Sub-critical — approaching onset"
        status_color = "#f39c12"
    elif ratio < 5.0:
        status = "Super-critical — weak convection"
        status_color = "#e67e22"
    else:
        status = "Strongly convective"
        status_color = "#e74c3c"

    # ── Nuclear decay thermal gradient reference table ────────────────────────
    decay_heat_ref = [
        {"scenario": "Natural geothermal",       "dTdz_min": 0.025, "dTdz_max": 0.030, "note": "Baseline"},
        {"scenario": "Repository far-field",      "dTdz_min": 0.03,  "dTdz_max": 0.10,  "note": "100–10 000 yr"},
        {"scenario": "Repository near-field",     "dTdz_min": 0.10,  "dTdz_max": 1.0,   "note": "100–1 000 yr"},
        {"scenario": "Near-field (early period)", "dTdz_min": 1.0,   "dTdz_max": 10.0,  "note": "10–100 yr"},
        {"scenario": "Immediate post-closure",    "dTdz_min": 10.0,  "dTdz_max": 50.0,  "note": "0–10 yr"},
    ]

    return {
        "T_C":             T_C,
        "P_MPa":           P_MPa,
        "dT_dz_C_m":       dT_dz_C_m,
        "H_m":             H_m,
        "aperture_um":     aperture_um,
        "rock_type":       rock_type,
        # Fluid props
        "rho_f_kg_m3":     _safe(rho_f),
        "mu_f_mPas":       _safe(mu_f * 1e3),
        "beta_T_1_K":      _safe(beta_T),
        "alpha_f_m2s":     _safe(alpha_f),
        "lambda_f_W_mK":   _safe(lam_f),
        # Rock thermal
        "lambda_rock_W_mK":_safe(rt["lambda_rock"]),
        "alpha_rock_m2s":  _safe(alpha_rock),
        # Fracture
        "k_f_m2":          _safe(k_f),
        # Rayleigh numbers
        "Ra_rock":         _safe(Ra_rock),
        "Ra_fluid":        _safe(Ra_fluid),
        "Ra_c":            _safe(Ra_c),
        "Ra_ratio":        _safe(ratio),
        "status":          status,
        "status_color":    status_color,
        # Convection metrics
        "Nu":              _safe(Nu),
        "v_buoy_m_s":      _safe(v_buoy),
        "v_buoy_m_yr":     _safe(v_buoy_yr),
        "q_conv_W_m2":     _safe(q_conv),
        "delta_T_K":       _safe(delta_T),
        # Reference
        "decay_heat_ref":  decay_heat_ref,
    }


def _v_from_Q(Q_ml_hr: float, core_diam_mm: float,
               porosity: float, S_w: float = 1.0) -> float:
    """
    Convert volumetric flow rate to aqueous-phase interstitial velocity.
        v_int = Q / (A · φ · S_w)
    Q is the total injected flow rate; S_w scales for the fraction of pore
    space occupied by the aqueous phase in multiphase flow.
    Returns velocity in m/s.
    """
    Q = Q_ml_hr * 1e-6 / 3600          # m³/s
    r = core_diam_mm * 0.5e-3           # m
    A = np.pi * r**2                    # m²
    return Q / (A * porosity * S_w)


def calculate_single_phase(
    T_C: float, P_MPa: float,
    v_m_s: float,
    rn_ids: list,
    rock_type: str,
    porosity: float,
    tortuosity: float,
    dispersivity_m: float,
    char_len_m: float,
    Eh_mV: float,
    pH: float,
    salinity_g_L: float,
    kd_scenario: str = "mean",
    # Optional: if provided, overrides v_m_s for transport calculations
    Q_ml_hr: float | None = None,
    core_diam_mm: float = 25.4,
) -> dict:
    """
    Single-phase aqueous transport analysis.

    Interstitial velocity refers to the aqueous phase: v = q / φ, where q is
    the Darcy flux.  If Q_ml_hr (volumetric flow rate) and core_diam_mm are
    provided, v is computed from Q and overrides the manual v_m_s entry.
    """
    T_K  = T_C + 273.15
    P_Pa = P_MPa * 1e6

    mu_w = _brine_viscosity(T_K, P_Pa, salinity_g_L)
    _ws_rho = _iapws_state(T_K, P_Pa)
    rho_w = _ws_rho.rho if _ws_rho else None

    # Resolve velocity: Q overrides manual v if provided
    v_source = "manual"
    if Q_ml_hr is not None and Q_ml_hr > 0 and core_diam_mm > 0:
        v_m_s   = _v_from_Q(Q_ml_hr, core_diam_mm, porosity, S_w=1.0)
        v_source = "derived from Q"

    results = {}
    for rn_id in rn_ids:
        sp_key  = speciate(rn_id, Eh_mV, pH)
        sp_info = SPECIES_PROPS[sp_key]

        D_m   = aqueous_diffusivity(sp_key, T_K, mu_w)
        D_out = effective_diffusivity(D_m, porosity, tortuosity,
                                       v_m_s, dispersivity_m, S_w=1.0)
        R_out = retardation_factor(sp_key, rock_type, porosity, kd_scenario)

        D_eff = D_out["D_eff_m2s"]
        R     = R_out["R"]
        v_eff = v_m_s / R if (R and R > 0) else None
        Pe    = peclet_number(v_m_s, char_len_m, D_eff) if D_eff else np.inf
        regime = transport_regime(Pe)

        # Dispersivity contribution as fraction of D_eff
        D_mech = D_out["D_mech_m2s"] or 0.0
        D_mol  = D_out["D_mol_m2s"]  or 0.0
        disp_frac = D_mech / (D_mech + D_mol) if (D_mech + D_mol) > 0 else 0.0

        da_out = damkohler_number(rn_id, v_eff, char_len_m)

        results[rn_id] = {
            "label":           SPECIATION_RULES[rn_id]["label"],
            "halflife":        SPECIATION_RULES[rn_id]["halflife"],
            "species_key":     sp_key,
            "species_label":   sp_info["label"],
            "charge":          sp_info["charge"],
            "D_m_m2s":         _safe(D_m),
            "diffusivity":     D_out,
            "disp_frac":       _safe(disp_frac),
            "retardation":     R_out,
            "v_eff_m_s":       _safe(v_eff),
            "v_eff_m_yr":      _safe(v_eff * _YR_S) if v_eff else None,
            "peclet":          regime,
            "damkohler":       da_out,
        }

    return {
        "T_C":             T_C,
        "P_MPa":           P_MPa,
        "v_m_s":           v_m_s,
        "v_source":        v_source,
        "Q_ml_hr":         Q_ml_hr,
        "core_diam_mm":    core_diam_mm,
        "kd_scenario":     kd_scenario,
        "rock_type":       rock_type,
        "porosity":        porosity,
        "tortuosity":      tortuosity,
        "dispersivity_m":  dispersivity_m,
        "char_len_m":      char_len_m,
        "Eh_mV":           Eh_mV,
        "pH":              pH,
        "salinity_g_L":    salinity_g_L,
        "mu_w_mPas":       _safe(mu_w * 1e3),
        "rho_w_kg_m3":     _safe(rho_w),
        "radionuclides":   results,
    }


# ── Multiphase transport analysis ─────────────────────────────────────────────

def calculate_multiphase(
    T_C: float, P_MPa: float,
    v_m_s: float,
    rn_ids: list,
    rock_type: str,
    porosity: float,
    tortuosity: float,
    dispersivity_m: float,
    char_len_m: float,
    Eh_mV: float,
    pH: float,
    salinity_g_L: float,
    # Multiphase extras
    gas_fluid_id: str,
    S_w: float,        # water saturation (0–1)
    Q_ml_hr: float | None = None,
    core_diam_mm: float = 25.4,
    theta_deg: float = 10.0,
    kd_scenario: str = "mean",
) -> dict:
    T_K  = T_C + 273.15
    P_Pa = P_MPa * 1e6

    mu_w = _brine_viscosity(T_K, P_Pa, salinity_g_L)
    _ws_rho = _iapws_state(T_K, P_Pa)
    rho_w = _ws_rho.rho if _ws_rho else None

    # Gas phase
    gas = gas_phase_props(gas_fluid_id, T_K, P_Pa)
    gas["fluid"]  = gas_fluid_id
    gas["label"]  = GAS_FLUIDS[gas_fluid_id]["label"]
    gas["color"]  = GAS_FLUIDS[gas_fluid_id]["color"]
    gas_sol       = gas_water_partition(gas_fluid_id, T_K, P_MPa)

    # pH modification if CO2
    co2_effect = None
    if gas_fluid_id == "CO2" and gas_sol.get("solubility_mol_frac"):
        co2_effect = co2_acidification(T_K, P_MPa)

    # Rn-specific partition
    rn_Kcc = rn_partition_coefficient(T_K) if "Rn" in rn_ids else None

    # Resolve velocity: Q overrides manual v if provided.
    # In multiphase, v is the aqueous-phase interstitial velocity:
    #   v_aq = Q / (A · φ · S_w)
    v_source = "manual"
    if Q_ml_hr is not None and Q_ml_hr > 0 and core_diam_mm > 0:
        v_m_s   = _v_from_Q(Q_ml_hr, core_diam_mm, porosity, S_w=S_w)
        v_source = "derived from Q (aqueous phase)"

    # Aqueous saturation reduces effective pore volume and diffusivity
    results = {}
    for rn_id in rn_ids:
        sp_key  = speciate(rn_id, Eh_mV, pH)
        sp_info = SPECIES_PROPS[sp_key]

        D_m   = aqueous_diffusivity(sp_key, T_K, mu_w)
        D_out = effective_diffusivity(D_m, porosity, tortuosity,
                                       v_m_s, dispersivity_m, S_w=S_w)
        R_out = retardation_factor(sp_key, rock_type, porosity, kd_scenario)

        D_eff = D_out["D_eff_m2s"]
        R     = R_out["R"]
        v_eff = v_m_s / R if (R and R > 0) else None
        Pe    = peclet_number(v_m_s, char_len_m, D_eff) if D_eff else np.inf
        regime = transport_regime(Pe)

        D_mech = D_out["D_mech_m2s"] or 0.0
        D_mol  = D_out["D_mol_m2s"]  or 0.0
        disp_frac = D_mech / (D_mech + D_mol) if (D_mech + D_mol) > 0 else 0.0

        # Rn gas phase fraction
        gas_frac = None
        if rn_id == "Rn" and rn_Kcc is not None:
            S_g = 1.0 - S_w
            gas_frac = (S_g * rn_Kcc) / (S_w + S_g * rn_Kcc) if (S_w + S_g * rn_Kcc) > 0 else 0.0

        da_out = damkohler_number(rn_id, v_eff, char_len_m)

        results[rn_id] = {
            "label":          SPECIATION_RULES[rn_id]["label"],
            "halflife":       SPECIATION_RULES[rn_id]["halflife"],
            "species_key":    sp_key,
            "species_label":  sp_info["label"],
            "charge":         sp_info["charge"],
            "D_m_m2s":        _safe(D_m),
            "diffusivity":    D_out,
            "disp_frac":      _safe(disp_frac),
            "retardation":    R_out,
            "v_eff_m_s":      _safe(v_eff),
            "v_eff_m_yr":     _safe(v_eff * _YR_S) if v_eff else None,
            "peclet":         regime,
            "damkohler":      da_out,
            "gas_phase_frac": _safe(gas_frac) if gas_frac is not None else None,
        }

    # Dimensionless flow numbers (always compute if Q provided)
    flow_numbers = None
    if (Q_ml_hr is not None and rho_w and
            gas.get("density_kg_m3") and gas.get("viscosity_uPas")):
        try:
            rho_nw   = gas["density_kg_m3"]
            ift_mN_m = 72.8 - 0.17 * T_C   # rough freshwater IFT [mN/m]
            cos_theta = max(np.cos(np.radians(theta_deg)), 1e-9)
            Ca = v_m_s * (gas["viscosity_uPas"] * 1e-6) / (ift_mN_m * 1e-3 * cos_theta)
            Bo = abs(rho_w - rho_nw) * 9.81 * (char_len_m**2) / (ift_mN_m * 1e-3 * cos_theta)
            M  = (gas["viscosity_uPas"] * 1e-6) / mu_w
            flow_numbers = {
                "Ca": _safe(Ca), "Bo": _safe(Bo),
                "M_drainage": _safe(M),
                "v_aq_m_s": _safe(v_m_s),
                "ift_mN_m": _safe(ift_mN_m),
            }
        except Exception:
            pass

    return {
        "T_C":            T_C,
        "P_MPa":          P_MPa,
        "v_m_s":          v_m_s,
        "v_source":       v_source,
        "Q_ml_hr":        Q_ml_hr,
        "core_diam_mm":   core_diam_mm,
        "S_w":            S_w,
        "kd_scenario":    kd_scenario,
        "rock_type":      rock_type,
        "porosity":       porosity,
        "tortuosity":     tortuosity,
        "dispersivity_m": dispersivity_m,
        "char_len_m":     char_len_m,
        "Eh_mV":          Eh_mV,
        "pH":             pH,
        "salinity_g_L":   salinity_g_L,
        "mu_w_mPas":      _safe(mu_w * 1e3),
        "rho_w_kg_m3":    _safe(rho_w),
        "gas_phase":      gas,
        "gas_solubility": gas_sol,
        "co2_effect":     co2_effect,
        "rn_Kcc":         rn_Kcc,
        "flow_numbers":   flow_numbers,
        "radionuclides":  results,
    }
