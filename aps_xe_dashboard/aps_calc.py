"""Xe / DI-water convection + X-ray contrast estimates for APS beamtime.

Self-contained (no import from geo_transport) so it can deploy alone.

Sign convention that matters here: dissolved Xe makes water DENSER
(partial molar volume 46.5 cm3/mol carries 131.3 g/mol), so the
Xe-saturated layer sinks. Dissolution drives a Rayleigh-Darcy
instability downward from the Xe/water interface.

Vmp enters TWICE and in opposing directions: it raises the solution
volume (lowering Delta-rho) and it appears in the
Krichevsky-Kasarnovsky pressure correction (lowering solubility).
"""
import numpy as np
import CoolProp.CoolProp as CP

import xray

R = 8.314                 # J/mol/K
R_Latm = 0.08205616
Mw_H2O = 0.018015         # kg/mol
G = 9.81                  # m/s2
DARCY_M2 = 9.869233e-13   # 1 Darcy in m2
RA_CRIT = 4.0 * np.pi**2  # ~39.5, classic Horton-Rogers-Lapwood threshold

# --- xenon properties ---
XE = {
    "Tc": 289.733, "Pc_atm": 57.64, "omega": 0.0036,
    # Partial molar volume of dissolved Xe in water [cm3/mol].
    # 46.5 is the low-pressure, genuinely-dissolved value (cf. Moore et al. 1982,
    # 42-52 cm3/mol). Apparent values of ~125 cm3/mol fitted to HIGH-pressure
    # solubility data (e.g. J. Chem. Phys. 93, 2724, 1990) are attributed to
    # clathrate-hydrate formation: the fitted Vmp absorbs the hydrate
    # contribution and is not a true molecular partial molar volume.
    # Use this model only where hydrate is NOT stable -- see
    # hydrate_pressure_MPa(). Note also that buoyancy would reverse sign at
    # Vmp = M_Xe/rho_w ~ 131 cm3/mol.
    "Vmp": 46.5,
    "Mw": 0.13129,      # [kg/mol]
    # Henry solubility constant, here in mol/(kg.bar).
    # Source value is Hcp = 4.3e-5 mol/(m3.Pa) at 298.15 K (Abraham & Matteoli
    # 1988; Clever 1979b; compiled by Sander 2015 / henrys-law.org).
    # Unit conversion to mol/(kg.bar) is x100, because
    #   55510 mol/m3 (water) x 0.018015 kg/mol = 1000.
    # An earlier version used x10, making Xe solubility 10x too low.
    "NIST_Hcp": 4.3e-5 * 100,
    # van't Hoff coefficient -d ln(H)/d(1/T) [K], same sources: both give 2300 K
    # (Fernandez-Prini et al. 2003 gives 2200 K; published range 2200-2500 K).
    "H_dT": 2300.0,
    # Aqueous diffusivity at 25 C [m2/s]. Jahne, Heinz & Dietrich (1987),
    # J. Geophys. Res. 92, 10767. Ra is inversely proportional to this.
    "D_25": 1.47e-9,
}

def kozeny_carman(d_um, phi, C):
    """Kozeny-Carman permeability [Darcy] from pore size and porosity.

        k = C * phi^3 * d^2 / (1 - phi)^2

    C is a shape/tortuosity constant carrying the unit conversion, so it must
    be calibrated against a grade of KNOWN permeability (see kc_constant).
    d_um is taken as the geometric mean of the quoted pore-size range.
    """
    return C * phi**3 * d_um**2 / (1.0 - phi)**2


def kc_constant(d_um, phi, k_D):
    """Back out C from a grade with measured k. Inverse of kozeny_carman."""
    return k_D * (1.0 - phi)**2 / (phi**3 * d_um**2)


# All permeabilities here are estimates / nominal grade values, not measured
# values for the specific cores.
#
# ROBU00 (250-500 um) and ROBU0 (160-250 um) have no quoted permeability, so
# they are extrapolated by Kozeny-Carman anchored on ROBU1. They are assigned
# ROBU1's porosity (0.38), so phi^3/(1-phi)^2 cancels and KC reduces to
# k ~ d^2, with d the geometric mean of the pore-size range:
#     ROBU0  = 200 D * (200.0/126.5)^2 =  500 D
#     ROBU00 = 200 D * (353.6/126.5)^2 = 1563 D
# Calibrating instead on ROBU2/ROBU3 (whose C values agree to 0.6%, while
# ROBU1 sits ~21% below that trend) would scale both up by ~1.26x.
_KC_D_UM = {"ROBU00": (250, 500), "ROBU0": (160, 250), "ROBU1": (100, 160),
            "ROBU2": (40, 100), "ROBU3": (16, 40)}

PRESETS = {
    "ROBU00":     {"label": "ROBU00 (250-500 um pore)", "k_D": 1563.0, "phi": 0.38, "H_mm": 20.0},
    "ROBU0":      {"label": "ROBU0 (160-250 um pore)", "k_D": 500.0, "phi": 0.38, "H_mm": 20.0},
    "ROBU1":      {"label": "ROBU1 (100-160 um pore)", "k_D": 200.0, "phi": 0.38, "H_mm": 20.0},
    "ROBU2":      {"label": "ROBU2 (40-100 um pore)",  "k_D": 45.0,  "phi": 0.35, "H_mm": 20.0},
    "ROBU3":      {"label": "ROBU3 (16-40 um pore)",   "k_D": 5.0,   "phi": 0.32, "H_mm": 20.0},
    "BENTHEIMER": {"label": "Bentheimer Sandstone",    "k_D": 2.0,   "phi": 0.22, "H_mm": 20.0},
}


def _pr_eos(T_K, P_atm):
    """Peng-Robinson for Xe -> (molar volume cm3/mol, fugacity atm, phi)."""
    if P_atm <= 0:
        return np.nan, np.nan, np.nan
    Tc, Pc, w = XE["Tc"], XE["Pc_atm"], XE["omega"]
    kk = 0.37464 + 1.54226*w - 0.26992*w**2
    alpha = (1 + kk*(1-(T_K/Tc)**0.5))**2
    a = 0.45724*R_Latm**2*Tc**2/Pc*alpha
    b = 0.07780*R_Latm*Tc/Pc
    A = a*P_atm/(R_Latm**2*T_K**2)
    B = b*P_atm/(R_Latm*T_K)
    roots = np.roots([1, -(1-B), A-3*B**2-2*B, -(A*B-B**2-B**3)])
    valid = roots[np.isreal(roots)].real
    valid = valid[valid > B]
    if valid.size == 0:
        return np.nan, np.nan, np.nan
    Z = float(np.max(valid))
    vm = Z*R_Latm*T_K/P_atm*1000.0
    den = (Z+2.414*B)/(Z-0.414*B)
    if den <= 0 or Z <= B:
        return vm, np.nan, np.nan
    phi = np.exp(Z-1-np.log(Z-B)-A/(2*2**0.5*B)*np.log(den))
    return vm, phi*P_atm, phi


# Reference-quality Xe EoS: Lemmon & Span (2006), the Helmholtz-energy
# formulation behind CoolProp's "Xenon". Preferred over Peng-Robinson because
# Xe's Tc is 16.58 C -- at 20 C we sit at Tr = 1.012, just above critical,
# where cubic EoS are least accurate (PR is ~3.4% low in fugacity here).
_XE_STATE = None


def _fugacity_heos(T_K, P_Pa):
    """Xe fugacity [Pa] and fugacity coefficient from the Helmholtz EoS.

    For a pure fluid with residual Helmholtz energy alphar(tau, delta):
        ln(phi) = alphar + (Z - 1) - ln(Z),   Z = 1 + delta*(d alphar/d delta)
    Returns (nan, nan) if the state cannot be evaluated, so the caller can
    fall back to Peng-Robinson.
    """
    global _XE_STATE
    try:
        if _XE_STATE is None:
            _XE_STATE = CP.AbstractState("HEOS", "Xenon")
        _XE_STATE.update(CP.PT_INPUTS, P_Pa, T_K)
        ar = _XE_STATE.alphar()
        Z = 1.0 + _XE_STATE.delta()*_XE_STATE.dalphar_dDelta()
        if Z <= 0:
            return np.nan, np.nan
        phi = np.exp(ar + (Z - 1.0) - np.log(Z))
        return phi*P_Pa, phi
    except Exception:
        return np.nan, np.nan


def _henry_MPa(T_K):
    """Henry constant for Xe in water [MPa].

    Eq. (15) of Rasoolzadeh et al. (2020) [after Fernandez-Prini et al.]:
        ln(H/bar) = 39.273*(1 - 188.78/T) - 36.855*(1 - 188.78/T)^2 + ln(1.01325)
    Preferred over the two-parameter van't Hoff form (kept below as
    _henry_MPa_vanthoff) because it is a published correlation valid across the
    whole temperature range of the hydrate data, whereas the van't Hoff form
    drifts ~10% by 5 C and ~5% by 40 C. The two agree to 1.9% at 25 C, which
    independently corroborates the calibrated Hcp in XE.
    """
    z = 1.0 - 188.78/T_K
    return float(np.exp(39.273*z - 36.855*z*z + np.log(1.01325))*0.1)


def _henry_MPa_vanthoff(T_K):
    """Legacy van't Hoff form, retained for comparison against _henry_MPa."""
    H_ref = 0.101325/(XE["NIST_Hcp"]*Mw_H2O)
    return H_ref/np.exp(XE["H_dT"]*(1.0/T_K - 1.0/298.15))


HCP_SANDER_XE = 4.3e-5      # mol/(m3.Pa) at 298.15 K, Sander (2015) compilation
C_WATER = 55510.0           # mol/m3 of liquid water

# --- sI xenon clathrate hydrate stability boundary -------------------------
# MEASURED liquid-water / hydrate / vapour (Lw-H-V) three-phase equilibrium:
# all 87 points of Table 4 of Rasoolzadeh et al. (2020), Fluid Phase Equilibria
# 512, 112528, doi:10.1016/j.fluid.2020.112528 (tensimeter, Cailletet apparatus
# and high-pressure autoclave; u(T) 0.01-0.02 K, u(P) 0.025 kPa - 0.05 MPa).
# Covers 273.15-343.75 K and 0.153-376.01 MPa, spanning the whole design space.
#
# The curve has a pronounced knee near 310 K where the Xe-rich phase becomes
# dense, so a single Clausius-Clapeyron fit CANNOT represent it -- an earlier
# two-point fit here was 57% low at 40 C. Interpolate the data instead.
_HYD_LW_H_V = [
    (273.15, 0.15300), (273.49, 0.15949), (273.56, 0.15979), (273.82, 0.16405),
    (274.43, 0.17468), (274.75, 0.18016), (274.83, 0.18178), (275.74, 0.19789),
    (276.67, 0.21805), (277.66, 0.24075), (278.67, 0.26851), (279.56, 0.29526),
    (280.62, 0.32839), (281.66, 0.36416), (282.18, 0.39), (282.66, 0.40419),
    (283.28, 0.43), (283.62, 0.44654), (284.05, 0.47), (284.66, 0.49426),
    (285.17, 0.52), (285.66, 0.55253), (286.06, 0.59), (289.07, 0.78),
    (291.25, 0.97), (292.24, 1.06), (293.33, 1.19), (294.15, 1.29),
    (294.73, 1.37), (295.11, 1.42), (296.06, 1.58), (296.11, 1.57),
    (297.11, 1.75), (297.99, 1.93), (298.36, 1.99), (298.86, 2.09),
    (299.13, 2.17), (300.05, 2.37), (300.89, 2.63), (302.08, 3.00),
    (302.88, 3.29), (303.77, 3.61), (304.93, 4.24), (305.77, 4.68),
    (307.00, 5.30), (308.15, 6.37), (309.35, 7.82), (310.15, 8.82),
    (310.85, 9.91), (311.21, 11.87), (311.99, 14.81), (312.51, 18.73),
    (313.96, 26.58), (314.83, 31.48), (316.13, 39.33), (317.28, 47.18),
    (318.34, 55.02), (319.44, 62.86), (319.80, 63.97), (320.49, 70.71),
    (321.46, 78.56), (321.72, 79.34), (322.48, 86.40), (323.02, 91.17),
    (323.41, 94.25), (324.45, 103.13), (324.88, 106.02), (325.08, 109.05),
    (325.85, 113.86), (326.69, 121.71), (327.40, 129.55), (328.06, 137.49),
    (328.58, 144.54), (329.48, 153.09), (330.17, 162.41), (330.88, 168.78),
    (331.65, 180.14), (333.43, 203.87), (334.66, 221.75), (335.93, 239.48),
    (337.11, 257.36), (338.18, 276.06), (339.29, 292.96), (340.30, 310.81),
    (341.45, 334.42), (342.74, 358.24), (343.75, 376.01),
]
_HYD_T = np.array([p[0] for p in _HYD_LW_H_V])
_HYD_LNP = np.log(np.array([p[1] for p in _HYD_LW_H_V]))
HYD_T_MIN_C = float(_HYD_T[0] - 273.15)
HYD_T_MAX_C = float(_HYD_T[-1] - 273.15)


def hydrate_pressure_MPa(T_C):
    """sI Xe hydrate (Lw-H-V) equilibrium pressure [MPa] at T_C.

    Log-linear interpolation of the measured data above. Xe hydrate is stable
    ABOVE this pressure. Measured values: 1.17 MPa at 20 C, 3.38 at 30 C,
    6.37 at 35 C, 21.9 at 40 C -- the window opens very steeply above ~35 C.
    Outside 0-70.6 C the value is clamped to the end of the data.
    """
    T_K = float(np.clip(T_C + 273.15, _HYD_T[0], _HYD_T[-1]))
    return float(np.exp(np.interp(T_K, _HYD_T, _HYD_LNP)))


def validate_solubility():
    """Check the Henry constant against the compiled literature value.

    The reference is converted from Sander's Hcp in its native units, so the
    check is independent of the unit conversion baked into XE["NIST_Hcp"] --
    that conversion is exactly what was wrong before. Returns
    (x_model, x_reference, ratio); run after touching NIST_Hcp or H_dT.
    """
    x_ref = HCP_SANDER_XE*101325.0/C_WATER  # mole fraction at 25 C, 1 atm
    x_mod = 0.101325/_henry_MPa(298.15)     # low P: f ~ P, KK correction ~ 1
    return x_mod, x_ref, x_mod/x_ref


def _water(T_K, P_Pa):
    """DI water density [kg/m3] and viscosity [Pa.s] from CoolProp."""
    rho = CP.PropsSI("D", "T", T_K, "P", P_Pa, "Water")
    mu = CP.PropsSI("V", "T", T_K, "P", P_Pa, "Water")
    return rho, mu


def _density_crossover_MPa(T_C, P_grid=None):
    """Pressure [MPa] where the bulk Xe phase density equals water's.

    Below it the Xe phase floats and dissolves downward into the water, which
    is the unstable (convecting) configuration. Above it the Xe phase sinks and
    the arrangement is gravitationally stable. Returns nan if no crossover
    exists within 0.5-60 MPa. At 20 C this is 6.21 MPa; at 40 C, 8.35 MPa.
    """
    T_K = T_C + 273.15

    def diff(P_MPa):
        return (CP.PropsSI("D", "T", T_K, "P", P_MPa*1e6, "Xenon")
                - CP.PropsSI("D", "T", T_K, "P", P_MPa*1e6, "Water"))

    lo, hi = 0.5, 60.0
    try:
        if diff(lo)*diff(hi) > 0:
            return float("nan")
        for _ in range(60):                     # bisection, no scipy needed
            mid = 0.5*(lo + hi)
            if diff(lo)*diff(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return 0.5*(lo + hi)
    except Exception:
        return float("nan")


def _clean(a):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(a, float)]


def calculate(P_min_MPa=1.0, P_max_MPa=20.0, T_C=40.0,
              H_mm=20.0, phi=0.35, k_D=45.0,
              E_keV=35.0, n=101):
    """Sweep pressure at constant T; return Ra and X-ray contrast series.

    H_mm     core height [mm]
    phi      porosity [-]
    k_D      permeability [Darcy]
    E_keV    photon energy [keV]
    """
    T_K = T_C + 273.15
    H = H_mm * 1e-3
    k = k_D * DARCY_M2

    mr_xe = xray.mu_rho_xenon(E_keV)
    mr_w = xray.mu_rho_water(E_keV)

    P = np.linspace(P_min_MPa, P_max_MPa, int(n))
    z = {kk: np.full(len(P), np.nan) for kk in (
        "x_sat", "C_xe", "rho_sol", "rho_w", "delta_rho", "D", "mu_w",
        "Ra", "u_darcy", "l_star", "t_star", "t_conv", "t_diff",
        "att_ratio", "d_mu", "mu_sol", "mu_water",
        "len_water_cm", "len_sol_cm", "len_ratio",
        "fugacity_MPa", "fug_coef", "rho_xe_bulk")}
    eos_tally = {}

    # Stokes-Einstein reference viscosity at 25 C
    mu25 = CP.PropsSI("V", "T", 298.15, "P", 1e5, "Water")

    for i, Pi in enumerate(P):
        P_Pa = Pi * 1e6
        rho_w, mu_w = _water(T_K, P_Pa)
        z["rho_w"][i] = rho_w
        z["mu_w"][i] = mu_w

        # --- solubility (Henry + Krichevsky-Kasarnovsky) ---
        # Fugacity from the Lemmon-Span Helmholtz EoS; Peng-Robinson only if
        # that state cannot be evaluated.
        f_Pa, phi_xe = _fugacity_heos(T_K, P_Pa)
        if np.isfinite(f_Pa):
            f_MPa = f_Pa*1e-6
            eos_used = "HEOS"
        else:
            _, f_atm, _ = _pr_eos(T_K, Pi/0.101325)
            f_MPa = f_atm*0.101325 if np.isfinite(f_atm) else Pi
            eos_used = "PR"
        z["fugacity_MPa"][i] = f_MPa
        z["fug_coef"][i] = phi_xe if np.isfinite(phi_xe) else np.nan
        eos_tally[eos_used] = eos_tally.get(eos_used, 0) + 1
        Pv = CP.PropsSI("P", "T", T_K, "Q", 0, "Water")*1e-6
        KK = _henry_MPa(T_K)*np.exp(XE["Vmp"]*(Pi-Pv)/R/T_K)
        x = f_MPa/KK
        z["x_sat"][i] = x

        # --- density of the saturated solution (molar volume mixing) ---
        V_w = Mw_H2O/rho_w                      # m3/mol
        V_xe = XE["Vmp"]*1e-6                   # m3/mol
        M_mix = x*XE["Mw"] + (1-x)*Mw_H2O
        V_mix = x*V_xe + (1-x)*V_w
        rho_sol = M_mix/V_mix
        z["rho_sol"][i] = rho_sol
        z["delta_rho"][i] = rho_sol - rho_w

        w_xe = x*XE["Mw"]/M_mix                 # mass fraction Xe
        z["C_xe"][i] = w_xe*rho_sol             # kg/m3

        # --- diffusivity (Stokes-Einstein T/eta correction) ---
        D = XE["D_25"]*(T_K/298.15)*(mu25/mu_w)
        z["D"][i] = D

        # --- Rayleigh-Darcy number and scalings ---
        drho = abs(rho_sol - rho_w)
        u_D = k*drho*G/mu_w                     # Darcy buoyancy velocity [m/s]
        z["u_darcy"][i] = u_D
        z["Ra"][i] = drho*G*k*H/(mu_w*phi*D)
        if u_D > 0:
            z["l_star"][i] = phi*D/u_D
            z["t_star"][i] = phi**2*D/u_D**2
            z["t_conv"][i] = H*phi/u_D
        z["t_diff"][i] = H**2/D

        # --- bulk Xe phase density, and finger/pore scale diagnostics ---
        try:
            z["rho_xe_bulk"][i] = CP.PropsSI("D", "T", T_K, "P", P_Pa, "Xenon")
        except Exception:
            pass

        # --- X-ray attenuation [1/cm] ---
        mu_water = (rho_w*1e-3)*mr_w
        mu_sol = (rho_sol*1e-3)*(w_xe*mr_xe + (1-w_xe)*mr_w)
        z["mu_water"][i] = mu_water
        z["mu_sol"][i] = mu_sol
        z["att_ratio"][i] = mu_sol/mu_water
        z["d_mu"][i] = mu_sol - mu_water

        # 1/e attenuation lengths [cm]. len_ratio is path-length independent:
        # lambda_sat/lambda_w = mu_w/mu_sat = 1/att_ratio, i.e. the reciprocal
        # of the coefficient ratio -- a material property, not a geometry one.
        z["len_water_cm"][i] = 1.0/mu_water
        z["len_sol_cm"][i] = 1.0/mu_sol
        z["len_ratio"][i] = mu_water/mu_sol

    out = {kk: _clean(v) for kk, v in z.items()}
    out["pressure_MPa"] = P.tolist()
    out["inputs"] = {"T_C": T_C, "H_mm": H_mm, "phi": phi, "k_D": k_D,
                     "E_keV": E_keV}
    out["constants"] = {
        "Ra_crit": RA_CRIT,
        "mu_rho_xe": mr_xe, "mu_rho_water": mr_w,
        "eos": max(eos_tally, key=eos_tally.get) if eos_tally else "none",
        "eos_tally": eos_tally,
        # Fraction of the swept range where the BULK Xe phase is denser than
        # water. There the Xe phase sinks below the water, so Xe dissolves from
        # BELOW and the dense Xe-rich water forms at the bottom -- a
        # gravitationally STABLE arrangement in which the Rayleigh-Darcy
        # instability does not exist, whatever Ra says.
        "inverted_frac": float(np.mean(
            np.nan_to_num(z["rho_xe_bulk"], nan=0.0) > z["rho_w"])),
        "P_density_cross_MPa": _density_crossover_MPa(T_C, P),
        "P_hydrate_MPa": hydrate_pressure_MPa(T_C),
        "hydrate_frac": float(np.mean(P > hydrate_pressure_MPa(T_C))),
        "Vmp": XE["Vmp"],
        "xe_Tc_C": CP.PropsSI("Tcrit", "Xenon") - 273.15,
        "xe_Pc_MPa": CP.PropsSI("pcrit", "Xenon")*1e-6,
        "xe_k_edge_keV": xray.XE_K_EDGE_KEV,
        "above_edge": bool(E_keV >= xray.XE_K_EDGE_KEV),
        "edge_jump": xray.edge_jump(),
    }
    return out


def saturation_xe_volume(P_tank_MPa, T_tank_C, P_pore_MPa, T_pore_C, V_water_L):
    """Volume of Xe gas at tank conditions to saturate water at pore conditions.

    Calculate how much Xe gas (at tank pressure/temperature) is needed to
    achieve 100% saturation of a given volume of water at different pressure
    and temperature conditions.

    Parameters:
        P_tank_MPa, T_tank_C: tank pressure and temperature (Xe gas)
        P_pore_MPa, T_pore_C: target pore pressure and temperature (saturated water)
        V_water_L: volume of water to saturate [liters]

    Returns: dict with keys:
        V_xe_tank_L: volume of Xe gas at tank conditions [L]
        x_sat: mole fraction Xe at saturation (pore conditions)
        n_xe_dissolved: moles of Xe at saturation
        n_water: moles of water
        rho_xe_tank: Xe density at tank conditions [kg/m³]
        P_hydrate_tank: hydrate boundary at tank temperature [MPa]
        P_hydrate_pore: hydrate boundary at pore temperature [MPa]
        warning: string or None
    """
    T_pore_K = T_pore_C + 273.15
    T_tank_K = T_tank_C + 273.15

    # Moles of water
    n_water = (V_water_L * 1e-3 * 1000.0) / Mw_H2O  # L -> m³ -> kg -> mol

    # Xe saturation at pore conditions (Henry + KK correction)
    P_pore_Pa = P_pore_MPa * 1e6
    f_pore_Pa, _ = _fugacity_heos(T_pore_K, P_pore_Pa)
    if not np.isfinite(f_pore_Pa):
        _, f_atm, _ = _pr_eos(T_pore_K, P_pore_MPa / 0.101325)
        f_pore_Pa = f_atm * 0.101325 * 1e6 if np.isfinite(f_atm) else P_pore_Pa
    f_pore_MPa = f_pore_Pa * 1e-6

    Pv_pore = CP.PropsSI("P", "T", T_pore_K, "Q", 0, "Water") * 1e-6
    KK_pore = _henry_MPa(T_pore_K) * np.exp(XE["Vmp"] * (P_pore_MPa - Pv_pore) / R / T_pore_K)
    x_sat = f_pore_MPa / KK_pore

    # Moles of Xe dissolved at saturation
    n_xe = n_water * x_sat / (1.0 - x_sat)

    # Xe density at tank conditions
    P_tank_Pa = P_tank_MPa * 1e6
    try:
        rho_xe_tank = CP.PropsSI("D", "T", T_tank_K, "P", P_tank_Pa, "Xenon")
    except Exception:
        # Fallback: Peng-Robinson
        vm_tank, _, _ = _pr_eos(T_tank_K, P_tank_MPa / 0.101325)
        if np.isfinite(vm_tank) and vm_tank > 0:
            rho_xe_tank = XE["Mw"] / vm_tank * 1e6
        else:
            # Last resort: ideal gas
            rho_xe_tank = XE["Mw"] * P_tank_MPa * 1e6 / (R * T_tank_K)

    # Volume of Xe at tank conditions
    V_xe_tank_m3 = n_xe * XE["Mw"] / rho_xe_tank
    V_xe_tank_L = V_xe_tank_m3 * 1000.0

    # Hydrate stability check
    P_hyd_tank = hydrate_pressure_MPa(T_tank_C)
    P_hyd_pore = hydrate_pressure_MPa(T_pore_C)
    warning = None
    if P_tank_MPa > P_hyd_tank:
        warning = f"Tank pressure {P_tank_MPa:.2f} MPa exceeds hydrate boundary {P_hyd_tank:.2f} MPa at {T_tank_C}°C — Xe clathrate may form"
    if P_pore_MPa > P_hyd_pore:
        warning = f"Pore pressure {P_pore_MPa:.2f} MPa exceeds hydrate boundary {P_hyd_pore:.2f} MPa at {T_pore_C}°C — hydrate-free model invalid"

    return {
        "V_xe_tank_L": float(V_xe_tank_L),
        "x_sat": float(x_sat),
        "n_xe_dissolved": float(n_xe),
        "n_water": float(n_water),
        "rho_xe_tank": float(rho_xe_tank),
        "P_hydrate_tank_MPa": float(P_hyd_tank),
        "P_hydrate_pore_MPa": float(P_hyd_pore),
        "warning": warning,
    }
