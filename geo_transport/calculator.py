import numpy as np
import CoolProp.CoolProp as CP

R = 8.314
R_Latm = 0.08205616
Mw_H2O = 0.018015

SOLUTES = {
    "CO2":  {"coolprop":"CO2",      "Tc":304.2,  "Pc_atm":72.9,   "omega":0.228,   "Vmp":35.1,  "Mw":0.044,    "NIST_Hcp":0.035,     "H_dT":2400, "D_25":2.2e-9,   "eos":"PR",     "label":"CO2", "color":"#e74c3c"},
    "H2":   {"coolprop":"Hydrogen", "Vmp":26.7,  "Mw":0.002,       "NIST_Hcp":7.8e-4,"H_dT":530,  "D_25":5.11e-9,  "eos":"virial",  "label":"H2",  "color":"#3498db"},
    "He":   {"coolprop":"Helium",   "Tc":5.2,    "Pc_atm":2.244,  "omega":-0.390,  "Vmp":18.0,  "Mw":0.004,    "NIST_Hcp":0.00037,   "H_dT":360,  "D_25":6.76e-9,  "eos":"PR",     "label":"He",  "color":"#27ae60"},
    "Kr":   {"coolprop":"Krypton",  "Tc":209.4,  "Pc_atm":54.3,   "omega":0.0,     "Vmp":34.0,  "Mw":0.08380,  "NIST_Hcp":2.5e-3,    "H_dT":1900, "D_25":1.84e-9,  "eos":"PR",     "label":"Kr",  "color":"#8e44ad"},
    "N2":   {"coolprop":"Nitrogen", "Tc":126.2,  "Pc_atm":33.53,  "omega":0.039,   "Vmp":34.05, "Mw":0.02802,  "NIST_Hcp":0.00060,   "H_dT":1300, "D_25":2.61e-9,  "eos":"PR",     "label":"N2",  "color":"#e67e22"},
    "Xe":   {"coolprop":"Xenon",    "Tc":289.73, "Pc_atm":57.64,  "omega":0.0,     "Vmp":46.5,  "Mw":0.13129,  "NIST_Hcp":4.3e-3,    "H_dT":2300, "D_25":1.47e-9,  "eos":"PR",     "label":"Xe",  "color":"#00b4d8"},
}

def _pr_eos(T_K, P_atm, Tc, Pc_atm, omega):
    if P_atm <= 0: return float("nan"), float("nan"), float("nan")
    k = 0.37464 + 1.54226*omega - 0.26992*omega**2
    alpha = (1 + k*(1-(T_K/Tc)**0.5))**2
    a = 0.45724*R_Latm**2*Tc**2/Pc_atm*alpha
    b = 0.07780*R_Latm*Tc/Pc_atm
    A = a*P_atm/(R_Latm**2*T_K**2)
    B = b*P_atm/(R_Latm*T_K)
    roots = np.roots([1, -(1-B), A-3*B**2-2*B, -(A*B-B**2-B**3)])
    valid = roots[np.isreal(roots)].real; valid = valid[valid > B]
    if valid.size == 0: return float("nan"), float("nan"), float("nan")
    Z = float(np.max(valid))
    vm = Z*R_Latm*T_K/P_atm*1000
    denom = (Z+2.414*B)/(Z-0.414*B)
    if denom <= 0 or Z <= B: return vm, float("nan"), float("nan")
    phi = np.exp(Z-1-np.log(Z-B)-A/(2*2**0.5*B)*np.log(denom))
    return vm, phi*P_atm, phi

def _h2_virial(T_K, P_MPa):
    if P_MPa <= 0: return float("nan"), float("nan"), float("nan")
    b = 15.84
    vm = R*T_K/P_MPa + b
    f  = P_MPa*np.exp(P_MPa*b/R/T_K)
    return vm, f, f/P_MPa

def _water_viscosity(T_K, P_Pa=1e5):
    """Liquid water viscosity [Pa.s]. Pass system pressure to stay in liquid phase above 100 C."""
    try:
        return CP.PropsSI("V", "T", T_K, "P", P_Pa, "Water")
    except Exception:
        t = T_K - 273.15
        log_ratio = ((20-t)/(t+96))*(1.2378-1.303e-3*(20-t)+3.06e-6*(20-t)**2+2.55e-8*(20-t)**3)
        return 1.002e-3 * 10**log_ratio

def _water_vapor_pressure(T_K):
    try:    return CP.PropsSI("P","T",T_K,"Q",0,"Water")*1e-6
    except: return 10**(8.07131-1730.63/(233.426+T_K-273.15))*133.322e-6

def _henry_MPa(sol, T_K):
    H_ref = 0.101325/(sol["NIST_Hcp"]*Mw_H2O)
    return H_ref/np.exp(sol["H_dT"]*(1.0/T_K-1.0/298.15))

def _clean(arr):
    return [float(v) if np.isfinite(v) else None for v in arr]

def calculate(mode, solute_ids, rho_water_kg_m3=1050.0,
              T_C=20.0, P_max_MPa=20.0,
              T_min_C=0.0, T_max_C=100.0, P_nominal_MPa=10.0,
              T_grad_C_km=25.0):
    """
    mode = 'pressure'    : sweep P at constant T_C
    mode = 'temperature' : sweep T from T_min_C to T_max_C at constant P_nominal_MPa
    mode = 'depth'       : sweep depth (via P) with T from surface T_C + gradient
    """
    g = 9.81
    rho_w = rho_water_kg_m3 * 1e-6  # kg/cm3

    if mode == "pressure":
        P_MPa = np.arange(0.1, P_max_MPa + 0.05, 0.1)
        T_K   = np.full(len(P_MPa), 273.15 + T_C)
        depth = P_MPa * 1e6 / (rho_water_kg_m3 * g)
        x_values = P_MPa.tolist()
        x_label  = "Pressure [MPa]"

    elif mode == "temperature":
        T_C_arr = np.arange(T_min_C, T_max_C + 0.5, 1.0)
        T_K     = T_C_arr + 273.15
        P_MPa   = np.full(len(T_K), P_nominal_MPa)
        depth   = P_MPa * 1e6 / (rho_water_kg_m3 * g)
        x_values = T_C_arr.tolist()
        x_label  = "Temperature [°C]"

    else:  # depth
        P_MPa = np.arange(0.1, P_max_MPa + 0.05, 0.1)
        depth = P_MPa * 1e6 / (rho_water_kg_m3 * g)
        T_K   = depth * T_grad_C_km / 1e3 + (273.15 + T_C)
        x_values = depth.tolist()
        x_label  = "Depth [m]"

    visc25 = _water_viscosity(298.15, 1e5)
    out = {
        "x_values": x_values, "x_label": x_label,
        "pressure_MPa": P_MPa.tolist(),
        "temperature_C": (T_K - 273.15).tolist(),
        "depth_m": depth.tolist(),
        "solutes": {},
    }

    for sid in solute_ids:
        sol = SOLUTES[sid]; n = len(P_MPa)
        vm=np.full(n,np.nan); fug=np.full(n,np.nan); phi=np.full(n,np.nan)
        rho=np.full(n,np.nan); visc=np.full(n,np.nan); H=np.full(n,np.nan)
        KK=np.full(n,np.nan); xs=np.full(n,np.nan); D=np.full(n,np.nan); Ra=np.full(n,np.nan)

        dm = sol["Vmp"]*rho_w - sol["Mw"]

        for i in range(n):
            Ti, Pi, Pai = T_K[i], P_MPa[i], P_MPa[i]/0.101325
            if sol["eos"]=="PR":
                vm[i],fa,phi[i] = _pr_eos(Ti,Pai,sol["Tc"],sol["Pc_atm"],sol["omega"])
                if np.isfinite(fa): fug[i]=fa*0.101325
            else:
                vm[i],fug[i],phi[i] = _h2_virial(Ti,Pi)
            if np.isfinite(vm[i]) and vm[i]>0: rho[i]=sol["Mw"]/vm[i]
            try: visc[i]=CP.PropsSI("V","T",Ti,"P",Pi*1e6,sol["coolprop"])
            except: pass
            H[i]  = _henry_MPa(sol,Ti)
            Pv    = _water_vapor_pressure(Ti)
            KK[i] = H[i]*np.exp(sol["Vmp"]*(Pi-Pv)/R/Ti)
            f     = fug[i] if np.isfinite(fug[i]) else Pi
            xs[i] = f/KK[i]
            visc_w = _water_viscosity(Ti, Pi*1e6)
            D[i]  = sol["D_25"]*(Ti/298.15)*(visc25/visc_w)
            if np.isfinite(xs[i]) and np.isfinite(D[i]) and D[i] > 0:
                delta_rho = abs(xs[i] * dm / Mw_H2O) * rho_water_kg_m3
                Ra[i] = delta_rho * g / (visc_w * D[i])

        out["solutes"][sid] = {
            "label":sol["label"], "color":sol["color"],
            "vm_cm3_mol":_clean(vm), "fugacity_MPa":_clean(fug),
            "fugacity_coef":_clean(phi), "rho_gas_kg_m3":_clean(rho*1e6),
            "visc_gas_uPas":_clean(visc*1e6), "henry_MPa":_clean(H),
            "partition_coef_MPa":_clean(KK), "solubility_mol_frac":_clean(xs),
            "diffusivity_m2s":_clean(D), "rayleigh_specific_per_m":_clean(Ra),
            "delta_m_kg_mol":float(dm),
        }
    return out
