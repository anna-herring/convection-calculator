import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import calculator

app = FastAPI(title="Radionuclide Transport Regime Calculator")


@app.middleware("http")
async def ngrok_skip(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


# ── Shared advection inputs ───────────────────────────────────────────────────

class AdvectionBase(BaseModel):
    T_C:            float = Field(25.0,  ge=0.0,    le=350.0)
    P_MPa:          float = Field(10.0,  ge=0.1,    le=100.0)
    v_m_s:          float = Field(1e-8,  ge=1e-15,  le=1e-2)
    rn_ids:         List[str] = ["Cs", "Sr", "I", "Tc", "U"]
    rock_type:      str   = "granite"
    porosity:       float = Field(0.05,  ge=0.001,  le=0.6)
    tortuosity:     float = Field(2.0,   ge=1.0,    le=20.0)
    dispersivity_m: float = Field(0.01,  ge=1e-6,   le=1000.0)
    char_len_m:     float = Field(1.0,   ge=1e-4,   le=1e6)
    Eh_mV:          float = Field(200.0, ge=-500.0, le=800.0)
    pH:             float = Field(7.0,   ge=0.0,    le=14.0)
    salinity_g_L:   float = Field(10.0,  ge=0.0,    le=350.0)
    kd_scenario:    str   = "mean"


class MultiphaseRequest(AdvectionBase):
    gas_fluid_id:   str   = "CO2"
    S_w:            float = Field(0.7,   ge=0.05,   le=1.0)
    theta_deg:      float = Field(10.0,  ge=0.0,    lt=90.0)


class ThermalConvectionRequest(BaseModel):
    T_C:          float = Field(25.0,  ge=0.0,   le=350.0)
    P_MPa:        float = Field(10.0,  ge=0.1,   le=100.0)
    dT_dz_C_m:   float = Field(0.03,  ge=0.0,   le=200.0)
    H_m:          float = Field(100.0, ge=0.1,   le=10000.0)
    aperture_um:  float = Field(100.0, ge=1.0,   le=10000.0)
    salinity_g_L: float = Field(10.0,  ge=0.0,   le=350.0)
    rock_type:    str   = "granite"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/metadata")
def get_metadata():
    return {
        "radionuclides": {
            k: {"label": v["label"], "halflife": v["halflife"]}
            for k, v in calculator.SPECIATION_RULES.items()
        },
        "rock_types": list(calculator.GRAIN_DENSITY.keys()),
        "gas_fluids": {
            k: {"label": v["label"], "color": v["color"], "group": v["group"]}
            for k, v in calculator.GAS_FLUIDS.items()
        },
    }


@app.post("/api/single-phase")
def single_phase(req: AdvectionBase):
    unknown = [r for r in req.rn_ids if r not in calculator.SPECIATION_RULES]
    if unknown:
        raise HTTPException(400, f"Unknown radionuclides: {unknown}")
    if req.rock_type not in calculator.GRAIN_DENSITY:
        raise HTTPException(400, f"Unknown rock type: {req.rock_type}")
    return calculator.calculate_single_phase(
        req.T_C, req.P_MPa, req.v_m_s, req.rn_ids,
        req.rock_type, req.porosity, req.tortuosity,
        req.dispersivity_m, req.char_len_m,
        req.Eh_mV, req.pH, req.salinity_g_L,
        req.kd_scenario,
    )


@app.post("/api/multiphase")
def multiphase(req: MultiphaseRequest):
    unknown = [r for r in req.rn_ids if r not in calculator.SPECIATION_RULES]
    if unknown:
        raise HTTPException(400, f"Unknown radionuclides: {unknown}")
    if req.rock_type not in calculator.GRAIN_DENSITY:
        raise HTTPException(400, f"Unknown rock type: {req.rock_type}")
    if req.gas_fluid_id not in calculator.GAS_FLUIDS:
        raise HTTPException(400, f"Unknown gas fluid: {req.gas_fluid_id}")
    return calculator.calculate_multiphase(
        req.T_C, req.P_MPa, req.v_m_s, req.rn_ids,
        req.rock_type, req.porosity, req.tortuosity,
        req.dispersivity_m, req.char_len_m,
        req.Eh_mV, req.pH, req.salinity_g_L,
        req.gas_fluid_id, req.S_w,
        None, 25.4, req.theta_deg,
        req.kd_scenario,
    )


@app.post("/api/thermal-convection")
def thermal_convection(req: ThermalConvectionRequest):
    if req.rock_type not in calculator.GRAIN_DENSITY:
        raise HTTPException(400, f"Unknown rock type: {req.rock_type}")
    return calculator.calculate_thermal_convection(
        req.T_C, req.P_MPa, req.dT_dz_C_m, req.H_m,
        req.aperture_um, req.salinity_g_L, req.rock_type,
    )


_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=_static, html=True), name="static")
