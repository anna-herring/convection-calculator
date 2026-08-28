import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

import aps_calc

app = FastAPI(title="APS Xe Beamtime Dashboard")


@app.middleware("http")
async def ngrok_skip(request, call_next):
    resp = await call_next(request)
    resp.headers["ngrok-skip-browser-warning"] = "true"
    return resp


class Req(BaseModel):
    P_min_MPa: float = Field(1.0, ge=0.5, le=60.0)
    P_max_MPa: float = Field(20.0, ge=0.5, le=60.0)
    T_C: float = Field(40.0, ge=1.0, le=90.0)
    H_mm: float = Field(20.0, gt=0.0, le=500.0)
    phi: float = Field(0.35, gt=0.0, lt=1.0)
    k_D: float = Field(45.0, gt=0.0, le=5000.0)
    E_keV: float = Field(35.0, ge=10.0, le=200.0)


@app.get("/api/presets")
def presets():
    return aps_calc.PRESETS


@app.post("/api/calculate")
def calculate(req: Req):
    lo, hi = sorted([req.P_min_MPa, req.P_max_MPa])
    if hi - lo < 0.1:
        hi = lo + 0.1
    return aps_calc.calculate(
        P_min_MPa=lo, P_max_MPa=hi, T_C=req.T_C,
        H_mm=req.H_mm, phi=req.phi, k_D=req.k_D,
        E_keV=req.E_keV,
    )


class SatReq(BaseModel):
    P_tank_MPa: float = Field(10.0, ge=0.5, le=100.0)
    T_tank_C: float = Field(20.0, ge=1.0, le=90.0)
    P_pore_MPa: float = Field(3.0, ge=0.5, le=60.0)
    T_pore_C: float = Field(40.0, ge=1.0, le=90.0)
    V_water_L: float = Field(1.0, gt=0.0, le=100.0)


@app.post("/api/saturation")
def saturation(req: SatReq):
    return aps_calc.saturation_xe_volume(
        P_tank_MPa=req.P_tank_MPa, T_tank_C=req.T_tank_C,
        P_pore_MPa=req.P_pore_MPa, T_pore_C=req.T_pore_C,
        V_water_L=req.V_water_L,
    )


_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=_static, html=True), name="static")
