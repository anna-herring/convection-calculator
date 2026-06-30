import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Literal
import calculator

app = FastAPI(title="Geo Transport Calculator")

@app.middleware("http")
async def ngrok_skip(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

class CalcRequest(BaseModel):
    mode: Literal["pressure","temperature","depth"] = "pressure"
    solutes: List[str] = ["CO2","H2"]
    rho_water_kg_m3: float = Field(1050.0, ge=990.0, le=1300.0)
    # pressure mode
    T_C: float = 20.0
    P_max_MPa: float = Field(20.0, ge=0.5, le=100.0)
    # temperature mode
    T_min_C: float = 0.0
    T_max_C: float = 100.0
    P_nominal_MPa: float = Field(10.0, ge=0.5, le=100.0)
    # depth mode
    T_grad_C_km: float = Field(25.0, ge=0.0, le=80.0)

@app.get("/api/solutes")
def list_solutes():
    return {k:{"label":v["label"],"color":v["color"]} for k,v in calculator.SOLUTES.items()}

@app.post("/api/calculate")
def calculate(req: CalcRequest):
    unknown = [s for s in req.solutes if s not in calculator.SOLUTES]
    if unknown: raise HTTPException(400, f"Unknown solutes: {unknown}")
    return calculator.calculate(
        mode=req.mode, solute_ids=req.solutes,
        rho_water_kg_m3=req.rho_water_kg_m3,
        T_C=req.T_C, P_max_MPa=req.P_max_MPa,
        T_min_C=req.T_min_C, T_max_C=req.T_max_C,
        P_nominal_MPa=req.P_nominal_MPa,
        T_grad_C_km=req.T_grad_C_km,
    )

_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=_static, html=True), name="static")
