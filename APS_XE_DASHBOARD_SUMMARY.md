# APS Xe Beamtime Dashboard — Summary for Future Sessions

**For direct reference in new Claude Code sessions, use this file to orient yourself to the project structure, capabilities, and recent modifications.**

---

## Quick Start

### Standalone (No Server Required)

**`aps_xe_dashboard/static/standalone.html`** — Pure JavaScript, no backend needed
- Open directly in browser: `file:///path/to/standalone.html`
- Deploy to GitHub Pages
- Embed in any static site
- Works offline

### Full Dashboard (FastAPI + Frontend)

The APS Xe Dashboard is a FastAPI web tool deployed at `aps_xe_dashboard/` that provides three interactive pages:

1. **Dashboard** (`/`) — Rayleigh–Darcy convection analysis and X-ray attenuation
2. **Saturation** (`/saturation.html`) — Xe gas volume calculator for saturation
3. **Equations** (`/theory.html`) — Equations, derivations, and references

**File structure:**
- `aps_xe_dashboard/main.py` — FastAPI endpoints
- `aps_xe_dashboard/aps_calc.py` — Physics calculations (EoS, solubility, transport)
- `aps_xe_dashboard/xray.py` — X-ray attenuation tables
- `aps_xe_dashboard/static/` — HTML/CSS/JavaScript frontends

---

## Core Physics & Models

### Xenon equation of state
- **Primary:** Helmholtz-energy residual EoS (Lemmon & Span 2006) via CoolProp
- **Fallback:** Peng–Robinson (cubic) only if reference fails
- **Why:** Cubic EoS have 4–5% errors near Xe's critical point (16.58 °C, 5.84 MPa); the beamtime runs at 40 °C where PR still has ~1.5–4% error, but reference is superior
- **Sign convention in aps_calc.py:** `Z = 1 + δ(∂α^r/∂δ)_τ`, `ln φ = α^r + (Z−1) − ln Z`

### Gas solubility & Krichevsky–Kasarnovsky correction
- **Henry's law baseline:** H^cp (NIST tabulated, units mol/(kg·bar); convert ×100 to SI mol/(m³·Pa))
  - For Xe: H^cp = 4.3e-5 mol/(kg·bar) → validate against Bunsen coefficient α = 0.108 at 25 °C/1 atm
- **Temperature dependence:** `H(T) = H_ref / exp(ΔsolnH/R × (1/T − 1/T_ref))`, ΔsolnH/R = 2300 K
- **Pressure correction:** `K_K = H × exp(V̄_m(P − P^sat_w) / RT)`
  - Partial molar volume: V̄_m = 46.5 cm³/mol (genuine dissolved Xe, NOT 125 from high-pressure artifact)
  - Water vapour pressure: IAPWS-95 correlation or Antoine-like fallback

### Hydrate stability boundary
- **Data source:** Rasoolzadeh et al. (2020), 87 L_w–H–V experimental points
- **Range:** 273.15–343.75 K, 0.15–376 MPa
- **Interpolation:** Log-linear fit (log P vs 1/T); monotone cubic for robustness
- **Operating constraint:** Beamtime target 40 °C, 1–5 MPa (hydrate-free at 40 °C; boundary ≈ 10 MPa there)

### Rayleigh–Darcy convection number
```
Ra = (Δρ × g / (ν × D)) × [characteristic length]
```
- **Δρ contribution:** x_sat × V̄_m × ρ_water (sign: positive, water gets denser)
- **Viscosity ν:** IAPWS-95 Huber et al. (2009), with T/P correction via CoolProp
- **Diffusivity D:** D_25 = 1.47e-9 m²/s (Jähne 1987), scaled by (T/T_ref) × (ν_ref/ν) Stokes–Einstein
- **Rayleigh criterion:** Ra_c ≈ 4π² ≈ 39.5 for an infinite steady layer (but boundary-layer onset is lower in transient; use this as a heuristic)

### X-ray attenuation
- **Xe mass attenuation coefficient μ/ρ:** NIST SRD 126, tabulated at 13 energies (10–200 keV)
  - K-edge at 34.5614 keV (jump 5.41×, from 6.129 to 33.16 cm²/g)
  - Interpolated via log-log between tabulated points (no invented intermediate values)
- **Water μ/ρ:** NIST table, ~0.2–0.02 cm²/g across the range
- **Transmission contrast:** 1/e length ratio = ln(I₀/I_sat) / ln(I₀/I_water)
  - Also reported as normalized inverse (1/e length relative to water)

### Permeability estimation (Kozeny–Carman)
```
k = (φ³ / (1−φ)²) × (d_pore / 150)²
```
- Applied to pore-size ranges; used for preset permeability defaults (ROBU0–3, Bentheimer)

---

## Dashboard Features (Main Calculations)

### Rayleigh–Darcy vs Pressure
- X-axis: pressure 1–20 MPa at 40 °C (or user-specified T)
- Y-axis: Ra per unit height (Ra / H in m⁻¹)
- Dashed line: Ra_c = 39.5 (steady onset, transient onset lower)
- Preset parameter sets: ROBU00 (250–500 µm), ROBU0 (160–250 µm), ROBU1 (100–160 µm), ROBU2 (40–100 µm), ROBU3 (16–40 µm), Bentheimer

### Attenuation ratio (Xe-saturated vs water)
- X-axis: pressure 1–20 MPa
- Y-axis: μ_sat / μ_water (dimensionless contrast, typically 1.05–1.25)
- Clathrate hydrate boundary flagged red above it, green below

### Transit vs diffusion (Péclet number)
- Pe = (v_D × ℓ) / D, where v_D is Darcy velocity and ℓ is diffusion length
- Pe >> 1: convection dominates; Pe << 1: diffusion dominates

### 1/e length ratio (normalized attenuation)
- Shown both absolute and normalized to water
- Integrates attenuation over the energy range (monochromatic assumption: peak at 35 keV)

---

## Saturation Volume Calculator

### Purpose
Given: tank pressure/temperature (where Xe sits as a compressed gas) and target pore conditions (T, P, water volume)  
Solve: volume of Xe at tank conditions needed to **saturate** that water at pore conditions

### Inputs
- **Tank:** Pressure (PSI, default 130), Temperature (°C, default 20)
- **Target:** Pressure (MPa, default 10), Temperature (°C, default 40), Water volume (ml, default 7)

### Outputs
- **Xe volume at tank (ml):** The amount you need to transfer
- **Saturation mole fraction:** x_sat at pore conditions (e.g., 2.6 mmol/mol)
- **Moles Xe dissolved:** n_Xe for the given water amount
- **Xe density at tank (kg/m³):** State of the gas at tank conditions
- **Hydrate stability:** Warning if conditions approach the clathrate boundary

### Unit conversions (internal)
- PSI → MPa: ×0.00689476
- ml → L: ×0.001
- Results display in ml (from L ×1000)

---

## Standalone Version (JavaScript, No Backend)

**New: `aps_xe_dashboard/static/standalone.html`** provides the saturation calculator as pure HTML/JavaScript:
- **Physics:** Peng–Robinson EoS, Henry's law with Krichevsky–Kasarnovsky, Rasoolzadeh hydrate boundary
- **No dependencies:** No CoolProp, no FastAPI, no server required
- **Deployment:** Run locally (`file://`), serve via HTTP, deploy to GitHub Pages, embed anywhere
- **Identical interface:** PSI tank pressure (130 PSI default), 10 MPa target, ml water volume, ml Xe results
- **Offline capability:** Works without internet after first load

Use this when:
- Render deployment fails or is unavailable
- Running locally without a Python environment
- Deploying to GitHub Pages or static hosting
- Embedding in documentation or papers
- Offline analysis

Physics implementations:
- Peng–Robinson cubic EoS (Cardano's method, gas-phase root selection)
- Henry constant temperature dependence via IUPAC compilations
- Log-linear hydrate boundary interpolation (87 measured points from Rasoolzadeh et al. 2020)

## Recent Modifications (2026-08-28 & 2026-08-29)

### Saturation calculator interface updates
1. **Tank pressure:** Changed from MPa to **PSI**, default 130 (≈0.896 MPa)
2. **Target pore pressure:** Default changed from 3 MPa to **10 MPa**
3. **Water volume:** Changed from liters to **ml**, default 7 (0.007 L)
4. **Xe volume display:** Now shown in **ml** (converted from liters via ×1000)

### Equations sheet
- Added **Section 12** covering saturation volume calculation: physical basis, solver procedure, hydrate stability
- Reference numbering updated (former §12 → §13)

---

## CoolProp & Henry's Law Gotchas

### ⚠️ Fugacity bug (fixed 2026-08-10)
- **Problem:** `CP.PropsSI("fugacity", ...)` is not a valid CoolProp call; it was silently falling back to ideal gas (f = P)
- **Symptom:** CO₂ solubility ~2.7× too high at 50 °C/10 MPa
- **Solution:** Compute fugacity coefficient φ from residual Helmholtz energy, then f = φ·P
  - See `aps_calc.py::_fugacity_MPa()` (residual-Gibbs method)
  - Verified against literature for H₂, He, Kr, N₂, CO₂, Xe

### ⚠️ Henry's law unit conversion
- **Problem:** NIST H^cp is in mol/(kg·bar), but easy to off-by-10 when converting to SI
- **Validation:** Cross-check against Bunsen coefficient α (dimensionless) at a reference point
  - For Xe at 25 °C / 1 atm: α = 0.108 → H = 8.68e-5 mol/mol/Pa (equivalently 8.68e-5 mole fraction per Pa partial pressure)
- **Current value:** H^cp = 4.3e-5 mol/(kg·bar) (correct) ✓

### Henry constant temperature dependence
- Formula: `H(T) = H_ref / exp(ΔH/R × (1/T − 1/T_ref))`
- For Xe: ΔH/R = 2300 K (validated against Clever 1979, Abraham & Matteoli 1988, Fernández-Prini et al. 2003)

### Partial molar volume
- **Value:** 46.5 cm³/mol (genuine dissolved Xe at infinite dilution, Moore et al. 1982)
- **Why not 125 cm³/mol?** That comes from high-pressure solubility fits and conflates clathrate formation (Kennan & Pollack 1990), which shifts V̄_m to huge values (~131 cm³/mol at the buoyancy sign-reversal threshold). Using 125 places the system dangerously close to a hypersensitivity boundary.

---

## File Paths & Git Status

**Deployed app locations:**
- Dashboard: https://aps-xe-dashboard.onrender.com (or localhost:8002 for local dev)
- GitHub repo: https://github.com/anna-herring/convection-calculator.git (contains only geo_transport, NOT aps_xe_dashboard)

**Local source (OneDrive-backed):**
```
C:/Users/u5259522/OneDrive - University of Tennessee/Desktop/Anna/2026_ClaudeCode/
├── aps_xe_dashboard/
│   ├── main.py
│   ├── aps_calc.py
│   ├── xray.py
│   └── static/
│       ├── index.html (Dashboard)
│       ├── saturation.html (Saturation calculator)
│       └── theory.html (Equations & references)
├── .claude/
│   └── launch.json (contains aps_xe_dashboard config for preview_start)
└── CLAUDE.md (main project documentation)
```

**.claude/launch.json entry:**
```json
{
  "name": "aps_xe_dashboard",
  "runtimeExecutable": "C:\\Users\\u5259522\\AppData\\Local\\anaconda3\\python.exe",
  "runtimeArgs": ["-m", "uvicorn", "main:app", "--app-dir", "aps_xe_dashboard", "--port", "8002"],
  "port": 8002
}
```

---

## Testing & Verification

### Quick validation checks
1. **Henry constant:** Run `aps_calc.validate_solubility()` to cross-check H² against literature Bunsen coefficients for all six solutes
2. **X-ray tables:** NIST values at 35 keV should match XCOM database (no manual interpolation)
3. **Hydrate boundary:** Interpolated log-linear fit should smoothly pass through all 87 Rasoolzadeh et al. points

### Live preview (local)
```bash
cd ~/2026_ClaudeCode
preview_start name:aps_xe_dashboard  # Starts FastAPI on localhost:8002
```
Then navigate to:
- http://localhost:8002 (Dashboard)
- http://localhost:8002/saturation.html (Saturation)
- http://localhost:8002/theory.html (Equations)

---

## Common Modifications

### Changing defaults (presets, units, ranges)
- **Dashboard presets:** Edit `aps_calc.PRESETS` dict in aps_calc.py
- **Saturation inputs:** Edit HTML form `value="..."` in `static/saturation.html`; also update JavaScript conversions if changing units
- **Temperature or pressure ranges:** Modify `min`, `max`, `step` on `<input>` tags; validate against EoS bounds

### Adding a new solute to geo_transport
- Add entry to `geo_transport/calculator.py::SOLUTES` dict with keys: `coolprop`, `Tc`, `Pc_atm`, `omega`, `Vmp`, `Mw`, `NIST_Hcp`, `H_dT`, `D_25`, `eos`, `label`, `color`
- Verify Henry constant via literature Bunsen coefficient at a reference point

### Updating X-ray attenuation
- Fetch new NIST SRD 126 table for the relevant element
- Edit `xray.py` tab-separated values; do NOT interpolate between tabulated points (use only NIST grid points)
- Test that interpolation is smooth across the K-edge (Xe at 34.5614 keV)

---

## References & Documentation

- **Rasoolzadeh et al. (2020):** Xe clathrate hydrate boundary (87 points, 273–343 K, 0.15–376 MPa)
- **Lemmon & Span (2006):** Xe reference EoS (Helmholtz-energy residual formulation)
- **Krichevsky & Kasarnovsky (1935):** Pressure correction to Henry's law
- **NIST SRD 126:** X-ray mass attenuation coefficients
- **Jähne et al. (1987):** Diffusivity of sparingly soluble gases in water (D_25 for Xe = 1.47e-9 m²/s)

See `static/theory.html` for the full reference list with DOIs.

---

## Contact / Future Work

**For beamtime-specific updates:**
- Core physics (EoS, solubility) is stable unless new experimental data warrant recalibration
- New X-ray energies or beamline geometries may require attenuation table updates (NIST-sourced, no guessing)
- User feedback on preset parameters (pore sizes, permeabilities) can be incorporated via ROBU0–3 defaults

**Known limitations:**
- Assumes isothermal core (no axial T gradient)
- Monochromatic X-ray beam assumed at K-edge (bandwidth straddling the edge dilutes attenuation contrast)
- Pore-scale Rayleigh number uses upscaled (continuum) transport properties, not pore-by-pore heterogeneity
- Permeability estimated from pore size alone (Kozeny–Carman); real cores may have tortuosity factors up to 2–3×

---

**Last updated:** 2026-08-28  
**Maintained by:** Anna Herring (annalisaherring@gmail.com)
