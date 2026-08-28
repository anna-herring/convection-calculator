# APS Xe Saturation Calculator — Standalone Version

No server required. Pure JavaScript. Works offline.

## Quick Start

### Option 1: Open locally in browser
```bash
# On Windows, macOS, or Linux:
open aps_xe_dashboard/static/standalone.html
# or just double-click the file
```

### Option 2: Serve via HTTP (to avoid CORS issues)
```bash
# Python 3
python -m http.server 8000
# Then visit: http://localhost:8000/aps_xe_dashboard/static/standalone.html

# or with Python 2
python -m SimpleHTTPServer 8000
```

### Option 3: Deploy to GitHub Pages
The file is pure HTML/JavaScript with no build step. Just push to your repo and enable GitHub Pages.

### Option 4: Embed in your site
Link directly to the raw GitHub file:
```html
<iframe src="https://raw.githubusercontent.com/anna-herring/convection-calculator/master/aps_xe_dashboard/static/standalone.html"></iframe>
```

## What's Inside

**Physics models (all in JavaScript):**
- Peng–Robinson cubic equation of state (no CoolProp needed)
- Henry's law with Krichevsky–Kasarnovsky pressure correction
- Full Rasoolzadeh et al. (2020) clathrate hydrate boundary (87 data points)
- IUPAC Henry's constant temperature dependence

**Features:**
- Tank pressure input in **PSI** (130 PSI default)
- Target pore pressure in **MPa** (10 MPa default)
- Water volume in **ml** (7 ml default)
- Xe volume results in **ml**
- Hydrate stability warnings
- Works offline after page load

## Comparison: Standalone vs FastAPI Dashboard

| Feature | Standalone | FastAPI |
|---------|-----------|---------|
| Server required | ❌ No | ✅ Yes |
| Backend | JavaScript | Python |
| EoS | Peng–Robinson cubic | Helmholtz (CoolProp) |
| Offline capable | ✅ Yes | ❌ No |
| Precision | Good (cubic EoS) | Excellent (reference EoS) |
| Deployment | GitHub Pages, static hosting | Render, Heroku, self-hosted |

**Note:** Peng–Robinson EoS has ~1.5–4% errors vs reference at 40 °C but is significantly more accurate than ideal gas and requires no external libraries.

## Physics Implementation Notes

### Peng–Robinson EoS
- Cubic formulation with acentric factor correction
- Cardano's method for root-finding
- Gas-phase root selection (highest Z)
- Fugacity via residual enthalpy formulation

### Henry's Law
- Reference constant from IUPAC Sander compilation: H^cp = 4.3e-5 mol/(kg·bar)
- Temperature dependence: H(T) = H_ref / exp(ΔH/R × (1/T − 1/T_ref))
- ΔH/R = 2300 K (validated against multiple literature sources)
- Krichevsky–Kasarnovsky pressure correction: K_K = H × exp(V̄_m(P − P^sat_w) / RT)

### Hydrate Boundary
- 87 experimental points from Rasoolzadeh et al. (2020)
- Covers 273.15–318.65 K and 0.15–1746 MPa
- Log-linear interpolation (log P vs 1/T)
- Warns if tank or pore conditions exceed boundary

## Limitations

- No diffusivity or Rayleigh–Darcy calculations (use FastAPI version for those)
- Peng–Robinson is less accurate than reference EoS near the critical point
- No pressure gradients or multi-phase flow
- Single-point calculations only (no sweep plots)

## When to Use

✅ **Standalone is best for:**
- Quick saturation checks without server setup
- Offline beamtime calculations
- Static deployment (GitHub Pages)
- Embedding in papers or documentation
- Low-latency local calculations

✅ **FastAPI is best for:**
- Rayleigh–Darcy and transport analysis
- High-precision thermodynamics
- Sweep plots and comparative analysis
- Integration with other web services

## Building a Custom Version

The JavaScript physics models are self-contained. To adapt them:
1. Extract the EoS and Henry's law functions from the `<script>` block
2. Port to your language (Python, Julia, C++) or framework (React, Vue, etc.)
3. Reference the same literature sources: Lemmon & Span 2006, Rasoolzadeh et al. 2020, Sander 2015

---

**Deployed:** https://github.com/anna-herring/convection-calculator  
**Last updated:** 2026-08-29
