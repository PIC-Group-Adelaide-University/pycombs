# pycombs — Interactive Lugiato–Lefever Equation Simulator

---

## 1. Description

pycombs is an interactive Python-based simulator for optical frequency comb generation in Kerr nonlinear microresonators using the Lugiato–Lefever Equation (LLE).

It provides a graphical interface for real-time exploration of comb dynamics, including spectrum, temporal pulse formation, and intracavity power evolution, with parameters accessible in both normalized and physical units.

---

## 2. How to Install and Start

### Requirements

* Python 3.10 or higher
* Python packages:

  * numpy
  * matplotlib
  * scipy

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

---

### File setup

Ensure the following structure:

```
pycombs/
│
├── pycombs.py
├── data/                  (optional)
│   └── dispersion.csv     (optional external Dint file)
└── exported_data/         (created automatically)
```

---

### Dispersion file (optional)

The simulator accepts an external integrated dispersion (Dint) file.

Expected format:

```
Wavelength [um], Frequency [THz], Dint/2pi [GHz]
```

* If provided, update the file path in the code:

  ```
  self.dint_file_path = "data/dispersion.csv"
  ```
* If not provided, a default dispersion profile is used

---

### Plotting / GUI backend

The simulator relies on an interactive matplotlib backend.

Recommended:

* Qt backend (fast and responsive)

If needed, set manually in the script:

```python
import matplotlib
matplotlib.use("Qt5Agg")
```

---

### Run the simulator

From the project directory:

```bash
python pycombs.py
```

This launches the graphical interface.

---

### Basic usage

1. Click **Run**
2. Adjust:

   * Detuning
   * Pump power
3. Observe:

   * Optical spectrum
   * Temporal pulse
   * Intracavity power

---

## 3. Physical Model and Simulation Settings

### Governing equation

The simulator solves the normalized Lugiato–Lefever Equation (LLE) for a driven Kerr nonlinear cavity.

---

### Numerical method

* Split-Step Fourier Method (SSFM)
* Alternating:

  * Linear step in frequency domain
  * Nonlinear step in time domain
* Fixed time step
* FFT-based spectral evolution

---

### Physical assumptions

#### Mean-field approximation

* Single averaged intracavity field per round trip
* No sub-round-trip dynamics

#### Single spatial mode

* Only one transverse mode family is considered
* No multimode coupling

#### Periodic boundary conditions

* Field periodic in fast time
* Discrete mode representation via FFT

#### Dispersion

* Defined through integrated dispersion (Dint)
* Can be imported or predefined
* Includes higher-order dispersion implicitly
* Static in time

#### Nonlinearity

* Instantaneous Kerr (χ³) only
* No Raman, Brillouin, or higher-order effects

#### Pump

* Continuous-wave coherent input
* Injected into central mode (μ = 0)

#### Loss and coupling

* Single decay rate κ
* External coupling defined by η
* No frequency dependence

#### Noise

* Complex Gaussian noise added to:

  * Pump field
  * Intracavity field
* Used to seed comb formation

#### Output field

* Through-port calculated using input–output relation:
  s_out = s_in − √κ_ex · a
* Spectrum reported in dBm per mode

#### Thermal and material effects

* No thermal dynamics
* No thermo-optic effects
* No parameter drift

---

### Simulation capabilities

The simulator can:

* Model Kerr comb initiation and evolution
* Reproduce modulation instability regimes
* Generate dissipative Kerr solitons
* Visualize spectral and temporal dynamics
* Compare with experimental dispersion data

---

### Limitations

The simulator does not:

* Solve full Maxwell equations
* Include multimode spatial dynamics
* Include thermal effects or feedback
* Include Raman or delayed nonlinear response
* Perform adaptive time stepping
* Guarantee quantitative agreement without parameter calibration

---

### Parameter representation

* All parameters can be expressed in:

  * Normalized units (LLE formalism)
  * Physical units (GHz, mW, ns)

* Internal conversion ensures consistency between:

  * Detuning
  * Power
  * Time scales

---

### Typical workflow (recommended)

1. Initialize with moderate pump power
2. Sweep detuning slowly
3. Observe:

   * Modulation instability onset
   * Comb formation
4. Identify soliton regime:

   * Smooth spectral envelope
   * Stable temporal pulse
5. Fine-tune parameters and record results

---
