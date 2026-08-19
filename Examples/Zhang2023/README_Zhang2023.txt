Zhang et al. (2023) Fig. 3b — double-checked pycombs examples
================================================================

Reference
---------
K. Zhang, W. Sun, Y. Chen, et al., “A power-efficient integrated lithium
niobate electro-optic comb generator,” Communications Physics 6, 17 (2023).
DOI: 10.1038/s42005-023-01137-9

What is reproduced
------------------
The middle-row simulated EO-comb trends in Fig. 3b for the four device types
at the SAME RF drive condition reported by Zhang et al.:

  RF frequency:       24.95 GHz
  RF power:           28 dBm = 0.630957 W
  load/impedance:     50 ohm
  RF Vpi at 25 GHz:   7.56 V
  resulting Vpeak:    7.9433 V
  single-pass beta:   pi*Vpeak/Vpi = 1.0507 pi

Device mapping used in pycombs
------------------------------
Type I   : 1 coherent PM pass  -> ~1.05 pi
Type II  : 2 coherent PM passes -> ~2.1 pi
Type III : 2 coherent PM passes -> ~2.1 pi
Type IV  : 4 coherent PM passes -> ~4.2 pi

This mapping follows the paper: Type I is single-pass; Types II and III are
physically different double-pass layouts; Type IV is the quadra-pass device.
In the present ideal phase-only pycombs EO model, Types II and III therefore
have the same effective modulation index. The measured paper spectra can differ
because the physical layouts and non-ideal mode multiplexers are not modeled.

Important model scope
---------------------
This is deliberately a NON-RESONANT EO-comb validation. The Zhang startup
presets therefore use eta = 0, flat Dint, zero detuning, and no stochastic
noise, so that Kerr cavity dynamics cannot contaminate the comparison. The EO
field is generated analytically as phase modulation of a CW optical carrier.

The carrier wavelength in these files is 1550.8 nm solely to center the plotted
comb in the telecom band near the Fig. 3 spectra. The paper explicitly reports
that the device is wavelength-tunable (1520–1610 nm) and does not identify
1550.8 nm as a unique electrical condition required for Fig. 3b. The spectral
spacing and relative line amplitudes are set by the verified RF parameters and
modulation index, not this choice of carrier wavelength.

How to run
----------
For each Type I–IV:
  1. Run the matching pycombs_v12_Zhang2023_Fig3b_Type_*.py file.
  2. Load the matching pycombs_recipe_zhang2023_fig3b_type_*.txt file.
  3. Press Run Recipe.
  4. pycombs enables EO modulation and saves the output.

Why there are four dedicated files
----------------------------------
The Wilson and Pasquazi examples each open directly in the literature-specific
state. These four files follow the same principle: no collaborator needs to
edit a preset name or manually alter the number of modulation passes.

Literature cross-check (Zhang et al., Fig. 3b text)
--------------------------------------------------
- 28 dBm at 24.95 GHz.
- Vpeak = 7.94 V.
- RF Vpi = 7.56 V at 25 GHz (DC Vpi = 6.20 V).
- Type I: ~1.05 pi, 15 measured comb lines.
- Type II: ~2.1 pi, 25 measured comb lines.
- Type III: ~2.1 pi, 27 measured comb lines.
- Type IV: theoretical ~4.2 pi, 47 measured comb lines.

The 25 vs 27 line difference between Types II and III is experimental and is
not asserted to arise from a different ideal modulation index.
