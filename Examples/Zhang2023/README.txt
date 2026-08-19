Zhang et al. (2023) EO-comb example for pycombs
================================================

Paper reproduced:
K. Zhang, W. Sun, Y. Chen, et al., Communications Physics 6, 17 (2023), Fig. 3b.

This folder mirrors the Wilson2019 and Pasquazi2018 examples: it contains a
dedicated pycombs script, a recipe, and the literature reference figure.

QUICK RUN — TYPE I (beta ~= 1.05*pi)
1. Run: pycombs_v12_Zhang2023_Fig3b_EO_comb.py
2. In pycombs, load: pycombs_recipe_zhang2023_fig3b_type_I.txt
3. Press Run Recipe.
4. The recipe enables the EO-comb stage and saves the output.

The dedicated script starts directly in:
  default_startup_zhang2023_fig3b_type_I_zero_coupling

Parameters encoded in that preset:
  pump wavelength = 1550.8 nm
  RF frequency / comb spacing = 24.95 GHz
  RF power = 28 dBm
  PM Vpi = 7.56 V
  PM passes = 1
  effective phase-modulation index ~= 1.05*pi
  cavity coupling eta = 0
  flat integrated dispersion
  noise = off

OTHER FIG. 3b CASES
The same script already contains these startup presets:
  default_startup_zhang2023_fig3b_type_II_zero_coupling  -> 2 PM passes (~2.1*pi)
  default_startup_zhang2023_fig3b_type_III_zero_coupling -> 2 PM passes (~2.1*pi)
  default_startup_zhang2023_fig3b_type_IV_zero_coupling  -> 4 PM passes (~4.2*pi)

Types II and III correspond to physically different layouts in Zhang et al., but
in this ideal phase-only pycombs model both reduce to two coherent modulation
passes and therefore give the same effective modulation index/spectrum.

To start directly in another case, change the single line
  self.startup_config_name = "default_startup_zhang2023_fig3b_type_I_zero_coupling"
to the desired preset name above, then use the same recipe.

Important scope:
This is an EO-comb validation example. It compares the phase-modulated comb
spectrum with Zhang et al. Fig. 3b. It does not claim to model the full physical
resonator/electrode layouts of Types II-IV.
