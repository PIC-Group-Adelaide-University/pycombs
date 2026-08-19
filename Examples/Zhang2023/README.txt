ZHANG 2023 FIG. 3b — CLICK-TO-RUN PYCOMBS EXAMPLES
==================================================

Purpose
-------
Each Python file is dedicated to one Zhang et al. Fig. 3b comb type and now
pre-loads the matching Zhang recipe filename in the GUI. A collaborator can:

  1. Put the .py file and its matching recipe .txt in the same folder.
  2. Run the Python file.
  3. Click "Run recipe".

No filename editing is required.

Matching pairs
--------------
Type I
  pycombs_v12_Zhang2023_Fig3b_Type_I.py
  pycombs_recipe_zhang2023_fig3b_type_I.txt
  1 PM pass -> beta ~ 1.05*pi

Type II
  pycombs_v12_Zhang2023_Fig3b_Type_II.py
  pycombs_recipe_zhang2023_fig3b_type_II.txt
  2 PM passes -> beta ~ 2.10*pi

Type III
  pycombs_v12_Zhang2023_Fig3b_Type_III.py
  pycombs_recipe_zhang2023_fig3b_type_III.txt
  2 PM passes -> beta ~ 2.10*pi

Type IV
  pycombs_v12_Zhang2023_Fig3b_Type_IV.py
  pycombs_recipe_zhang2023_fig3b_type_IV.txt
  4 PM passes -> beta ~ 4.20*pi

Verified shared electrical conditions encoded in the dedicated startup presets
-----------------------------------------------------------------------------
RF frequency / comb spacing: 24.95 GHz
RF drive power:               28 dBm
RF Vpi:                       7.56 V
RF impedance:                 50 ohm
Intensity modulator:          disabled
Noise:                        disabled for the comparison
Kerr cavity coupling:         disabled for the EO-only validation case
Integrated dispersion:        flat/zero for this EO-only comparison

Important scope note
--------------------
These examples reproduce the ideal EO phase-modulation spectra corresponding
to the modulation indices shown in Zhang et al. They are not a full model of
fabrication imperfections, insertion-loss asymmetries, TE-mode routing, or
other experimental parasitic effects. In particular, Zhang Types II and III
are physically different devices even though both correspond to an ideal net
phase modulation of approximately 2.1*pi in the current Pycombs model.

Run-recipe behavior
-------------------
Each recipe explicitly performs:
  noise off -> EO off -> reset -> EO on -> wait -> save all
so that clicking "Run recipe" produces the intended EO-only example from the
startup state and saves the result.
