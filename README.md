# PyCombs V16R Ready

PyCombs provides an interactive Python GUI for Kerr microcombs, direct electro-optic comb generation, and avoided mode crossings using two coupled Lugiato–Lefever equations. This release synchronizes all manuscript examples with one engine.

## Install and run

Install Python 3.10 or newer, then run these commands from this folder:

```bash
python -m pip install -r requirements.txt
python pycombs.py
```

A graphical desktop and a Matplotlib GUI backend are required. PyQt5 supplies the Qt backend; the numerical checks can run without a desktop using `MPLBACKEND=Agg`.

The general launcher starts the Wilson GaP preset. Each standalone example starts its own preset and selects the matching embedded recipe. Open the script, then click **Run recipe** to run the example sequence. Presets define a configuration; recipes define its evolution and exports.

## Manuscript examples

| Example | Script in `Examples/` | Recipe / target |
| --- | --- | --- |
| Wilson GaP | `Wilson2019/pycombs_v16r_Wilson2019_GaP_simulated_soliton.py` | GaP parameter-based Kerr simulation |
| Pasquazi Fig. 19 | `Pasquazi2018/pycombs_v16r_Pasquazi2018_Fig19.py` | Slow scan towards normalized detuning 12 |
| Zhang Type I | `Zhang2023/pycombs_v16r_Zhang2023_Fig3b_Type_I.py` | One ideal phase-modulation pass |
| Zhang Type II | `Zhang2023/pycombs_v16r_Zhang2023_Fig3b_Type_II.py` | Two ideal passes |
| Zhang Type III | `Zhang2023/pycombs_v16r_Zhang2023_Fig3b_Type_III.py` | Two ideal passes; same spectrum as II in this approximation |
| Zhang Type IV | `Zhang2023/pycombs_v16r_Zhang2023_Fig3b_Type_IV.py` | Four ideal passes |
| Yi Fig. 2c | `Yi2017/pycombs_v16r_Yi2017_Fig2c_i.py` | Four-case AMC recipe; separate startup scripts also supplied for ii, iii, iv |

For example:

```bash
python Examples/Zhang2023/pycombs_v16r_Zhang2023_Fig3b_Type_II.py
python Examples/Yi2017/pycombs_v16r_Yi2017_Fig2c_i.py
```

Keep the complete example folder together. AMC presets use `fig2c_benchmark.py` as a companion for pulse initialization and plotting. The AMC recipe runs all four cases independently, writes 2 µs snapshots to a timestamped `yi2017_fig2c_runs/` directory, plots the spectra, and pauses the GUI. Allow several minutes; the GUI can take longer than the headless solver.

## Physical models and limits

The Kerr path uses symmetric second-order split-step Fourier integration of the mean-field LLE. Direct EO combs use analytical phase modulation of the input optical field. The AMC path adds a reciprocally coupled, unpumped auxiliary family, Kerr nonlinearities in both families, and a Raman shock term in the pumped family. The coupled linear half steps use an exact mode-wise 2 × 2 propagator with a constant-drive integral.

The AMC approximation omits inter-family cross-phase modulation and nonlinear mixing. It is not a full multimode Maxwell model, a thermal model, or a mode-locked laser simulator. The auxiliary-family export is intracavity power, not an independently calibrated through-port signal.

The Yi comparison is qualitative. Pump power, the bare auxiliary-family intercept, and the seeded A field are reconstruction choices. Case iii uses detuning 31 instead of the paper's approximately 29; case i peaks at mode 73 instead of 72; case iv remains transient. The published hysteresis branches at equal detuning have not been recovered. Numerical tests establish consistency for the tested states and intervals, not exact paper reproduction or general long-term stability.

The Wilson example uses reconstructed dispersion and Wilson-like device parameters; it does not establish reproduction of an experimentally observed soliton in that paper. The Pasquazi soliton count depends on noise and initial conditions. Zhang II and III represent distinct layouts with the same effective ideal modulation index in this model.

## Validation and maintenance

```bash
MPLBACKEND=Agg python tests/test_release.py
MPLBACKEND=Agg python tests/test_amc.py
MPLBACKEND=Agg python tests/test_gui_smoke.py
MPLBACKEND=Agg python tests/test_fig2c_recipe.py
```

The final command runs the complete four-case benchmark and compares modal powers against the bundled v16e reference snapshots. Reference snapshots in `Examples/Yi2017/results/` remain labelled as prior validation evidence; new simulations receive separate timestamped folders.

`pycombs_v16r.py` is the canonical source. After editing it, regenerate the standalone examples:

```bash
python tools/sync_examples.py
```

Only their startup preset differs. `examples.json` records every case, and the release test checks that all copies match their generated source and that external and embedded recipes have identical commands. Earlier code snapshots are retained in `Archive/` and are not the active release.

See [CHANGELOG_v16r.md](CHANGELOG_v16r.md) for collaborator review notes. The code is MIT licensed. The Yi reference figure is attributed in the example documentation and is distributed under CC BY 4.0.
