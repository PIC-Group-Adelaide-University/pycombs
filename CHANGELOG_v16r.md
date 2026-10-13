# V16R Ready changes for collaborator review

- Unified the general launcher, Wilson and Pasquazi examples, Zhang Types I–IV, and Yi cases i–iv on the same engine. Standalone copies are generated from a canonical source and checked for consistency.
- Integrated the coupled-family AMC path, its separate controls, pumped-family Raman shock term, auxiliary intracavity exports, and exact coupled linear propagation.
- Included the downloadable and embedded Yi Fig. 2c recipe with independent initialization, exact 2 µs endpoints, four-case exports, and comparison plotting.
- Retained the Kerr and EO presets and workflows. Loading a legacy preset turns AMC off and restores its original integration step.
- Corrected embedded recipe terminators that previously included a literal backslash-n. Corrected the stale Type I recipe selection in the Drive copies of Zhang II–IV through preset-based selection.
- Added run instructions, a dependencies file, numerical regression checks, headless GUI checks, and full AMC recipe equivalence validation. Earlier version snapshots remain available as history.

## Evidence and limitations

The release checks cover all ten launchers, their parameters, external/embedded recipe equivalence, the six existing cases against v15r propagation, EO activation, and five coupled-solver checks. The AMC recipe additionally compares all four full-length snapshots with the prior benchmark. Agg GUI tests exercise actual widgets and callbacks; a native Windows interactive session has not been tested here. The stochastic Wilson/Pasquazi full trajectories were not newly rerun to establish identical soliton counts.

The AMC spectral agreement with Yi et al. is qualitative; the detuning mismatch, transient case iv, and unverified hysteresis remain explicit. Please review the coupled equations, normalization, parameter reconstruction, numerical checks, and whether the intended release claims match this evidence.
