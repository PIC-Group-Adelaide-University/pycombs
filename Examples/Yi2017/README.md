# Yi2017 manuscript example — V16R Ready

Install the dependencies in the root requirements.txt file. Start one of the following scripts, then click **Run recipe**. The appropriate embedded recipe is selected automatically.

- `pycombs_v16r_Yi2017_Fig2c_i.py` — recipe_yi2017_fig2c.txt
- `pycombs_v16r_Yi2017_Fig2c_ii.py` — recipe_yi2017_fig2c.txt
- `pycombs_v16r_Yi2017_Fig2c_iii.py` — recipe_yi2017_fig2c.txt
- `pycombs_v16r_Yi2017_Fig2c_iv.py` — recipe_yi2017_fig2c.txt

Keep the entire folder together. All scripts use the common V16R engine and include the AMC companion. External recipes have the same commands as their embedded versions. See the root README and CHANGELOG for validation and model limits.

The recipe always simulates all four cases, even when started from ii, iii, or iv. Existing results are prior reference snapshots; new simulations go to separate timestamped directories. The comparison is qualitative, and case iv is transient. Reference image: Yi et al., Nature Communications 8, 14869 (2017), doi:10.1038/ncomms14869, © The Authors 2017, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
