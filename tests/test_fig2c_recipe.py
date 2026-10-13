"""End-to-end recipe benchmark and equivalence to the original snapshots.
Run with MPLBACKEND=Agg python test_fig2c_recipe.py (several minutes).
"""
import importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'Examples/Yi2017'
spec=importlib.util.spec_from_file_location('v16recipe',ROOT/'pycombs_v16r_Yi2017_Fig2c_i.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
st=m.LLEState();runner=m.RecipeRunner(st)
def load_preset(name):
 st.startup_config_name=name;m.apply_named_startup_preset(st,name);st.reinitialize(new_n_modes=st.number_modes)
 assert st.round_trips_per_integration==.5
runner.bind_actions(load_preset=load_preset)
runner.load(str(ROOT/'recipe_yi2017_fig2c.txt'))
start=time.time();last=-1
while runner.active:
 runner.advance()
 if not runner.active:break
 n=runner.propagation_limit(137)
 if n:st.step(n)
 if runner.index!=last:
  print('Recipe step',runner.index,'elapsed',round(time.time()-start,1),flush=True);last=runner.index
assert runner.complete and not runner.error,runner.error
errors={}
for name,alpha in m.load_fig2c_benchmark_module().CASES:
 actual=np.load(runner.fig2c_output_dir/f'case_{name}.npz');expected=np.load(ROOT/'results'/f'case_{name}.npz')
 history=json.loads((runner.fig2c_output_dir/f'case_{name}_history.json').read_text())
 assert abs(history['history'][-1]['time_us']-2)<1e-10
 assert history['alpha']==alpha
 errors[name]={}
 for family in 'AB':
  a=abs(actual[family])**2;b=abs(expected[family])**2
  error=float(np.linalg.norm(a-b)/np.linalg.norm(b));errors[name][family]=error
  assert error<1e-9,(name,family,error)
assert (runner.fig2c_output_dir/'fig2c_comparison.png').is_file()
# Recipe loading clears a previous benchmark session; legacy recipes unaffected.
runner.load('default_recipe_yi2017_amc_hold');assert runner.fig2c_output_dir is None
assert runner.propagation_limit(137)==137
out={'status':'passed','recipe':'recipe_yi2017_fig2c.txt','relative_modal_power_L2_errors':errors,'elapsed_seconds':time.time()-start}
(ROOT.parents[1]/'validation_fig2c_v16r.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
