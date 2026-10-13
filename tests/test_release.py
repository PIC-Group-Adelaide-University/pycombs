"""Release consistency, legacy solver equivalence, and executable EO recipes."""
from pathlib import Path
import importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def module(path):
 spec=importlib.util.spec_from_file_location('release_test_'+path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=(ROOT/'pycombs_v16r.py').read_text();entries=json.loads((ROOT/'examples.json').read_text());report={}
for e in entries:
 p=ROOT/e['path'];text=p.read_text();expected=base.replace('DEFAULT_STARTUP_PRESET = "default_startup_wilson2019_GaP"','DEFAULT_STARTUP_PRESET = '+json.dumps(e['preset']),1)
 assert text==expected,p
 m=module(p);st=m.LLEState();assert st.startup_config_name==e['preset'];assert np.isfinite(st.E_t_fast_norm).all()
 recipe=m.default_recipe_for_startup(e['preset']);assert recipe and recipe in m.EMBEDDED_RECIPES
 external=(p.parent/e['recipe']).read_text();a=m.RecipeRunner(st).parse_text(external);b=m.RecipeRunner(st).parse_text(m.EMBEDDED_RECIPES[recipe]);assert [(s['cmd'],s['args']) for s in a]==[(s['cmd'],s['args']) for s in b],e['preset']
 if e['group']!='Yi2017':
  old=next((ROOT/'Archive/Examples'/e['group']).glob('*v15R*'+p.stem.split('v16r_')[1]+'.py'));v=module(old);previous=v.LLEState();assert v.STARTUP_PRESETS[e['preset']]==m.STARTUP_PRESETS[e['preset']]
  assert not st.amc['enabled'];assert st.round_trips_per_integration==1
  for state in [st,previous]:
   state.noise_switch=False;state.pump_noise_enabled=False;state.cavity_noise_enabled=False
   state.startup_pump_noise_enabled=False;state.startup_cavity_noise_enabled=False;state.pump_refresh_noise_enabled=False
   state.E_t_fast_norm=.1+.01*np.cos(2*np.pi*np.arange(state.number_modes)/state.number_modes)+0j
   state.spectrum_E_t_fast_norm=m.fftshift(m.fft(state.E_t_fast_norm))
  st.step(50);previous.step(50)
  np.testing.assert_allclose(st.spectrum_E_t_fast_norm,previous.spectrum_E_t_fast_norm,rtol=1e-12,atol=1e-12)
  if e['group']=='Zhang2023':
   runner=m.RecipeRunner(st)
   def load(name):
    st.startup_config_name=name;m.apply_named_startup_preset(st,name);st.reinitialize(new_n_modes=st.number_modes)
   runner.bind_actions(load_preset=load);runner.load(recipe);runner.advance()
   assert runner.complete and not runner.error,runner.error
   assert st.eo_enabled, 'EO recipe must end with a real eo on command'
   assert not st.amc['enabled']
 else:
  assert st.amc['enabled'] and st.round_trips_per_integration==.5
  assert np.max(np.abs(st.spectrum_B_t_fast_norm))==0
  st.step(8);assert np.max(np.abs(st.spectrum_B_t_fast_norm))>0
 report[p.name]={'status':'passed','startup':e['preset'],'recipe':recipe};print('PASS',p.name,flush=True)
(ROOT/'validation_v16r.json').write_text(json.dumps(report,indent=2)+'\n')
