"""Agg GUI smoke checks for every launch preset; no desktop session needed."""
from pathlib import Path
import importlib.util,json
import matplotlib.pyplot as plt
from matplotlib.backend_bases import CloseEvent
ROOT=Path(__file__).resolve().parents[1]
plt.show=lambda *a,**k:None
spec=importlib.util.spec_from_file_location('gui_release',ROOT/'pycombs_v16r.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Suppress expensive Agg redraws, while constructing actual widgets and callbacks.
from matplotlib.backends.backend_agg import FigureCanvasAgg
original_idle=FigureCanvasAgg.draw_idle;FigureCanvasAgg.draw_idle=lambda self:None
for e in json.loads((ROOT/'examples.json').read_text()):
 m.DEFAULT_STARTUP_PRESET=e['preset'];m.run_gui();app=m.APP
 assert app.ui['txt_recipe'].text==m.default_recipe_for_startup(e['preset'])
 assert app.st.startup_config_name==e['preset']
 assert 'recipe_runner' in app.ui
 runner=app.ui['recipe_runner']
 if e['group']=='Zhang2023':
  runner.load(app.ui['txt_recipe'].text);runner.advance();assert runner.complete and not runner.error,runner.error;assert app.st.eo_enabled
 app.fig.canvas.callbacks.process('close_event',CloseEvent('close_event',app.fig.canvas))
 plt.close('all');print('PASS GUI',e['preset'],flush=True)
FigureCanvasAgg.draw_idle=original_idle
