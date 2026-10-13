'''
Github page: zzd
References are found at the end of the file

Vocabulary:
. User defined variables are signed by a 'zzu; searchable token. Ex.: " #zzu DEFAULT_DINT_CSV: (...) "
. Comments on the code are assigned with a 'zzc' searcheable token. Ex.: " #zzc (...) "


'''


import numpy as np
import matplotlib.pyplot as plt
import os
import time
import matplotlib.animation as animation
import io

from matplotlib.widgets import Slider, Button, TextBox
from scipy.fft import fft, ifft, fftshift
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator 

t_start = time.time() #zzd


# ============================================================
# USER STARTUP CONFIGURATION
# Change only these values if you want to set the simulation
# initial conditions manually in one place.
# ============================================================

USER_STARTUP_CONFIG = dict(
    # Resonator / material
    number_modes=200,
    pump_wavelength_nm=1550.0,
    fsr_hz=1.0e12,
    Q=1e5,
    Aeff_um2=0.270,
    n2_m2_per_W=1.1e-17,
    eta=0.5,

    # Startup operating point
    # Detuning values and scan rate kept Wilson/GaP-like: DV = -3 -> 8 at 1 DV/ns.
    P_norm=2.4,
    power_slew_rate_Pnorm_per_ns=100.0,
    detuning_start_DV=-3.0,
    detuning_stop_DV=8.0,
    detuning_slew_rate_DV_per_ns=1.0,

    # Sources (preset name or filename)
    dint_source="default_dint_InGaP_SiO2_thi500nm_wid600nm_mode1",
    pump_source="default_cw_1550",

    # Noise
    noise_switch=True,
    pump_noise_enabled=True,
    cavity_noise_enabled=True,
    noise_level=1.0,

    # Fine-grained noise behavior
    startup_pump_noise_enabled=False,
    startup_cavity_noise_enabled=True,
    pump_refresh_noise_enabled=False,

    # Startup field style
    startup_style="noisy",

    # Optional extra startup values
    save_step_point=2000,
    plot_step=5000,
    temporal_interpol=False,

    # EO-comb pump generator (NEOS-like first version)
    eo_enabled=False,
    eo_rf_spacing_mu=1.0,
    eo_pm_rf_power_dBm=28.0,
    eo_pm_vpi_V=3.5,
    eo_num_pm=1,
    eo_pm_loss_dB=0.0,
    eo_pm_phase_rad=0.0,
    eo_im_enabled=False,
    eo_im_rf_power_dBm=25.0,
    eo_im_vpi_V=4.0,
    eo_im_bias=0.5,
    eo_im_loss_dB=0.0,
    eo_im_phase_rad=0.0,
    eo_rf_impedance_ohm=50.0,
)

STARTUP_PRESETS = {
    "default_startup_current": dict(USER_STARTUP_CONFIG),
    "default_startup_pasquazi2018_MgF2": dict(
        number_modes=200,
        pump_wavelength_nm=1553.0,
        fsr_hz=35.2e9,
        Q=4e8,
        Aeff_um2=1.6,
        n2_m2_per_W=1.1e-20,
        eta=0.5,
        P_norm=12.0,
        power_slew_rate_Pnorm_per_ns=100.0,
        detuning_start_DV=-4.0,
        detuning_stop_DV=15.0,
        detuning_slew_rate_DV_per_ns=0.01,
        dint_source="default_dint_pasquazi2018_MgF2",
        pump_source="default_cw_1550",
        noise_switch=True,
        pump_noise_enabled=True,
        cavity_noise_enabled=True,
        noise_level=1.0,
        startup_pump_noise_enabled=True,
        startup_cavity_noise_enabled=True,
        pump_refresh_noise_enabled=True,
        startup_style="noisy",
        save_step_point=2000,
        plot_step=5000,
        temporal_interpol=False,
    ),
    "default_startup_pasquazi2018_fig19_MgF2_scan": dict(
        # Pasquazi et al., Physics Reports 729 (2018), Fig. 19.
        # 35.2 GHz MgF2 resonator, normalized pump power X = 12
        # (reported physical input power approximately 9.3 mW).
        # The companion recipe scans normalized detuning from -4 to +15.5.
        number_modes=512,
        pump_wavelength_nm=1553.0,
        fsr_hz=35.2e9,
        Q=4e8,
        Aeff_um2=1.6,
        n2_m2_per_W=1.1e-20,
        eta=0.5,
        P_norm=12.0,
        power_slew_rate_Pnorm_per_ns=100.0,
        detuning_start_DV=-4.0,
        detuning_stop_DV=15.5,
        detuning_slew_rate_DV_per_ns=0.01,
        dint_source="default_dint_pasquazi2018_MgF2",
        pump_source="default_cw_1550",
        noise_switch=True,
        pump_noise_enabled=True,
        cavity_noise_enabled=True,
        noise_level=1.0,
        startup_pump_noise_enabled=False,
        startup_cavity_noise_enabled=True,
        pump_refresh_noise_enabled=False,
        startup_style="noisy",
        save_step_point=1000,
        plot_step=1000,
        temporal_interpol=False,
        eo_enabled=False,
    ),
    "default_startup_pasquazi2018_fig19_exact_quadratic": dict(
        # Dedicated reproduction preset for Pasquazi et al., Physics Reports
        # 729 (2018), Fig. 19. The paper solves the elementary normalized LLE
        # with X = 12 while Delta is adiabatically scanned from -4 to +15.5.
        # The embedded MgF2 Dint profile used here is purely quadratic over the
        # simulated mode range, so higher-order dispersion is excluded.
        number_modes=512,
        pump_wavelength_nm=1553.0,
        fsr_hz=35.2e9,
        Q=4e8,
        Aeff_um2=1.6,
        n2_m2_per_W=1.1e-20,
        eta=0.5,
        P_norm=12.0,
        power_slew_rate_Pnorm_per_ns=100.0,
        detuning_start_DV=-4.0,
        detuning_stop_DV=15.5,
        # Slow enough for the field to follow the nonlinear branch and enter
        # the MI, chaotic, breathing-soliton and stable-soliton regimes.
        detuning_slew_rate_DV_per_ns=0.0003,
        dint_source="default_dint_pasquazi2018_MgF2",
        pump_source="default_cw_1550",
        noise_switch=True,
        pump_noise_enabled=True,
        cavity_noise_enabled=True,
        noise_level=1.0,
        startup_pump_noise_enabled=False,
        startup_cavity_noise_enabled=True,
        pump_refresh_noise_enabled=False,
        startup_style="noisy",
        save_step_point=1000,
        plot_step=1000,
        temporal_interpol=False,
        eo_enabled=False,
    ),
    "default_startup_pasquazi2018_fig25_SiN_DW": dict(
        # Pasquazi et al., Physics Reports 729 (2018), Fig. 25.
        # Approximate 226 GHz Si3N4 resonator used to illustrate dispersive-wave
        # emission. The source paper states: finesse = 350, pump power = 760 mW,
        # normalized pump strength X = 6, stable cavity-soliton spectrum at
        # detuning Delta = 6, and unstable-MI spectrum at Delta = 3.
        number_modes=1024,
        pump_wavelength_nm=1562.0,
        fsr_hz=226.0e9,
        R_res_um=100.0,
        Q=2.972345e5,
        Aeff_um2=1.51,
        n2_m2_per_W=2.4e-19,
        eta=0.5,
        P_phys_W=0.760,
        P_norm=6.0,
        power_slew_rate_Pnorm_per_ns=0.02,
        detuning_start_DV=-4.0,
        detuning_stop_DV=6.0,
        detuning_slew_rate_DV_per_ns=0.002,
        dint_source="default_dint_pasquazi2018_fig25_SiN_DW",
        pump_source="default_cw_1550",
        noise_switch=True,
        pump_noise_enabled=True,
        cavity_noise_enabled=True,
        noise_level=1.0,
        startup_pump_noise_enabled=False,
        startup_cavity_noise_enabled=True,
        pump_refresh_noise_enabled=False,
        startup_style="noisy",
        save_step_point=1000,
        plot_step=1000,
        temporal_interpol=False,
        eo_enabled=False,
    ),
    "default_startup_zhang2023_fig3b_type_I": dict(
        # Zhang et al., Commun. Phys. 6, 17 (2023), Fig. 3b, device I.
        # Single-pass non-resonant LN phase modulator: 28 dBm at 24.95 GHz,
        # RF Vpi = 7.56 V -> beta = pi*Vpeak/Vpi ~= 1.05*pi (15 lines).
        number_modes=256,
        pump_wavelength_nm=1550.8,
        fsr_hz=24.95e9,
        Q=4e8,
        Aeff_um2=1.6,
        n2_m2_per_W=1.1e-20,
        eta=0.5,
        P_norm=12.0,
        power_slew_rate_Pnorm_per_ns=100.0,
        detuning_start_DV=0.0,
        detuning_stop_DV=0.0,
        detuning_slew_rate_DV_per_ns=0.0,
        dint_source="default_dint_pasquazi2018_MgF2",
        pump_source="default_cw_1550",
        startup_style="cw_only",
        noise_switch=False,
        pump_noise_enabled=False,
        cavity_noise_enabled=False,
        noise_level=0.0,
        startup_pump_noise_enabled=False,
        startup_cavity_noise_enabled=False,
        pump_refresh_noise_enabled=False,
        save_step_point=2000,
        plot_step=5000,
        temporal_interpol=False,
        eo_enabled=True,
        eo_rf_spacing_mu=1.0,
        eo_pm_rf_power_dBm=28.0,
        eo_pm_vpi_V=7.56,
        eo_num_pm=1,
        eo_pm_loss_dB=0.0,
        eo_pm_phase_rad=0.0,
        eo_im_enabled=False,
        eo_rf_impedance_ohm=50.0,
    ),
    "default_startup_zhang2023_fig3b_type_I_zero_coupling": dict(
        # Zhang et al., Fig. 3b, Type I: single-pass EO-comb validation.
        # Start from the experimental CW laser state; press Apply EO to
        # generate the simulated 1.05π profile (about 15 visible lines).
        number_modes=256,
        pump_wavelength_nm=1550.8,
        fsr_hz=24.95e9,
        Q=4e8,
        Aeff_um2=1.6,
        n2_m2_per_W=2.5e-19,
        eta=0.0,
        P_phys_W=20e-3,
        P_norm=0.0,
        power_slew_rate_Pnorm_per_ns=0.0,
        detuning_start_DV=0.0,
        detuning_stop_DV=0.0,
        detuning_slew_rate_DV_per_ns=0.0,
        dint_source="default_dint_zero_flat",
        pump_source="default_cw_1550",
        startup_style="cw_only",
        noise_switch=False,
        pump_noise_enabled=False,
        cavity_noise_enabled=False,
        noise_level=0.0,
        startup_pump_noise_enabled=False,
        startup_cavity_noise_enabled=False,
        pump_refresh_noise_enabled=False,
        save_step_point=2000,
        plot_step=5000,
        temporal_interpol=False,
        eo_enabled=False,
        eo_rf_spacing_mu=1.0,
        eo_pm_rf_power_dBm=28.0,
        eo_pm_vpi_V=7.56,
        eo_num_pm=1,
        eo_pm_loss_dB=0.0,
        eo_pm_phase_rad=0.0,
        eo_im_enabled=False,
        eo_rf_impedance_ohm=50.0,
    ),
    "default_startup_wilson2019_GaP": dict(
        number_modes=200,
        pump_wavelength_nm=1550.0,
        fsr_hz=1.0e12,
        Q=7.7e4,
        Aeff_um2=0.2,
        n2_m2_per_W=1.1e-17,
        eta=0.5,
        P_norm=2.4,
        power_slew_rate_Pnorm_per_ns=100.0,
        P_phys_W=36e-3,
        detuning_start_DV=-3.0,
        detuning_stop_DV=8.0,
        detuning_slew_rate_DV_per_ns=1.0,
        dint_source="default_dint_wilson2019_GaP",
        # Use the embedded CW source so the Wilson setup is self-contained.
        pump_source="default_cw_1550",
        startup_style="noisy",
        noise_switch=True,
        pump_noise_enabled=True,
        cavity_noise_enabled=True,
        noise_level=1.0,
        startup_pump_noise_enabled=False,
        # A weak intracavity seed is required for spontaneous MI/comb growth.
        startup_cavity_noise_enabled=True,
        pump_refresh_noise_enabled=False,
        save_step_point=2000,
        plot_step=5000,
        temporal_interpol=False,
    ),
    "default_startup_InGaP_SiO2_500x600_soliton": dict(
        # InGaP / SiO2 geometry-specific startup: thickness 500 nm, width 600 nm, mode 1
        # Variant requested: lower Q, Wilson/GaP-like 1 THz FSR, Pnorm = 2.4, and Wilson/GaP-like detuning ratio/scan.
        number_modes=200,
        pump_wavelength_nm=1550.0,
        fsr_hz=1.0e12,
        Q=1e5,
        Aeff_um2=0.270,
        n2_m2_per_W=1.1e-17,
        eta=0.5,
        P_norm=2.4,
        power_slew_rate_Pnorm_per_ns=100.0,
        detuning_start_DV=-3.0,
        detuning_stop_DV=8.0,
        detuning_slew_rate_DV_per_ns=1.0,
        dint_source="default_dint_InGaP_SiO2_thi500nm_wid600nm_mode1",
        pump_source="default_cw_1550",
        startup_style="cw_only",
        noise_switch=True,
        pump_noise_enabled=True,
        cavity_noise_enabled=True,
        noise_level=1.0,
        startup_pump_noise_enabled=False,
        startup_cavity_noise_enabled=True,
        pump_refresh_noise_enabled=False,
        save_step_point=2000,
        plot_step=5000,
        temporal_interpol=False,
    ),
}


# Zhang et al. (2023), Fig. 3b ideal non-resonant EO-comb startup presets.
# All four cases begin from the same CW laser and generate the comb only
# when the user presses Apply EO / enables the EO-comb stage.
# Types II and III are physically different layouts, but in the ideal phase-only
# simulation both provide two coherent modulation passes and therefore the same
# effective modulation index (~2.1*pi at 28 dBm).
for _zhang_name, _zhang_passes in (
    ("default_startup_zhang2023_fig3b_type_II_zero_coupling", 2),
    ("default_startup_zhang2023_fig3b_type_III_zero_coupling", 2),
    ("default_startup_zhang2023_fig3b_type_IV_zero_coupling", 4),
):
    STARTUP_PRESETS[_zhang_name] = dict(
        STARTUP_PRESETS["default_startup_zhang2023_fig3b_type_I_zero_coupling"],
        eo_num_pm=_zhang_passes,
    )


def apply_user_startup_config(st, cfg=None):
    """Apply the user startup configuration to an LLEState instance."""
    cfg = USER_STARTUP_CONFIG if cfg is None else cfg

    st.number_modes = int(cfg.get("number_modes", st.number_modes))
    st.wvl_pump = float(cfg.get("pump_wavelength_nm", st.wvl_pump * 1e9)) * 1e-9
    st.frq_pump = st.c / st.wvl_pump
    st.ome_pump = 2 * np.pi * st.frq_pump

    st.fsr = float(cfg.get("fsr_hz", st.fsr))
    if "R_res_um" in cfg:
        st.R_res = float(cfg["R_res_um"]) * 1e-6
        st.L = 2.0 * np.pi * st.R_res
    st.Q = float(cfg.get("Q", st.Q))
    st.Aeff = float(cfg.get("Aeff_um2", st.Aeff * 1e12)) * 1e-12
    st.n2 = float(cfg.get("n2_m2_per_W", st.n2))
    st.eta = float(cfg.get("eta", st.eta))

    st.P_in_phys = cfg.get("P_phys_W", None)
    if st.P_in_phys is not None:
        st.P_in_phys = float(st.P_in_phys)
    st.P_in_norm_default = [float(cfg.get("P_norm", st.P_in_norm_default[0]))]
    st.power_slew_rate = float(cfg.get("power_slew_rate_Pnorm_per_ns", getattr(st, "power_slew_rate", 100.0)))

    st.Detuning_normalized_start = float(cfg.get("detuning_start_DV", st.Detuning_normalized_start))
    st.Detuning_normalized_stop = float(cfg.get("detuning_stop_DV", st.Detuning_normalized_stop))
    st.detuning_sweep_rate = float(cfg.get("detuning_slew_rate_DV_per_ns", getattr(st, "detuning_slew_rate", 0.01)))

    st.dint_file_choice = True
    st.dint_file_path = str(cfg.get("dint_source", st.dint_file_path))
    st.input_field_file = str(cfg.get("pump_source", getattr(st, "input_field_file", "default_cw_1550")))

    st.noise_switch = bool(cfg.get("noise_switch", st.noise_switch))
    st.pump_noise_enabled = bool(cfg.get("pump_noise_enabled", st.pump_noise_enabled))
    st.cavity_noise_enabled = bool(cfg.get("cavity_noise_enabled", st.cavity_noise_enabled))
    st.noise_level = max(float(cfg.get("noise_level", getattr(st, "noise_level", 1.0))), 0.0)
    if hasattr(st, "_update_noise_amplitudes"):
        st._update_noise_amplitudes()
    st.startup_pump_noise_enabled = bool(
        cfg.get("startup_pump_noise_enabled", getattr(st, "startup_pump_noise_enabled", True))
    )
    st.startup_cavity_noise_enabled = bool(
        cfg.get("startup_cavity_noise_enabled", getattr(st, "startup_cavity_noise_enabled", True))
    )
    st.pump_refresh_noise_enabled = bool(
        cfg.get("pump_refresh_noise_enabled", getattr(st, "pump_refresh_noise_enabled", True))
    )

    # EO-comb settings are kept here so presets can define them later.
    st.eo_enabled = bool(cfg.get("eo_enabled", getattr(st, "eo_enabled", False)))
    st.eo_rf_spacing_mu = float(cfg.get("eo_rf_spacing_mu", getattr(st, "eo_rf_spacing_mu", 1.0)))
    st.eo_pm_rf_power_dBm = float(cfg.get("eo_pm_rf_power_dBm", getattr(st, "eo_pm_rf_power_dBm", 28.0)))
    st.eo_pm_vpi_V = float(cfg.get("eo_pm_vpi_V", getattr(st, "eo_pm_vpi_V", 3.5)))
    st.eo_num_pm = int(cfg.get("eo_num_pm", getattr(st, "eo_num_pm", 1)))
    st.eo_pm_loss_dB = float(cfg.get("eo_pm_loss_dB", getattr(st, "eo_pm_loss_dB", 0.0)))
    st.eo_pm_phase_rad = float(cfg.get("eo_pm_phase_rad", getattr(st, "eo_pm_phase_rad", 0.0)))
    st.eo_im_enabled = bool(cfg.get("eo_im_enabled", getattr(st, "eo_im_enabled", False)))
    st.eo_im_rf_power_dBm = float(cfg.get("eo_im_rf_power_dBm", getattr(st, "eo_im_rf_power_dBm", 25.0)))
    st.eo_im_vpi_V = float(cfg.get("eo_im_vpi_V", getattr(st, "eo_im_vpi_V", 4.0)))
    st.eo_im_bias = float(cfg.get("eo_im_bias", getattr(st, "eo_im_bias", 0.5)))
    st.eo_im_loss_dB = float(cfg.get("eo_im_loss_dB", getattr(st, "eo_im_loss_dB", 0.0)))
    st.eo_im_phase_rad = float(cfg.get("eo_im_phase_rad", getattr(st, "eo_im_phase_rad", 0.0)))
    st.eo_rf_impedance_ohm = float(cfg.get("eo_rf_impedance_ohm", getattr(st, "eo_rf_impedance_ohm", 50.0)))
    st.input_power_amplitude_scale = float(getattr(st, "input_power_amplitude_scale", 1.0))

    st.save_step_point = int(cfg.get("save_step_point", getattr(st, "save_step_point", 2000)))
    st.plot_step = int(cfg.get("plot_step", getattr(st, "plot_step", 5000)))
    st.temporal_interpol = bool(cfg.get("temporal_interpol", getattr(st, "temporal_interpol", False)))
    st.startup_style = str(cfg.get("startup_style", getattr(st, "startup_style", "noisy")))

def apply_named_startup_preset(st, preset_name):
    """Apply a named startup preset to an LLEState instance."""
    preset_name = (preset_name or "").strip()
    if preset_name not in STARTUP_PRESETS:
        raise KeyError(
            f"Unknown startup preset: {preset_name}. "
            f"Available presets: {', '.join(STARTUP_PRESETS.keys())}"
        )
    apply_user_startup_config(st, STARTUP_PRESETS[preset_name])


def get_startup_runtime_defaults(st):
    """Return the full live operating-point defaults encoded in the startup preset/state."""
    pnorm_default = getattr(st, "P_in_norm_default", [getattr(st, "P_norm", 0.0)])
    if isinstance(pnorm_default, (list, tuple, np.ndarray)):
        pnorm_default = float(pnorm_default[0])
    else:
        pnorm_default = float(pnorm_default)

    pphys_default = getattr(st, "P_in_phys", None)
    if pphys_default is not None:
        pphys_default = float(pphys_default)
        # Physical power is authoritative when present, just like in the old Wilson code.
        pnorm_default = float(pphys_default / st.P_in_norm_factor)

    return dict(
        DV=float(getattr(st, "Detuning_normalized_start", getattr(st, "DV", 0.0))),
        slew=float(getattr(st, "detuning_sweep_rate", 0.01)),
        Pnorm=pnorm_default,
        Pphys=pphys_default,
        power_slew=float(getattr(st, "power_slew_rate", 100.0)),
    )


def sync_sidebar_from_state(ui, st):
    """Update sidebar text boxes to reflect the current state."""
    mapping = {
        "startup": getattr(st, "startup_config_name", ""),
        "modes": f"{st.number_modes}",
        "fsr": f"{st.fsr:.2e}",
        "lambda": f"{st.wvl_pump * 1e9:.1f}",
        "q": f"{st.Q:.2e}",
        "aeff": f"{st.Aeff * 1e12:.2f}",
        "n2": f"{st.n2:.2e}",
        "eta": f"{st.eta:.3f}",
        "dint": getattr(st, "dint_file_path", ""),
        "pump": getattr(st, "input_field_file", ""),
        "noise_level": f"{getattr(st, 'noise_level', 1.0):.3g}",
    }
    for key, val in mapping.items():
        tb = ui.get(f"txt_{key}")
        if tb is not None:
            tb.set_val(val)



#zzu DEFAULT_DINT_CSV: Default Dint (integrated dispersion) values used for the simulation if none are given through the code or interface
#zzc Originally taken graphically from [4], cited by [5]

# Embedded defaults are grouped near the end of the file.


class pycombsApp:
    def __init__(self):
        self.fig = None
        self.ui = None
        self.st = None
        self.ani = None


class RecipeRunner:
    """Small non-blocking recipe runner for the live Matplotlib GUI.

    Recipe files are plain text. Supported commands are intentionally simple.

    Recommended recipe names:
      set detuning_dv <DV>
      set detuning_ghz <GHz>
      set detuning_rate_dv_per_ns <DV/ns>
      set detuning_rate_ghz_per_ns <GHz/ns>
      set pump_power_norm <Pnorm>
      set pump_power_mw <mW>
      set pump_power_rate_norm_per_ns <Pnorm/ns>
      set pump_power_rate_mw_per_ns <mW/ns>

    Short aliases still accepted:
      set dv <DV>
      set pnorm <Pnorm>
      set slew <DV/ns>
      set power_rate <Pnorm/ns>

    Other commands:
      set noise_level <value>
      noise on|off
      eo on|off
      wait_ns <simulated_ns>
      wait_steps <integer>
      apply_live
      apply_reset
      save_plot | save_data | save_pulse | save_all

    Lines starting with # are ignored. Commas are treated as spaces.
    The runner is advanced from the animation loop, so it does not freeze the UI.
    """

    def __init__(self, st, status_artist=None):
        self.st = st
        self.status_artist = status_artist
        self.steps = []
        self.index = 0
        self.active = False
        self.complete = False
        self.error = None
        self.recipe_path = None
        self.wait_until_ns = None
        self.wait_until_j = None
        self.actions = {}
        self._last_status = "Recipe status: idle"
        self.ui_sync_needed = False

    def bind_actions(self, **actions):
        self.actions.update(actions)

    def current_time_ns(self):
        return 1e9 * (2.0 / self.st.kappa_avg) * (self.st.j * self.st.tal_step)

    def set_status(self, text):
        self._last_status = str(text)
        if self.status_artist is not None:
            display_text = self._last_status
            if getattr(self.status_artist, "_pycombs_recipe_sidebar", False):
                words = display_text.split()
                lines, line = [], ""
                for word in words:
                    trial = (line + " " + word).strip()
                    if len(trial) > 24 and line:
                        lines.append(line)
                        line = word
                    else:
                        line = trial
                if line:
                    lines.append(line)
                display_text = "\n".join(lines[:4])
            self.status_artist.set_text(display_text)

    def parse_file(self, path):
        parsed = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.replace(",", " " ).split()
                if not parts:
                    continue
                cmd = parts[0].lower()
                args = parts[1:]
                label = " ".join(parts)
                parsed.append(dict(cmd=cmd, args=args, label=label, line=line_no))
        if not parsed:
            raise ValueError("Recipe file has no executable steps.")
        return parsed

    def load(self, path):
        path = (path or "").strip().strip('\"').strip("'")
        if not path:
            raise ValueError("Empty recipe filename.")
        candidates = [path]
        try:
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
        except Exception:
            pass
        candidates.append(os.path.join(os.getcwd(), path))
        file_found = next((c for c in candidates if os.path.isfile(c)), None)
        if file_found is None:
            raise FileNotFoundError(f"Recipe file not found: {path}")

        self.steps = self.parse_file(file_found)
        self.index = 0
        self.active = True
        self.complete = False
        self.error = None
        self.recipe_path = file_found
        self.wait_until_ns = None
        self.wait_until_j = None
        self.ui_sync_needed = True
        self.set_status(f"Recipe loaded: {os.path.basename(file_found)} | step 1/{len(self.steps)}")
        print(f"[Recipe] Loaded {len(self.steps)} steps from: {file_found}")

    def finish(self):
        self.active = False
        self.complete = True
        self.set_status("Recipe complete")
        print("[Recipe] Complete.")

    def fail(self, message):
        self.active = False
        self.complete = False
        self.error = str(message)
        self.set_status(f"Recipe error: {self.error}")
        print(f"[Recipe] ERROR: {self.error}")

    def _execute_instant(self, step):
        cmd = step["cmd"]
        args = step["args"]

        if cmd == "set":
            if len(args) < 2:
                raise ValueError("set requires a parameter name and value")
            key = args[0].lower()
            val = args[1]
            if key in ("detuning_dv", "detuning", "dv"):
                self.st.set_targets(detuning=float(val))
            elif key in ("detuning_ghz", "detuning_ghz_value"):
                dv = float(val) * 1e9 * (2.0 * np.pi) / self.st.kappa_avg
                self.st.set_targets(detuning=dv)
            elif key in ("detuning_rate_dv_per_ns", "detuning_slew_dv_per_ns", "slew", "detuning_slew", "detuning_rate"):
                self.st.detuning_slew_rate = max(float(val), 0.0)
            elif key in ("detuning_rate_ghz_per_ns", "detuning_slew_ghz_per_ns"):
                self.st.detuning_slew_rate = max(float(val), 0.0) * 1e9 * (2.0 * np.pi) / self.st.kappa_avg
            elif key in ("pump_power_norm", "power_norm", "power", "pnorm", "p_norm"):
                self.st.set_targets(P_norm=float(val))
            elif key in ("pump_power_mw", "power_mw"):
                self.st.set_targets(P_norm=(float(val) * 1e-3) / self.st.P_in_norm_factor)
            elif key in ("pump_slew_norm_per_ns", "pump_power_slew_norm_per_ns", "pump_power_rate_norm_per_ns", "power_rate_norm_per_ns", "power_slew", "power_rate", "power_slew_rate", "pnorm_slew"):
                self.st.power_slew_rate = max(float(val), 0.0)
            elif key in ("pump_power_rate_mw_per_ns", "power_rate_mw_per_ns"):
                self.st.power_slew_rate = max(float(val), 0.0) * 1e-3 / self.st.P_in_norm_factor
            elif key in ("noise_level", "noise"):
                self.st.set_noise_level(float(val), refresh_pump=True)
            elif key in ("pump", "pump_source"):
                self.st.input_field_file = str(val)
                self.st._load_user_or_default_pump(self.st.input_field_file)
                self.st._rebuild_input_pump(add_startup_noise=False)
            elif key in ("dint", "dispersion"):
                self.st.dint_file_path = str(val)
                self.st._load_dispersion_with_fallback()
                self.st.dint_norm = 2 * self.st.dint / self.st.kappa_avg
            else:
                raise ValueError(f"Unknown set parameter: {key}")
            self.ui_sync_needed = True
            return

        if cmd == "wait_ns":
            if not args:
                raise ValueError("wait_ns requires a duration in ns")
            self.wait_until_ns = self.current_time_ns() + max(float(args[0]), 0.0)
            return "waiting"

        if cmd == "wait_steps":
            if not args:
                raise ValueError("wait_steps requires an integer number of solver steps")
            self.wait_until_j = int(self.st.j) + max(int(float(args[0])), 0)
            return "waiting"

        if cmd == "noise":
            if not args:
                raise ValueError("noise requires on or off")
            action = self.actions.get("set_all_noise")
            if action is None:
                raise RuntimeError("Noise action is not bound.")
            action(args[0].lower() in ("on", "true", "1", "yes"))
            self.ui_sync_needed = True
            return

        if cmd == "eo":
            if not args:
                raise ValueError("eo requires on or off")
            self.st.apply_eo_comb_pump(enabled=args[0].lower() in ("on", "true", "1", "yes"), rebuild=True)
            self.ui_sync_needed = True
            return

        action_map = {
            "apply_live": "apply_live",
            "apply_reset": "apply_reset",
            "reset": "apply_reset",
            "save_plot": "save_plot",
            "save_data": "save_data",
            "save_pulse": "save_pulse",
            "save_pump": "save_pulse",
            "save_all": "save_all",
        }
        if cmd in action_map:
            action = self.actions.get(action_map[cmd])
            if action is None:
                raise RuntimeError(f"Recipe action is not bound: {cmd}")
            action()
            self.ui_sync_needed = True
            return

        raise ValueError(f"Unknown recipe command: {cmd}")

    def advance(self):
        if not self.active or self.error:
            return

        try:
            if self.wait_until_ns is not None:
                if self.current_time_ns() < self.wait_until_ns:
                    remaining = self.wait_until_ns - self.current_time_ns()
                    self.set_status(f"Recipe step {self.index + 1}/{len(self.steps)}: waiting {remaining:.2f} ns")
                    return
                self.wait_until_ns = None
                self.index += 1

            if self.wait_until_j is not None:
                if int(self.st.j) < self.wait_until_j:
                    remaining = self.wait_until_j - int(self.st.j)
                    self.set_status(f"Recipe step {self.index + 1}/{len(self.steps)}: waiting {remaining} solver steps")
                    return
                self.wait_until_j = None
                self.index += 1

            while self.index < len(self.steps):
                step = self.steps[self.index]
                self.set_status(f"Recipe step {self.index + 1}/{len(self.steps)}: {step['label']}")
                result = self._execute_instant(step)
                if result == "waiting":
                    return
                self.index += 1

            self.finish()
        except Exception as e:
            step_line = None
            if 0 <= self.index < len(self.steps):
                step_line = self.steps[self.index].get("line")
            prefix = f"line {step_line}: " if step_line is not None else ""
            self.fail(prefix + str(e))


APP = None

class LLEState:
    
    def __init__(self):
        '''Constants'''

        self.hbar = 1.0545718e-34  # [J*s] Barred Plancks constant
        self.c = 299792458         # [m/s] Speed of light
        self.i = 1j                # Imaginary unit

        '''Pump light / user-startup-configurable independent variables'''

        self.wvl_pump = 1553e-9                      #zzu [m] Pump wavelength
        self.frq_pump = self.c / self.wvl_pump       #zzu [Hz] Pump frequency
        self.ome_pump = 2 * np.pi * self.frq_pump    #zzu [rad/s] Pump angular frequency

        '''Dispersion input'''

        self.dint_file_choice = True #zzu Decides if a Dint file is used (True) or not (False)
        self.dint_file_path = "default_dint_pasquazi2018_MgF2" #zzu
        self.d2_file_path = "700w_d2_output.txt" #zzu 
        self.d2_default = 50  # [rad/s]  (check units / meaning)

        '''Resonator parameters'''

        self.R_res = (27e-6) / 2                # [m]
        self.L = self.R_res * np.pi * 2         # [m]
        self.fsr = 35.2e9                       # [Hz]
        self.Q = 4e8                            # [-]

        '''Material parameters and nonlinearity'''

        self.n2 = 1.1e-20       # [m^2/W] (check your chosen units)
        self.gamma_v = 2.4e2    # [1/(W*m)] (check)
        self.Aeff = 1.6 * ((1e-6)**2) # [m^2]
        
        '''Power and coupling'''

        self.P_in_phys = None
        self.P_in_norm_default = [12]
        self.power_slew_rate = 100.0  # [Pnorm/ns] slew limit for pump power target changes
        self.eta = 0.5                         # [-] coupling efficiency
        # self.kappa_ex will be derived below

        '''Detuning start and stop'''

        self.Detuning_normalized_start = -4
        self.Detuning_normalized_stop = 15
        self.detuning_sweep_rate = 0.01

        '''Simulation parameters'''

        self.number_modes = 200
        self.save_step_point = 2000
        self.plot_step = 5000
        
        '''Noise control'''

        self.noise_switch = True
        self.pump_noise_enabled = True
        self.cavity_noise_enabled = True
        self.noise_level = 1.0
        self.startup_pump_noise_enabled = True
        self.startup_cavity_noise_enabled = True
        self.pump_refresh_noise_enabled = True
        self.startup_style = "noisy"

        '''EO-comb pump generator defaults'''
        self.eo_enabled = False
        self.eo_rf_spacing_mu = 1.0
        self.eo_pm_rf_power_dBm = 28.0
        self.eo_pm_vpi_V = 3.5
        self.eo_num_pm = 1
        self.eo_pm_loss_dB = 0.0
        self.eo_pm_phase_rad = 0.0
        self.eo_im_enabled = False
        self.eo_im_rf_power_dBm = 25.0
        self.eo_im_vpi_V = 4.0
        self.eo_im_bias = 0.5
        self.eo_im_loss_dB = 0.0
        self.eo_im_phase_rad = 0.0
        self.eo_rf_impedance_ohm = 50.0
        self.eo_last_beta_pm = 0.0
        self.eo_last_beta_im = 0.0
        self.eo_last_transmission = 1.0
        self.input_power_amplitude_scale = 1.0
        self.eo_previous_pump_source = "default_cw_1550"
        self.eo_output_field_label = "EO-modulated CW field"

        # Apply the user-facing startup block in one shot.
        # Start from the zero-coupling, zero-dispersion Zhang validation case.
        self.startup_config_name = "default_startup_wilson2019_GaP"
        apply_named_startup_preset(self, self.startup_config_name)

        '''Derived physical parameters'''

        self.t_phys_round_trip = 1 / self.fsr                 # [s]
        self.linewidth = self.frq_pump / self.Q               # [Hz]
        self.kappa_avg = self.linewidth * 2 * np.pi           # [rad/s]
        self.alpha = self.t_phys_round_trip * self.kappa_avg / 2
        self.norm_t = 1 / (self.alpha / self.t_phys_round_trip)
        self.kappa_ex = self.eta * self.kappa_avg

        self.mu = np.arange(-self.number_modes // 2, self.number_modes // 2)
        self.frq_grid = self.frq_pump + self.mu * self.fsr
        self.ome_grid = self.frq_grid * (2 * np.pi)
        self.wvl_grid = self.c / self.frq_grid

        '''Definitions of time and integration time for the simulation'''

        self.round_trips_per_integration = 1
        self.tal_step = self.round_trips_per_integration * self.t_phys_round_trip / self.norm_t
        
        #this bool enables cubic interpolation on the temporal plot
        #doesn't seem that useful but the wiggles don't seem to be artifacts if num modes cranked in comparison
        self.temporal_interpol = False 

        '''Dispersion file loading'''

        self._load_dispersion_with_fallback()


        '''Avoided mode crossing (AMX)'''

        self.AMX_strength = 0
        self.AMX_loc = 72.5
        self.AMX_epsilon = 1e-6
        self.AMX_Lorentzian_width = 1.0

        self.AMX = -self.AMX_strength / (((self.mu - self.AMX_loc) + self.AMX_epsilon)**2 + self.AMX_Lorentzian_width**2)
        self.dint = self.dint + self.AMX

        '''Normalized integrated dispersion'''

        self.dint_norm = 2 * self.dint / self.kappa_avg
        self.kappa_all = np.ones(self.number_modes) * self.kappa_avg

        '''Derived physical coefficients'''

        self.ng0 = self.c / (self.fsr * self.L)

        # self.Aeff = self.n2 * (2 * np.pi * self.frq_pump) / (self.c * self.gamma_v)  # alt definition

        self.veff = self.Aeff * self.L
        self.g = self.hbar * (2 * np.pi * self.frq_pump)**2 * self.c * self.n2 / (self.ng0**2 * self.veff)

        self.E_amp_2_norm_factor = np.sqrt(self.kappa_avg / (2 * self.g))
        # Keep the physical-power GUI conversion finite at eta=0. The actual
        # cavity drive is independently forced to zero in step() below.
        eta_for_power_scale = self.eta if self.eta > 0.0 else 1.0
        self.P_in_norm_factor = (
            self.hbar * self.ome_pump * self.kappa_avg**2
        ) / (8 * self.g * eta_for_power_scale)

        '''Choice of physical input or normalized input'''

        if self.P_in_phys is not None:
            self.P_in_norm = [self.P_in_phys / self.P_in_norm_factor]
            print(f"Using physical input power: {self.P_in_phys*1e6:.1f} µW (P_in_norm = {self.P_in_norm[0]:.3f})")
        else:
            self.P_in_norm = self.P_in_norm_default
            print(f"Using default normalized input power: {self.P_in_norm[0]}")

        self.noise_amp_base = np.sqrt(1.0 / (2.0 * self.tal_step)) / self.E_amp_2_norm_factor
        self._update_noise_amplitudes()

        # ---- GUI version: pick one input power value (no loop) ----
        self.P_norm = float(self.P_in_norm[0])     # canonical variable
        self.P_norm_target = self.P_norm
        
        self.S = np.sqrt(self.P_norm)
        self.S_target = self.S

        # Detuning range is still useful for slider limits / labels
        self.detuning_norm_sweep_start = self.Detuning_normalized_start
        self.detuning_norm_sweep_stop = self.Detuning_normalized_stop

        # These were batch-sweep bookkeeping; keep if you still want them for reference
        self.t_norm_total = round(
            self.detuning_sweep_rate * (-self.detuning_norm_sweep_start + self.detuning_norm_sweep_stop) / self.save_step_point
        ) * self.save_step_point
        self.t_phys_total = (2 / self.kappa_avg) * self.t_norm_total
        self.N_iter = int(self.t_norm_total / self.round_trips_per_integration)

        self.delta_detuning_norm_int_step = (self.detuning_norm_sweep_stop - self.detuning_norm_sweep_start) / max(self.N_iter, 1)

        # In GUI, DV is controlled by slider:
        self.DV = self.detuning_norm_sweep_start
        self.target_detuning = self.DV
        #st.detuning_slew = 0.5 / steps_per_frame
        # Slew rate in DV per nanosecond (DV/ns)
        self.detuning_slew_rate = float(self.detuning_sweep_rate)
        self.dt_phys_s  = (2.0 / self.kappa_avg) * self.tal_step
        self.dt_phys_ns = self.dt_phys_s * 1e9

        '''Input field (CW by default, optional single file with time + frequency sections)'''

        # Keep the pump source selected by the active startup preset.
        self.input_field_file = str(getattr(self, "input_field_file", "default_cw_1550"))
        self.pump_authoritative_domain = "time"
        self.pump_normalization = "mean_time_power_1"
        self.using_custom_pump_profile = False
        self.pump_profile_t = np.ones(self.number_modes, dtype=complex)
        self.pump_profile_mu = np.zeros(self.number_modes, dtype=complex)

        eo_startup_enabled = bool(getattr(self, "eo_enabled", False))
        self._load_user_or_default_pump(self.input_field_file)
        if eo_startup_enabled:
            # _load_user_or_default_pump intentionally clears EO mode for normal
            # pump loads, so explicitly restore the preset-generated EO field.
            self.eo_enabled = True
            self.apply_eo_comb_pump(enabled=True, rebuild=False)
        self._initialize_startup_fields()


        '''Fast time axis'''

        self.tau_phys = np.linspace(-self.t_phys_round_trip / 2, self.t_phys_round_trip / 2, (self.number_modes))
        self.tau_ps = self.tau_phys * 1e12
        self.tau_fs = self.tau_phys * 1e15
        self.tau = self.tau_phys

        self.j = 0
        self.pump_idx = np.where(self.mu == 0)[0][0]

        # Optional convenience (so you can use kappa_all/kappa_avg in one array)
        self.kappa_all_over_kappa_avg = self.kappa_all / self.kappa_avg

    def set_targets(self, detuning=None, slew=None, P_norm=None, power_slew=None):
        if detuning is not None:
            self.target_detuning = float(detuning)
            
        if slew is not None:
            self.detuning_slew_rate = max(float(slew), 0.0)

        if power_slew is not None:
            self.power_slew_rate = max(float(power_slew), 0.0)
    
        if P_norm is not None:
            self.P_norm_target = max(float(P_norm), 0.0)
            self.S_target = np.sqrt(self.P_norm_target)

    def _update_noise_amplitudes(self):
        """Update pump/cavity noise amplitudes from the physical base value and user multiplier."""
        base = float(getattr(self, "noise_amp_base", getattr(self, "noise_amp", 0.0)))
        level = max(float(getattr(self, "noise_level", 1.0)), 0.0)
        self.noise_amp_base = base
        self.noise_level = level
        self.noise_amp = level * base
        self.pump_noise_amp = self.noise_amp
        self.sig_noise_amp = self.noise_amp

    def set_noise_level(self, level, refresh_pump=True):
        """Set the user-facing noise amplitude multiplier. 1.0 keeps the default physics."""
        self.noise_level = max(float(level), 0.0)
        self._update_noise_amplitudes()

        # Rebuild the driven pump immediately so pump noise changes are visible
        # without requiring a full reset. The intracavity field is left untouched.
        if refresh_pump and hasattr(self, "fE_in_o"):
            if bool(getattr(self, "pump_refresh_noise_enabled", True)):
                self.fE_in = self.fE_in_o + self._complex_noise(
                    self.pump_noise_amp, self.number_modes, kind="pump"
                )
            else:
                self.fE_in = self.fE_in_o.copy()
            self.input_pump_half_step = self.fE_in * (self.tal_step / 2)

    def _normalize_time_profile(self, profile_t):
        profile_t = np.asarray(profile_t, dtype=complex)
        if profile_t.size != self.number_modes:
            raise ValueError(f"Pump time profile length mismatch: expected {self.number_modes}, got {profile_t.size}")
        norm = np.sqrt(np.mean(np.abs(profile_t) ** 2))
        if norm == 0:
            raise ValueError("Pump time profile is all zeros.")
        return profile_t / norm

    def _normalize_freq_profile(self, profile_mu):
        profile_mu = np.asarray(profile_mu, dtype=complex)
        if profile_mu.size != self.number_modes:
            raise ValueError(f"Pump spectral profile length mismatch: expected {self.number_modes}, got {profile_mu.size}")
        profile_t = ifft(fftshift(profile_mu))
        profile_t = self._normalize_time_profile(profile_t)
        profile_mu = fftshift(fft(profile_t))
        return profile_mu

    def _sync_pump_profiles_from_time(self, profile_t):
        self.pump_profile_t = self._normalize_time_profile(profile_t)
        self.pump_profile_mu = fftshift(fft(self.pump_profile_t))

    def _sync_pump_profiles_from_freq(self, profile_mu):
        self.pump_profile_mu = self._normalize_freq_profile(profile_mu)
        self.pump_profile_t = ifft(fftshift(self.pump_profile_mu))

    def _load_pump_file(self, filepath):
        meta = {}
        time_rows = []
        freq_rows = []
        section = None

        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    body = line[1:].strip()
                    if ":" in body:
                        key, val = body.split(":", 1)
                        meta[key.strip().upper()] = val.strip()
                    continue

                upper_line = line.upper()
                if upper_line == "[TIME]":
                    section = "time"
                    continue
                if upper_line == "[FREQ]":
                    section = "freq"
                    continue

                if section is None:
                    raise ValueError(f"Pump file parse error at line {line_no}: data found before a [TIME] or [FREQ] section.")

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    raise ValueError(f"Pump file parse error at line {line_no}: expected at least 3 comma-separated values.")

                idx = int(parts[0])
                value = float(parts[1]) + 1j * float(parts[2])

                if section == "time":
                    time_rows.append((idx, value))
                else:
                    freq_rows.append((idx, value))

        auth = meta.get("AUTHORITATIVE_DOMAIN", "time").strip().lower()
        if auth not in ("time", "frequency", "freq"):
            raise ValueError(f"Unsupported AUTHORITATIVE_DOMAIN: {auth}")
        self.pump_authoritative_domain = "frequency" if auth == "freq" else auth
        self.pump_normalization = meta.get("NORMALIZATION", "mean_time_power_1")

        time_vec = None
        freq_vec = None

        if time_rows:
            time_rows = sorted(time_rows, key=lambda x: x[0])
            time_indices = [idx for idx, _ in time_rows]
            if time_indices != list(range(self.number_modes)):
                raise ValueError(
                    f"TIME section indices must run from 0 to {self.number_modes-1}. Got [{time_indices[0]} ... {time_indices[-1]}]"
                )
            time_vec = np.array([val for _, val in time_rows], dtype=complex)

        if freq_rows:
            freq_rows = sorted(freq_rows, key=lambda x: x[0])
            freq_indices = [idx for idx, _ in freq_rows]
            expected_mu = list(self.mu)
            if freq_indices != expected_mu:
                raise ValueError(
                    f"FREQ section indices must match simulation mu grid from {expected_mu[0]} to {expected_mu[-1]}."
                )
            freq_vec = np.array([val for _, val in freq_rows], dtype=complex)

        if self.pump_authoritative_domain == "time":
            if time_vec is None:
                raise ValueError("Pump file says AUTHORITATIVE_DOMAIN: time but no [TIME] section was found.")
            self._sync_pump_profiles_from_time(time_vec)
        else:
            if freq_vec is None:
                raise ValueError("Pump file says AUTHORITATIVE_DOMAIN: frequency but no [FREQ] section was found.")
            self._sync_pump_profiles_from_freq(freq_vec)

        self.using_custom_pump_profile = True

        if time_vec is not None and freq_vec is not None:
            time_norm = self._normalize_time_profile(time_vec)
            freq_norm = self._normalize_freq_profile(freq_vec)
            freq_from_time = fftshift(fft(time_norm))
            time_from_freq = ifft(fftshift(freq_norm))
            err_freq = np.linalg.norm(freq_from_time - freq_norm) / max(np.linalg.norm(freq_norm), 1e-15)
            err_time = np.linalg.norm(time_from_freq - time_norm) / max(np.linalg.norm(time_norm), 1e-15)
            print(f"Loaded custom pump file '{filepath}' (authoritative domain: {self.pump_authoritative_domain}; err_time={err_time:.3e}, err_freq={err_freq:.3e})")
        else:
            print(f"Loaded custom pump file '{filepath}' (authoritative domain: {self.pump_authoritative_domain})")

    def _rebuild_input_pump(self, add_startup_noise=None):
        amp_scale = float(getattr(self, "input_power_amplitude_scale", 1.0))

        if self.pump_authoritative_domain == "time":
            self.tE_in = self.S * amp_scale * self.pump_profile_t
            self.E_in = fft(self.tE_in)
            self.fE_in_o = fftshift(self.E_in)
        elif self.pump_authoritative_domain == "frequency":
            self.fE_in_o = self.S * amp_scale * self.pump_profile_mu
            self.E_in = fftshift(self.fE_in_o)
            self.tE_in = ifft(self.E_in)
        else:
            raise ValueError(f"Unknown pump_authoritative_domain: {self.pump_authoritative_domain}")

        if add_startup_noise is None:
            add_startup_noise = bool(getattr(self, "startup_pump_noise_enabled", True))

        if add_startup_noise:
            self.fE_in = self.fE_in_o + self._complex_noise(
                self.pump_noise_amp, self.number_modes, kind="pump"
            )
        else:
            self.fE_in = self.fE_in_o.copy()

        self.input_pump_half_step = self.fE_in * (self.tal_step / 2)
        
    def _rf_power_dBm_to_vpeak(self, power_dBm, impedance_ohm=None):
        """Convert RF power in dBm into RF peak voltage for a matched sinusoidal 50-ohm drive."""
        R = float(self.eo_rf_impedance_ohm if impedance_ohm is None else impedance_ohm)
        P_W = 1e-3 * 10.0 ** (float(power_dBm) / 10.0)
        V_rms = np.sqrt(max(P_W, 0.0) * R)
        return np.sqrt(2.0) * V_rms

    def _phase_modulation_index_from_rf(self, power_dBm, vpi_V, impedance_ohm=None):
        """beta = pi*V_peak/Vpi. This assumes the Vpi convention is for phase swing of pi."""
        vpi = max(float(vpi_V), 1e-15)
        Vpk = self._rf_power_dBm_to_vpeak(power_dBm, impedance_ohm=impedance_ohm)
        return np.pi * Vpk / vpi

    def build_eo_comb_profile(self, cfg=None):
        """
        Build a NEOS-like EO-comb pump profile in fast time.

        The profile is normalized by _sync_pump_profiles_from_time(), while
        PM/IM insertion losses are retained through input_power_amplitude_scale.
        """
        if cfg is None:
            cfg = self.get_eo_config()

        N = int(self.number_modes)
        n = np.arange(N, dtype=float)
        theta_rf = 2.0 * np.pi * float(cfg.get("rf_spacing_mu", 1.0)) * n / float(N)

        beta_pm_single = self._phase_modulation_index_from_rf(
            cfg.get("pm_rf_power_dBm", 28.0),
            cfg.get("pm_vpi_V", 3.5),
            impedance_ohm=cfg.get("rf_impedance_ohm", 50.0),
        )
        beta_pm_total = int(cfg.get("num_pm", 1)) * beta_pm_single

        field = np.exp(1j * beta_pm_total * np.sin(theta_rf + float(cfg.get("pm_phase_rad", 0.0))))

        beta_im = 0.0
        if bool(cfg.get("im_enabled", False)):
            beta_im = self._phase_modulation_index_from_rf(
                cfg.get("im_rf_power_dBm", 25.0),
                cfg.get("im_vpi_V", 4.0),
                impedance_ohm=cfg.get("rf_impedance_ohm", 50.0),
            )
            # First-version Mach-Zehnder IM model. The bias is expressed in pi-radians:
            # im_bias=0.5 means quadrature bias, im_bias=0 means maximum transmission.
            im_bias_phase = np.pi * float(cfg.get("im_bias", 0.5))
            im_drive = beta_im * np.sin(theta_rf + float(cfg.get("im_phase_rad", 0.0)))
            field = field * np.cos(0.5 * (im_bias_phase + im_drive))

        total_loss_dB = max(float(cfg.get("pm_loss_dB", 0.0)), 0.0) + (
            max(float(cfg.get("im_loss_dB", 0.0)), 0.0) if bool(cfg.get("im_enabled", False)) else 0.0
        )
        transmission = 10.0 ** (-total_loss_dB / 10.0)

        self.eo_last_beta_pm = float(beta_pm_total)
        self.eo_last_beta_im = float(beta_im)
        self.eo_last_transmission = float(transmission)

        return field.astype(complex), transmission

    def get_eo_config(self):
        """Return current EO-comb settings as a simple dictionary."""
        return dict(
            enabled=bool(getattr(self, "eo_enabled", False)),
            rf_spacing_mu=float(getattr(self, "eo_rf_spacing_mu", 1.0)),
            pm_rf_power_dBm=float(getattr(self, "eo_pm_rf_power_dBm", 28.0)),
            pm_vpi_V=float(getattr(self, "eo_pm_vpi_V", 3.5)),
            num_pm=int(getattr(self, "eo_num_pm", 1)),
            pm_loss_dB=float(getattr(self, "eo_pm_loss_dB", 0.0)),
            pm_phase_rad=float(getattr(self, "eo_pm_phase_rad", 0.0)),
            im_enabled=bool(getattr(self, "eo_im_enabled", False)),
            im_rf_power_dBm=float(getattr(self, "eo_im_rf_power_dBm", 25.0)),
            im_vpi_V=float(getattr(self, "eo_im_vpi_V", 4.0)),
            im_bias=float(getattr(self, "eo_im_bias", 0.5)),
            im_loss_dB=float(getattr(self, "eo_im_loss_dB", 0.0)),
            im_phase_rad=float(getattr(self, "eo_im_phase_rad", 0.0)),
            rf_impedance_ohm=float(getattr(self, "eo_rf_impedance_ohm", 50.0)),
        )

    def set_eo_config(self, **cfg):
        """Update EO-comb settings on the state object."""
        self.eo_enabled = bool(cfg.get("enabled", getattr(self, "eo_enabled", False)))
        self.eo_rf_spacing_mu = float(cfg.get("rf_spacing_mu", getattr(self, "eo_rf_spacing_mu", 1.0)))
        self.eo_pm_rf_power_dBm = float(cfg.get("pm_rf_power_dBm", getattr(self, "eo_pm_rf_power_dBm", 28.0)))
        self.eo_pm_vpi_V = float(cfg.get("pm_vpi_V", getattr(self, "eo_pm_vpi_V", 3.5)))
        self.eo_num_pm = max(int(cfg.get("num_pm", getattr(self, "eo_num_pm", 1))), 0)
        self.eo_pm_loss_dB = max(float(cfg.get("pm_loss_dB", getattr(self, "eo_pm_loss_dB", 0.0))), 0.0)
        self.eo_pm_phase_rad = float(cfg.get("pm_phase_rad", getattr(self, "eo_pm_phase_rad", 0.0)))
        self.eo_im_enabled = bool(cfg.get("im_enabled", getattr(self, "eo_im_enabled", False)))
        self.eo_im_rf_power_dBm = float(cfg.get("im_rf_power_dBm", getattr(self, "eo_im_rf_power_dBm", 25.0)))
        self.eo_im_vpi_V = float(cfg.get("im_vpi_V", getattr(self, "eo_im_vpi_V", 4.0)))
        self.eo_im_bias = float(cfg.get("im_bias", getattr(self, "eo_im_bias", 0.5)))
        self.eo_im_loss_dB = max(float(cfg.get("im_loss_dB", getattr(self, "eo_im_loss_dB", 0.0))), 0.0)
        self.eo_im_phase_rad = float(cfg.get("im_phase_rad", getattr(self, "eo_im_phase_rad", 0.0)))
        self.eo_rf_impedance_ohm = max(float(cfg.get("rf_impedance_ohm", getattr(self, "eo_rf_impedance_ohm", 50.0))), 1e-9)

    def apply_eo_comb_pump(self, enabled=True, rebuild=True):
        """
        Enable/disable the generated EO-comb pump and rebuild the injected field.

        This changes only the pump driving field. It does not reset the intracavity
        field unless the caller separately runs reinitialize().
        """
        self.eo_enabled = bool(enabled)

        if not self.eo_enabled:
            self.input_power_amplitude_scale = 1.0
            restore_source = getattr(self, "eo_previous_pump_source", "default_cw_1550")
            if not restore_source or restore_source == "generated_eo_comb":
                restore_source = "default_cw_1550"
            self._load_user_or_default_pump(restore_source)
            if rebuild:
                self._rebuild_input_pump(add_startup_noise=False)
            print(f'[EO comb] Disabled; restored pump source "{self.input_field_file}".')
            return

        # The laser source remains a CW pump. EO modulation is a separate
        # processing stage and must not replace the pump-source label.
        current_source = getattr(self, "input_field_file", "default_cw_1550")
        self.eo_previous_pump_source = current_source

        profile_t, transmission = self.build_eo_comb_profile()
        self.pump_authoritative_domain = "time"
        self.pump_normalization = "mean_time_power_1"
        self.using_custom_pump_profile = False
        self.eo_output_field_label = "EO-modulated CW field"
        self.input_power_amplitude_scale = np.sqrt(max(float(transmission), 0.0))
        self._sync_pump_profiles_from_time(profile_t)
        if rebuild:
            self._rebuild_input_pump(add_startup_noise=False)

        print(
            "[EO comb] Enabled: "
            f"spacing={self.eo_rf_spacing_mu:g} modes, "
            f"beta_PM={self.eo_last_beta_pm:.3g}, "
            f"IM={'on' if self.eo_im_enabled else 'off'}, "
            f"transmission={100*self.eo_last_transmission:.2f}%"
        )

    def _load_embedded_default_pump(self):
        """
        Embedded default pump:
        CW laser at central mode (1550 nm operating point).
        Flat in fast-time domain.
        """
        self._load_embedded_preset_pump("default_cw_1550")

    def _load_embedded_preset_pump(self, preset_name):
        """Load one of the named built-in pump presets."""
        profile_t = build_embedded_pump_profile(preset_name, self.number_modes)
        self.pump_authoritative_domain = "time"
        self.pump_normalization = "mean_time_power_1"
        self.using_custom_pump_profile = False
        self._sync_pump_profiles_from_time(profile_t)
        print(f"[Pump] Using embedded pump preset: {preset_name}")

    def _load_user_or_default_pump(self, pump_source):
        """
        Accept either:
          - a user filename
          - a built-in preset name
          - empty / missing -> embedded default CW
        """
        pump_source = (pump_source or "").strip()

        # Normal pump presets/files use the GUI pump power directly.
        # The EO-comb generator overrides this to include PM/IM insertion loss.
        self.eo_enabled = False
        self.input_power_amplitude_scale = 1.0

        if not pump_source:
            self.input_field_file = "default_cw_1550"
            self._load_embedded_default_pump()
            return

        if pump_source in EMBEDDED_PUMP_PRESET_NAMES:
            self.input_field_file = pump_source
            self._load_embedded_preset_pump(pump_source)
            return

        if os.path.isfile(pump_source):
            self.input_field_file = pump_source
            self._load_pump_file(pump_source)
            return

        self.input_field_file = "default_cw_1550"
        print(f'[Pump] Pump file not found: "{pump_source}" -> using embedded default CW pump.')
        self._load_embedded_default_pump()
        
            
    def _initialize_startup_fields(self):
        """
        Initialize pump/intracavity startup according to selected preset style.

        Styles:
          - "cw_only": deterministic pump, empty cavity (V7-like)
          - "noisy": configurable startup pump noise + configurable startup cavity noise
        """
        style = str(getattr(self, "startup_style", "noisy")).strip().lower()

        if style == "cw_only":
            self._rebuild_input_pump(add_startup_noise=False)
            self.E_t_fast_norm = np.zeros(self.number_modes, dtype=complex)
        else:
            self._rebuild_input_pump(
                add_startup_noise=bool(getattr(self, "startup_pump_noise_enabled", True))
            )

            if bool(getattr(self, "startup_cavity_noise_enabled", True)):
                self.E_t_fast_norm = self._complex_noise(
                    self.sig_noise_amp, self.number_modes, kind="cavity"
                )
            else:
                self.E_t_fast_norm = np.zeros(self.number_modes, dtype=complex)

        self.spectrum_E_t_fast_norm = fftshift(fft(self.E_t_fast_norm))
        self.input_pump_half_step = self.fE_in * (self.tal_step / 2)

    def _complex_noise(self, amp, size, kind="cavity"):
        # master switch overrides everything
        if not self.noise_switch:
            return np.zeros(size, dtype=complex)
    
        if kind == "pump" and not self.pump_noise_enabled:
            return np.zeros(size, dtype=complex)
    
        if kind == "cavity" and not self.cavity_noise_enabled:
            return np.zeros(size, dtype=complex)
    
        return amp * (np.random.randn(size) + 1j*np.random.randn(size))


    def step(self, n_steps=100):
        for _ in range(n_steps):
            self.j += 1

            # --- physical time per internal step ---
            dt_phys = (2.0 / self.kappa_avg) * self.tal_step   # seconds
            dt_phys_ns = dt_phys * 1e9                         # ns

            # --- slew-limited detuning update ---
            max_DV_step = self.detuning_slew_rate * dt_phys_ns
            d = self.target_detuning - self.DV
            self.DV += np.clip(d, -max_DV_step, max_DV_step)

            # --- slew-limited power update: preserve custom pump shape; only rescale amplitude ---
            if self.P_norm_target != self.P_norm:
                max_P_step = self.power_slew_rate * dt_phys_ns
                dP = self.P_norm_target - self.P_norm
                if max_P_step <= 0:
                    P_next = self.P_norm
                else:
                    P_next = self.P_norm + np.clip(dP, -max_P_step, max_P_step)

                if P_next != self.P_norm:
                    self.P_norm = max(float(P_next), 0.0)
                    self.S = np.sqrt(self.P_norm)
                    self.S_target = np.sqrt(max(self.P_norm_target, 0.0))
                    self._rebuild_input_pump()

            # --- refresh pump noise occasionally, like reference code ---
            if self.j % self.save_step_point == 0:
                if bool(getattr(self, "pump_refresh_noise_enabled", True)):
                    self.fE_in = self.fE_in_o + self._complex_noise(
                        self.pump_noise_amp, self.number_modes, kind="pump"
                    )
                else:
                    self.fE_in = self.fE_in_o.copy()

                self.input_pump_half_step = self.fE_in * (self.tal_step / 2)

            # --- propagation constants ---
            L = -(self.kappa_all / self.kappa_avg) - 1j * self.DV - 1j * self.dint_norm
            dt = self.tal_step

            # --- Drive per unit time, integrated exactly over each linear half step ---
            # This matches the v7 physics path better than the older:
            #     exp(L*h) * (A + fE_in*h)
            # approximation.
            # Zero external coupling means that the bus field cannot drive the
            # resonator. The bus/through-port EO field remains available to the
            # output calculation, but the LLE cavity forcing is exactly zero.
            F = self.fE_in if self.kappa_ex > 0.0 else np.zeros_like(self.fE_in)
            h = dt / 2.0
            x = L * h
            expL = np.exp(x)
            eps = 1e-14
            phi = np.where(np.abs(L) > eps, np.expm1(x) / L, h)

            # first linear half-step with exact pump integration
            A = self.spectrum_E_t_fast_norm
            self.spectrum_E_t_fast_norm = A * expL + F * phi

            # nonlinear step
            self.E_t_fast_norm = ifft(fftshift(self.spectrum_E_t_fast_norm))
            self.E_t_fast_norm *= np.exp(1j * (np.abs(self.E_t_fast_norm) ** 2) * dt)

            # second linear half-step with exact pump integration
            self.spectrum_E_t_fast_norm = fftshift(fft(self.E_t_fast_norm))
            A = self.spectrum_E_t_fast_norm
            self.spectrum_E_t_fast_norm = A * expL + F * phi

    def get_spectrum_dbm_like(self):
        P = np.abs(self.spectrum_E_t_fast_norm)**2
        P = np.maximum(P, 1e-30)  # safer floor
        return 10*np.log10(P)
    
    def get_output_spectrum_dBm(self):
        """
        OSA-like spectrum at the through port (bus waveguide), per mode, in dBm.
    
        Uses:
          - A_mu_norm from current spectrum_E_t_fast_norm
          - a_mu_phys = E_amp_2_norm_factor * A_mu_norm
          - s_out = s_in - sqrt(kappa_ex) * a_mu_phys
          - P_out = hbar * omega_mu * |s_out|^2  [W]
        """
        # Convert normalized spectral field -> normalized mode amplitudes.
        # NOTE: your FFT conventions vary; this /number_modes matches your older batch code.
        A_mu_norm = self.spectrum_E_t_fast_norm / self.number_modes
    
        # Physical mode amplitudes
        a_mu_phys = self.E_amp_2_norm_factor * A_mu_norm
    
        # Build the complete physical bus input, including every EO sideband.
        # pump_profile_mu/N has unit total spectral power by Parseval because
        # mean(|pump_profile_t|^2)=1.
        P_in_phys_W = (
            float(self.P_in_phys) if self.P_in_phys is not None
            else float(self.P_norm * self.P_in_norm_factor)
        )
        weights = np.asarray(self.pump_profile_mu, dtype=complex) / float(self.number_modes)
        weights *= float(getattr(self, "input_power_amplitude_scale", 1.0))
        s_in = np.sqrt(max(P_in_phys_W, 0.0)) * weights / np.sqrt(self.hbar * self.ome_grid)
    
        # Through port field
        s_out = s_in - np.sqrt(self.kappa_ex) * a_mu_phys
    
        # Output power per mode [W]
        P_out_W = self.hbar * self.ome_grid * (np.abs(s_out) ** 2)
    
        # Convert to dBm safely
        P_out_W = np.maximum(P_out_W, 1e-30)
        P_out_dBm = 10.0 * np.log10(P_out_W / 1e-3)
    
        return P_out_dBm


    def get_intracavity_mean(self):
        return float(np.mean(np.abs(self.E_t_fast_norm)**2))
    
    def get_intracavity_mean_W(self):
        """
        Approx physical mean intracavity power in W.
        Uses the same normalization logic as your per-mode formula:
          a = E_amp_2_norm_factor * A
          P ≈ sum(hbar*omega*|a|^2) / N
        """
        A = self.E_t_fast_norm                       # normalized field vs fast time
        a = self.E_amp_2_norm_factor * A             # physical amplitude (per your normalization)
        P_inst = self.hbar * self.ome_pump * np.abs(a)**2  # W-like, using pump omega as approx
        return float(np.mean(P_inst))
    
    def reinitialize(self, new_n_modes=None):
        """Recalculates physical constants and resets the simulation state."""
        if new_n_modes is not None:
            self.number_modes = int(new_n_modes)
            self.mu = np.arange(-self.number_modes // 2, self.number_modes // 2)
            self.kappa_all = np.ones(self.number_modes) * self.kappa_avg
        
        # --- REBUILD TIME NORMALIZATION (missing right now) ---
        # Make sure pump frequency is consistent with current wavelength
        self.frq_pump = self.c / self.wvl_pump
        self.ome_pump = 2 * np.pi * self.frq_pump
        
        self.t_phys_round_trip = 1.0 / self.fsr
        self.linewidth = self.frq_pump / self.Q
        self.kappa_avg = self.linewidth * 2 * np.pi
        self.kappa_ex  = self.eta * self.kappa_avg
        
        # normalized time definitions (same as __init__)
        self.alpha  = self.t_phys_round_trip * self.kappa_avg / 2.0
        self.norm_t = 1.0 / (self.alpha / self.t_phys_round_trip)
        
        self.round_trips_per_integration = 1
        self.tal_step = self.round_trips_per_integration * self.t_phys_round_trip / self.norm_t
        
        # update dt used by DV/ns logic
        self.dt_phys_s  = (2.0 / self.kappa_avg) * self.tal_step
        self.dt_phys_ns = self.dt_phys_s * 1e9

        
        # Recalculate Nonlinearity g and normalization factors
        self.L = self.R_res * np.pi * 2
        self.veff = self.Aeff * self.L
        self.ng0 = self.c / (self.fsr * self.L)
        self.g = self.hbar * (2 * np.pi * self.frq_pump)**2 * self.c * self.n2 / (self.ng0**2 * self.veff)
        self.E_amp_2_norm_factor = np.sqrt(self.kappa_avg / (2 * self.g))
        eta_for_power_scale = self.eta if self.eta > 0.0 else 1.0
        self.P_in_norm_factor = (
            self.hbar * self.ome_pump * self.kappa_avg**2
        ) / (8 * self.g * eta_for_power_scale)

        self.noise_amp_base = np.sqrt(1.0 / (2.0 * self.tal_step)) / self.E_amp_2_norm_factor
        self._update_noise_amplitudes()

        # --- KEEP PHYSICAL INPUT POWER CONSTANT WHEN WAVELENGTH CHANGES ---
        if self.P_in_phys is not None:
            self.P_in_norm = [self.P_in_phys / self.P_in_norm_factor]
        else:
            self.P_in_norm = self.P_in_norm_default

        # canonical GUI variables
        self.P_norm = float(self.P_in_norm[0])
        self.P_norm_target = self.P_norm
        self.power_slew_rate = float(getattr(self, "power_slew_rate", 100.0))
        self.S = np.sqrt(self.P_norm)
        self.S_target = self.S

        # restore startup operating point as the live runtime state
        self.detuning_slew_rate = float(getattr(self, "detuning_sweep_rate", self.detuning_slew_rate))
        self.DV = float(getattr(self, "Detuning_normalized_start", getattr(self, "DV", 0.0)))
        self.target_detuning = self.DV

        # Update grids
        self.frq_grid = self.frq_pump + self.mu * self.fsr
        self.ome_grid = self.frq_grid * (2 * np.pi)
        self.wvl_grid = self.c / self.frq_grid
        
        self._load_dispersion_with_fallback()
        self.dint_norm = 2 * self.dint / self.kappa_avg


        self.dint_norm = 2 * self.dint / self.kappa_avg
        
        # Reset state
        self.j = 0

        self.tau_phys = np.linspace(-self.t_phys_round_trip/2, self.t_phys_round_trip/2, self.number_modes)
        self.tau_ps = self.tau_phys * 1e12

        self.pump_idx = np.where(self.mu == 0)[0][0]

        # Rebuild pump profile from current source, then initialize startup fields.
        # Preserve a generated EO-comb pump across Apply reset.
        eo_was_enabled = bool(getattr(self, "eo_enabled", False))
        eo_cfg = self.get_eo_config() if hasattr(self, "get_eo_config") else {}
        pump_source = getattr(self, "input_field_file", "default_cw_1550")
        self._load_user_or_default_pump(pump_source)
        if eo_was_enabled:
            self.set_eo_config(**eo_cfg)
            self.apply_eo_comb_pump(enabled=True, rebuild=False)
        self._initialize_startup_fields()

        self.kappa_all = np.ones(self.number_modes) * self.kappa_avg
        self.kappa_all_over_kappa_avg = self.kappa_all / self.kappa_avg

    def _load_embedded_dint_preset(self, preset_name):
        if preset_name not in EMBEDDED_DINT_PRESET_NAMES:
            raise ValueError(
                f"Unknown embedded Dint preset: {preset_name}. "
                f"Available presets: {', '.join(EMBEDDED_DINT_PRESET_NAMES)}"
            )

        if preset_name == "default_dint_zero_flat":
            self.dint = np.zeros(self.number_modes, dtype=float)
            print('[Dispersion] Using embedded flat-zero Dint preset.')
            return

        dint_csv = EMBEDDED_DINT_PRESETS[preset_name]
        dint_data = np.genfromtxt(io.StringIO(dint_csv), delimiter=',', skip_header=1)
        if dint_data.ndim == 1:
            dint_data = dint_data.reshape(1, -1)
        self._build_dint_from_table(dint_data)
        print(f'[Dispersion] Using embedded Dint preset: {preset_name}')

    def _load_user_or_default_dint(self, dint_source):
        source = (dint_source or '').strip()

        if source in EMBEDDED_DINT_PRESET_NAMES:
            self._load_embedded_dint_preset(source)
            return

        if source:
            candidates = [source]
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                candidates.append(os.path.join(base_dir, source))
            except NameError:
                pass

            file_found = next((p for p in candidates if os.path.isfile(p)), None)
            if file_found:
                dint_data = np.loadtxt(file_found, skiprows=1, delimiter=',')
                self._build_dint_from_table(dint_data)
                print(f'Using Dint file: "{file_found}"')
                return

            print(f'[Dispersion] Dint source not found: "{source}" -> using embedded default preset.')
        else:
            print('[Dispersion] Empty Dint source -> using embedded default preset.')

        self._load_embedded_dint_preset('default_dint_pasquazi2018_MgF2')

    def _load_dispersion_with_fallback(self):
        """
        Sets:
          self.dint, self.d2, and prints what was used.
        Accepts either:
          - an embedded preset name in the Dint text box
          - a file path
          - blank / missing source, which falls back to the embedded Herr MgF2 default
        """
        try:
            self._load_user_or_default_dint(self.dint_file_path)
            return
        except Exception as e:
            print(f"[Dispersion] Embedded/file Dint load failed ({e!r}) -> using constant d2_default.")

        self.d2 = self.d2_default * np.ones_like(self.mu)
        self.dint = 0.5 * self.d2 * self.mu**2
        print(f"[Dispersion] Using constant d2_default = {self.d2_default}")

    def _build_dint_from_table(self, dint_data):
        """
        dint_data columns expected:
          col0: wavelength [um]
          col1: frequency [THz]
          col2: (dint/2pi) [GHz]
        """
        self.dint_wvl = dint_data[:, 0] * 1e-6
        self.dint_frq = dint_data[:, 1] * 1e12
        self.dint_ome = self.dint_frq * (2 * np.pi)
    
        self.dint_vals = dint_data[:, 2] * 1e9              # (dint/2pi) in Hz
        self.dint_vals_ome = self.dint_vals * 2 * np.pi      # [rad/s]
    
        dint_interp = interp1d(self.dint_ome, self.dint_vals_ome, kind='cubic', fill_value="extrapolate")
        self.dint = dint_interp(self.ome_grid)               # [rad/s]
    
        # Your existing d2 extraction (keep as-is)
        self.d2_ome = np.zeros_like(self.dint)
        mask = (self.mu != 0)
        self.d2_ome[mask] = self.dint[mask] * 2 / (self.mu[mask]**2)
        self.d2_frq = self.d2_ome / (2 * np.pi)
        self.d2 = self.d2_ome.copy()
        self.d2[self.mu == 0] = 0



def run_gui():
    plt.close("all")
    st = LLEState()
    # Create a persistent container for all interactive elements
    ui = {}
    
    DEFAULTS = dict(
        DV=st.DV,
        Pnorm=st.P_norm,
        slew=st.detuning_slew_rate,
        power_slew=st.power_slew_rate,
    )
    
    # ---- Spectrum Y-axis mode ----
    SPECTRUM_LIVE_AUTOSCALE = False   # True = live autoscale, False = fixed limits
    
    # Fixed limits (only used if autoscale is False)
    SPECTRUM_YMIN = -150
    SPECTRUM_YMAX = 100
    baseline_db = -299
    
    # ---- Detuning limits (single source of truth) ----
    DV_min = -5.0
    DV_max = 15.0
    
    Pnorm_min = 0.0
    Pnorm_max = 20.0

    # ---- Live settings ----
    steps_per_frame = 100 #10 #50          # sim steps per animation frame
    Nkeep = 3000                  # how many power samples to keep (ring buffer)
    interval_ms = 100              # animation interval

    # ---- Figure / responsive dashboard layout ----
    # The old GUI used a 2x2 subplot grid plus absolute widgets at the far left/right.
    # That made different monitor sizes risky because plot labels, side panels and sliders
    # were competing for the same normalized figure space.  This version reserves four
    # explicit zones: left parameter column, central plots, right action column, bottom sliders.
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.canvas.manager.set_window_title("pycombs")
    fig.set_constrained_layout(False)

    # Normalized figure regions.  Keeping every widget inside one of these regions prevents
    # overlaps when the window is resized or opened on a different monitor.
    LCOL = dict(x=0.018, w=0.090, top=0.900, bottom=0.315)
    # Shift the complete 2x2 plot grid left while preserving its total width.
    # This creates more clearance for the multiple right-side power axes.
    PCOL = dict(x0=0.190, x1=0.760, top=0.890, bottom=0.315)
    # Narrower right-side control panel, moved farther right so the
    # intracavity-power axes and their outward-offset labels remain unobstructed.
    RCOL = dict(x=0.892, w=0.090, top=0.900, bottom=0.315)
    SCOL = dict(x=0.185, w=0.675, top=0.230, bottom=0.045)

    gs = fig.add_gridspec(
        2, 2,
        left=PCOL["x0"], right=PCOL["x1"],
        bottom=PCOL["bottom"], top=PCOL["top"],
        width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.0],
        wspace=0.34, hspace=0.44,
    )

    ax_spec  = fig.add_subplot(gs[0, 0])   # spectrum (top-left)
    ax_map   = fig.add_subplot(gs[1, 0])   # spectral evolution map (bottom-left)
    ax_pow   = fig.add_subplot(gs[0, 1])   # intracavity power evolution (top-right)
    ax_pulse = fig.add_subplot(gs[1, 1])   # temporal profile (bottom-right)

    # Central product title. The active literature preset remains visible only
    # in its dedicated field and is not repeated in the application title.
    title_artist = fig.text(
        0.5 * (PCOL["x0"] + PCOL["x1"]), 0.975,
        "pycombs v1.0",
        ha="center", va="top",
        fontsize=26,
        fontweight="bold",
        color="black"
    )

    def _font_scale():
        """Scale fonts gently with the usable figure size, while keeping them readable."""
        w, h = fig.get_size_inches()
        return float(np.clip(min(w / 16.0, h / 9.0), 0.78, 1.18))

    def apply_responsive_fonts(event=None):
        fs = _font_scale()
        title_artist.set_fontsize(13 * fs)
        for ax in (ax_spec, ax_map, ax_pow, ax_pulse):
            ax.title.set_fontsize(12 * fs)
            ax.xaxis.label.set_fontsize(9.0 * fs)
            ax.yaxis.label.set_fontsize(9.0 * fs)
            ax.tick_params(axis="both", labelsize=7.5 * fs)
        # Keep the extra right-side power axes smaller than the plot axes; otherwise
        # their long labels invade the save panel on smaller monitors.
        for _name in ("ax_pow_norm", "ax_pow_R2", "ax_pow_R3"):
            ax_extra = locals().get(_name) or globals().get(_name)
            if ax_extra is not None:
                try:
                    ax_extra.yaxis.label.set_fontsize(6.5 * fs)
                    ax_extra.tick_params(axis="y", labelsize=6.2 * fs, pad=2)
                except Exception:
                    pass
        try:
            ax_spec_top.xaxis.label.set_fontsize(9.0 * fs)
            ax_spec_top.tick_params(axis="x", labelsize=8.0 * fs)
        except Exception:
            pass
        try:
            spectral_cbar.set_label("Spectrum (dB)", fontsize=8.0 * fs)
            spectral_cbar.ax.tick_params(labelsize=7.0 * fs)
        except Exception:
            pass
        fig.canvas.draw_idle()

    # make window open maximized when the backend supports it
    manager = plt.get_current_fig_manager()
    try:
        if hasattr(manager, "window") and hasattr(manager.window, "showMaximized"):
            manager.window.showMaximized()
        elif hasattr(manager, "window") and hasattr(manager.window, "state"):
            manager.window.state("zoomed")
        elif hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()
    except Exception:
        pass


    # ---- Spectrum plot (OSA-like) ----
    
    # --- Frequency axis (uniformly spaced, correct physics) ---
    x_THz = st.frq_grid * 1e-12   # main x-axis

    
    def wvl_nm_to_THz(wvl_nm):
        return st.c / (wvl_nm * 1e-9) * 1e-12
    
    def THz_to_wvl_nm(freq_THz):
        return st.c / (freq_THz * 1e12) * 1e9

    def wvl_um_to_THz(wvl_um):
        wvl_um = np.asarray(wvl_um, dtype=float)
        wvl_um = np.where(np.abs(wvl_um) < 1e-12, np.nan, wvl_um)
        return st.c / (wvl_um * 1e-6) * 1e-12


    def THz_to_wvl_um(freq_THz):
        return st.c / (freq_THz * 1e12) * 1e6

    
    def choose_baseline_db(spec_db, pump_idx, q=10, margin_db=10):
        tmp = np.array(spec_db, float)
        tmp[pump_idx] = np.nan
        floor = np.nanpercentile(tmp, q)
        return float(floor - margin_db)
    
    def apply_minor_ticks(ax):
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="minor", length=3)

    
    # Initial OSA-like spectrum (sorted to match x_nm)
    y_db = st.get_output_spectrum_dBm()
    spec_line, = ax_spec.plot(x_THz, y_db, linewidth=0.0)

    ax_spec.set_title("Pulse spectrum", fontsize=13, fontweight="bold")    
    ax_spec.set_xlabel("Frequency (THz)")
    ax_spec.set_ylabel("Spectrum optical power\n(through-port) [dBm]")
    spec_color = "tab:blue"  # same as stem_sc/stem_lc
    ax_spec.yaxis.label.set_color(spec_color)
    ax_spec.yaxis.label.set_color(spec_color)
    ax_spec.grid(True, alpha=0.25)
    
    # Always match simulated-mode span
    ax_spec.set_xlim(x_THz[0], x_THz[-1])
    apply_minor_ticks(ax_spec)

    
    # Optional fixed Y limits
    if not SPECTRUM_LIVE_AUTOSCALE:
        ax_spec.set_ylim(SPECTRUM_YMIN, SPECTRUM_YMAX)
    
    # Secondary top axis: frequency THz
    ax_spec_top = ax_spec.secondary_xaxis(
        "top",
        functions=(THz_to_wvl_um, wvl_um_to_THz)
    )
    ax_spec_top.set_xlabel("Wavelength (µm)")
    
    ax_spec_top.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_spec_top.minorticks_on()
    ax_spec_top.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_spec_top.tick_params(which="minor", length=3)
    
    # Baseline for stems
    pump_idx = st.pump_idx
    #baseline_db = choose_baseline_db(y_db, pump_idx)
    baseline_state = {"y": baseline_db, "locked": True}
    baseline_line = ax_spec.axhline(baseline_state["y"], color="k", linestyle="--", linewidth=1)
    
    # Stems + dots in WAVELENGTH units (sorted)
    segments = [((x, baseline_state["y"]), (x, y)) for x, y in zip(x_THz, y_db)]
    stem_sc = ax_spec.scatter(x_THz, y_db, c="tab:blue", s=0)
    stem_lc = LineCollection(segments, colors="tab:blue", linewidths=1)
    ax_spec.add_collection(stem_lc)
    
    WINDOW_NS = 10.0  # shared rolling time window for power history and spectral map

    # ---- Temporal pulse plot (INIT ONCE) ----
    t_ps = st.tau_ps
    I_t  = np.abs(st.E_t_fast_norm)**2
    t_pulse = t_ps
    I_t_pulse = I_t
    
    def sync_x_axes_to_grid(st, ax_spec, ax_pulse, ax_spec_top=None, pad_frac=0.0):
        # --- Spectrum axis (THz) ---
        x_THz = st.frq_grid * 1e-12
        xmin, xmax = float(x_THz[0]), float(x_THz[-1])
    
        # optional small padding
        pad = (xmax - xmin) * pad_frac
        ax_spec.set_xlim(xmin - pad, xmax + pad)
    
        # --- Temporal axis (ps) ---
        t_ps = st.tau_ps
        tmin, tmax = float(t_ps[0]), float(t_ps[-1])
        pad_t = (tmax - tmin) * pad_frac
        ax_pulse.set_xlim(tmin - pad_t, tmax + pad_t)
    
        # If you use a secondary wavelength axis, it will follow xlim automatically,
        # but forcing a draw helps refresh tick formatting.
        if ax_spec_top is not None:
            ax_spec_top.figure.canvas.draw_idle()
    
        return x_THz, t_ps


    pulse_line, = ax_pulse.plot(t_pulse, I_t_pulse, linewidth=1.2, color="tab:blue")

    # ---- Spectral-evolution map below the intracavity power plot ----
    # Mode number on x; rolling physical time window on y.
    pulse_view = {"mode": "temporal"}  # kept for compatibility; temporal profile stays visible
    spectral_time_hist_ns = np.full(Nkeep, np.nan, dtype=float)
    spectral_db_hist = np.full((Nkeep, st.number_modes), np.nan, dtype=float)

    mode_axis = np.arange(st.number_modes) - st.pump_idx
    spectral_map_image = ax_map.imshow(
        np.full((2, st.number_modes), np.nan),
        origin="lower",
        aspect="auto",
        extent=[float(mode_axis[0]), float(mode_axis[-1]), 0.0, WINDOW_NS],
        vmin=-80, vmax=0,
        interpolation="nearest",
        cmap="turbo",
        visible=True,
    )
    spectral_map_image.set_animated(True)

    # Colorbar for the spectral-evolution view.
    map_pos = ax_map.get_position()
    ax_spectral_cbar = fig.add_axes([map_pos.x1 + 0.006, map_pos.y0, 0.012, map_pos.height])
    spectral_cbar = fig.colorbar(spectral_map_image, cax=ax_spectral_cbar)
    spectral_cbar.set_label("Spectrum (dB)", fontsize=8)
    spectral_cbar.ax.tick_params(labelsize=7)
    ax_spectral_cbar.set_visible(True)

    def reposition_spectral_colorbar(event=None):
        pos = ax_map.get_position()
        ax_spectral_cbar.set_position([pos.x1 + 0.006, pos.y0, 0.010, pos.height])

    ax_map.set_title("Spectral evolution", fontsize=13, fontweight="bold")
    ax_map.set_xlabel("Mode number")
    ax_map.set_ylabel("Time (ns)")
    ax_map.set_xlim(float(mode_axis[0]), float(mode_axis[-1]))
    ax_map.set_ylim(0.0, WINDOW_NS)
    
    ax_pulse.set_title("Temporal profile", fontsize=13,fontweight="bold")
    ax_pulse.set_xlabel("Fast time τ (ps)")
    ax_pulse.set_ylabel("Intracavity intensity\n(arb. units)")
    pulse_color = "tab:blue"
    ax_pulse.yaxis.label.set_color(pulse_color)
    ax_pulse.yaxis.label.set_color(pulse_color)
    ax_pulse.grid(True, alpha=0.3)
    ax_pulse.set_ylim(0, 25)
    apply_minor_ticks(ax_pulse)
    
    # Sync spectrum + temporal x axes to current N and FSR
    x_THz, t_ps = sync_x_axes_to_grid(st, ax_spec, ax_pulse, ax_spec_top, pad_frac=0.01)


    def DV_to_detuning_GHz(DV, kappa_avg):
        # detuning_Hz = DV * kappa/(2π)
        return DV * kappa_avg / (2*np.pi) * 1e-9

    def detuning_GHz_to_DV(det_GHz, kappa_avg):
        return det_GHz * 1e9 * (2*np.pi) / kappa_avg
    
    def Pnorm_to_W(Pnorm, Pnorm_factor):
        return Pnorm * Pnorm_factor

    def W_to_Pnorm(P_W, Pnorm_factor):
        return P_W / Pnorm_factor
    
    def S_to_W(S, Pnorm_factor):
        return (S**2) * Pnorm_factor
    
    def W_to_S(P_W, Pnorm_factor):
        return np.sqrt(max(P_W / Pnorm_factor, 0.0))

    def get_fast_time_axis(st):
        """
        Returns (x, xlabel) for the time-domain plot.
        Tries to use physical fast time in ps if available; otherwise uses index.
        """
        if hasattr(st, "tau_ps"):
            return st.tau_ps, "Fast time τ (ps)"
        if hasattr(st, "tau_phys"):
            return st.tau_phys * 1e12, "Fast time τ (ps)"
        if hasattr(st, "tau"):
            return st.tau, "Fast time τ (norm)"
        # fallback
        return np.arange(st.number_modes), "Sample index"

    # ---- Power plot (ring buffer): 3 TRACES ONLY (all normalized),
    #      with correlated physical axes (secondary axes, no extra traces) ----
    pow_x = np.full(Nkeep, np.nan, dtype=float)
    pow_i = {"i": 0}

    # Histories (ONLY 3 quantities)
    y_Pcav_W = np.full(Nkeep, np.nan, dtype=float)  # displayed normalized intracavity power [norm]
    y_DV     = np.full(Nkeep, np.nan, dtype=float)  # DV
    y_Pnorm  = np.full(Nkeep, np.nan, dtype=float)  # P_norm


    # ---------- styling knobs ----------
    FS_LABEL = 6.0
    FS_TICK  = 5.6
    LW_MAIN  = 1.2

    RIGHT_OFFSETS = [0, 46, 92]  # extra separation between the three right-side axes

    def _style_axis_right(ax, color, label, offset_pts=0):
        # Move spine outward
        ax.spines["right"].set_position(("outward", offset_pts))
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
    
        # Ticks & label on the right only
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    
        # Spine styling
        ax.spines["right"].set_linewidth(1.0)
        ax.spines["right"].set_color(color)
    
        # Tick styling
        ax.tick_params(
            axis="y",
            which="both",
            colors=color,
            labelsize=FS_TICK,
            pad=2,
            width=1.0,
            length=4
        )
    
        # Label styling
        ax.set_ylabel(label, fontsize=FS_LABEL, labelpad=3, color=color)
        ax.yaxis.label.set_color(color)
        
    # Secondary RIGHT axis = normalized intracavity power scale (mirrors the left axis)
    
    K_Pcav_mW = 1e3 * float(st.hbar * st.ome_pump * (st.E_amp_2_norm_factor**2)) / st.t_phys_round_trip

    def Pcav_mW_to_norm(PmW):
        return PmW / K_Pcav_mW
    
    def Pcav_norm_to_mW(Pn):
        return Pn * K_Pcav_mW


    # Base axis: normalized intracavity mean ⟨|E|²⟩ (LEFT side)
    ax_pow.set_title("Intracavity power evolution", fontsize=13, fontweight="bold")
    ax_pow.set_xlabel("Time (ns)")
    ax_pow.grid(True, alpha=0.25)
    ax_pow.set_ylim(0, 5)
    apply_minor_ticks(ax_pow)

    
    # Compact HUD: use the same font size as the live numeric text boxes.
    HUD_FONT_SIZE = 5.0
    hud = ax_pow.text(
        0.010, 0.985, "",               # (x,y) in axes fraction
        transform=ax_pow.transAxes,
        ha="left", va="top",
        fontsize=HUD_FONT_SIZE,
        linespacing=1.0,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.5", alpha=0.85)
    )

    # Recipe status must be attached to an Axes when blit=True.
    # A figure-level text has axes=None and can trigger:
    # AttributeError: 'NoneType' object has no attribute '_get_view'
    # in Matplotlib's blit cache.  Placing it just below the temporal
    # plot keeps the same visual intent while remaining blit-safe.
    recipe_status_text = ax_pulse.text(
        0.00, -0.34,
        "Recipe: idle",
        transform=ax_pulse.transAxes,
        ha="left", va="center",
        fontsize=9,
        color="black",
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.65", alpha=0.90)
    )
    recipe_status_text.set_animated(True)
    recipe_runner = RecipeRunner(st, recipe_status_text)
    ui["recipe_runner"] = recipe_runner
    ui["recipe_status_text"] = recipe_status_text



    # Base axis = NORMALIZED intracavity power (LEFT)
    ax_pow.set_ylabel(
        "Intracavity power [norm]",
        fontsize=FS_LABEL,
        color="tab:blue"
    )
    
    ax_pow.tick_params(
        axis="y",
        labelsize=FS_TICK,
        pad=4,
        width=1.0,
        length=4,
        colors="tab:blue"
    )
    
    ax_pow.spines["left"].set_visible(True)
    ax_pow.spines["left"].set_color("tab:blue")
    ax_pow.spines["left"].set_linewidth(1.0)

    ax_pow.spines["right"].set_visible(False)



    
    ax_pow_norm = ax_pow.secondary_yaxis(
        "right",
        functions=(Pcav_norm_to_mW, Pcav_mW_to_norm)
    )
    ax_pow_norm.set_ylabel("Intracavity power [mW]", fontsize=FS_LABEL, color="tab:blue")
    ax_pow_norm.tick_params(axis="y", which="both", colors="tab:blue", labelsize=FS_TICK, pad=4, width=1.0, length=4)
    ax_pow_norm.spines["right"].set_color("tab:blue")
    ax_pow_norm.spines["right"].set_linewidth(1.0)
    
    ax_pow_norm.spines["right"].set_position(("outward", RIGHT_OFFSETS[0]))  # = 0, explicit
    ax_pow_norm.set_zorder(10)  # keep it on top of base patch
    apply_minor_ticks(ax_pow_norm)


    # ----- Extra RIGHT axes for DV and P_norm (normalized), still only 3 traces total -----
    ax_pow_R2 = ax_pow.twinx()
    _style_axis_right(ax_pow_R2, "tab:green", "Detuning [norm]", offset_pts=RIGHT_OFFSETS[1])
    ax_pow_R2.set_ylim(DV_min, DV_max)
    apply_minor_ticks(ax_pow_R2)
    
    ax_pow_R3 = ax_pow.twinx()
    _style_axis_right(ax_pow_R3, "tab:purple", "Pump input [norm]", offset_pts=RIGHT_OFFSETS[2])
    ax_pow_R3.set_ylim(Pnorm_min, Pnorm_max)
    apply_minor_ticks( ax_pow_R3) 

    # Lines (3 TOTAL)
    line_Pcav_W, = ax_pow.plot([], [], color="tab:blue", linewidth=LW_MAIN)
    line_DV,        = ax_pow_R2.plot([], [], color="tab:green", linewidth=LW_MAIN)
    line_Pnorm,     = ax_pow_R3.plot([], [], color="tab:purple", linewidth=LW_MAIN)

    WINDOW_NS = 10.0  # show 0 to 10 ns
        
    # Seed a clean initial 0..10 ns window before the first user interaction
    x_init = np.linspace(0.0, WINDOW_NS, 2)
    y_nan2 = np.array([np.nan, np.nan], dtype=float)
    line_Pcav_W.set_data(x_init, y_nan2)
    line_DV.set_data(x_init, y_nan2)
    line_Pnorm.set_data(x_init, y_nan2)

    ax_pow.tick_params(axis="x", labelsize=FS_TICK, pad=2, width=0.8, length=3)
    WINDOW_NS = 10.0  # visible intracavity-power history window
    ax_pow.set_xlim(0.0, WINDOW_NS)


    # # ---- Sliders ----
    
    # ---- Sliders ----
    
    slider_spacing = 0.0225
    det_slider_bottom = SCOL["top"]
    slider_height = 0.0185
    slider_width = SCOL["w"]
    slider_x = SCOL["x"]
    
    def textbox_next_to_slider(ax_slider, width=0.040, height=None, gap=0.008):
        """Create a value box aligned exactly to a slider axis.

        The key detail is that the box uses the same y-position and height as the
        slider axis, so adjacent rows cannot vertically overlap.
        """
        pos = ax_slider.get_position()
        if height is None:
            height = pos.height
        return plt.axes([pos.x1 + gap, pos.y0, width, height])


    def add_slider_ticks_inside_step(ax, slider, step=1.0, y0=0.25, y1=0.75, lw=0.8, color="0.25"):
        vmin, vmax = float(slider.valmin), float(slider.valmax)
    
        start = np.ceil(vmin / step) * step
        xs = np.arange(start, vmax + 0.5*step, step)
    
        trans = ax.get_xaxis_transform()  # x=data, y=axes
        lines = []
        for x in xs:
            lines.append(
                ax.plot([x, x], [y0, y1], transform=trans,
                        clip_on=True, linewidth=lw, color=color, zorder=5)[0]
            )
        return lines

    
    ax_det = plt.axes([slider_x, det_slider_bottom, slider_width ,slider_height])
    det_slider = Slider(ax_det, "Detuning (DV)", DV_min, DV_max, valinit=st.DV, valfmt='%.3f')
    ui["det_inside_ticks"] = add_slider_ticks_inside_step(ax_det, det_slider, step=1.0)

    det_init_GHz = DV_to_detuning_GHz(st.DV, st.kappa_avg)
    
    
    det_slider.valtext.set_visible(False)
    
    ax_det_text = textbox_next_to_slider(ax_det)
    det_text_DV = TextBox(ax_det_text,'',initial=f"{st.DV:.3f}",color='0.95', hovercolor='0.95')
    
    
    det_min_GHz = DV_to_detuning_GHz(DV_min, st.kappa_avg)
    det_max_GHz = DV_to_detuning_GHz(DV_max, st.kappa_avg)
    det_init_GHz = DV_to_detuning_GHz(st.DV, st.kappa_avg)

    
    ax_detGHz = plt.axes([slider_x, det_slider_bottom-slider_spacing,slider_width,slider_height])
    slider_detGHz = Slider(
        ax_detGHz,
        "Detuning (GHz)",
        det_min_GHz, det_max_GHz,
        valinit=det_init_GHz,
        valfmt='%.3f')
    ui["detGHz_inside_ticks"] = add_slider_ticks_inside_step(ax_detGHz, slider_detGHz, step=1.0)


    slider_detGHz.valtext.set_visible(False)
    
    ax_det_GHz_text = textbox_next_to_slider(ax_detGHz)
    det_text_GHz = TextBox(ax_det_GHz_text,'',initial=f"{det_init_GHz:.3f}",color='0.95', hovercolor='0.95')
    
    # Slew is now DV/ns
    slew_min = 1e-6
    slew_max = 1e1
    
    ax_slew = plt.axes([slider_x, det_slider_bottom-slider_spacing*2,slider_width,slider_height])
    slider_slew = Slider(
        ax_slew,
        "Detuning slew (DV/ns)",
        slew_min, slew_max,
        valinit=st.detuning_slew_rate,
        valfmt='%.3f'
    )
    ui["slew_inside_ticks"] = add_slider_ticks_inside_step(ax_slew, slider_slew, step=1.0)


    slider_slew.valtext.set_visible(False)
    
    ax_slew_text = textbox_next_to_slider(ax_slew)
    text_slew = TextBox(ax_slew_text, '', initial=f"{st.detuning_slew_rate:.3f}",color='0.95', hovercolor='0.95')
    
    
    
    ax_Pnorm = plt.axes([slider_x, det_slider_bottom-slider_spacing*4,slider_width,slider_height])
    Pnorm_slider = Slider(
        ax_Pnorm,
        "Pump power (norm)",
        Pnorm_min,
        Pnorm_max,
        valinit=st.P_norm,
        valfmt='%.2f'
    )
    ui["Pnorm_inside_ticks"] = add_slider_ticks_inside_step(ax_Pnorm, Pnorm_slider, step=1.0)

    Pnorm_slider.valtext.set_visible(False)
    
    
    ax_Pnorm_text = textbox_next_to_slider(ax_Pnorm)
    Pnorm_text = TextBox(ax_Pnorm_text,'',initial=f"{st.P_norm:.2f}",color='0.95', hovercolor='0.95')
    
    # ---- Physical power slider in mW (derived from Pnorm limits) ----
    P_min_mW  = 1e3 * Pnorm_to_W(Pnorm_min, st.P_in_norm_factor)
    P_max_mW  = 1e3 * Pnorm_to_W(Pnorm_max, st.P_in_norm_factor)
    P_init_mW = 1e3 * Pnorm_to_W(st.P_norm,  st.P_in_norm_factor)
    
    ax_PmW = plt.axes([slider_x, det_slider_bottom-slider_spacing*5,slider_width,slider_height])
    PmW_slider = Slider(
        ax_PmW,
        "Pump power (mW)",
        P_min_mW,
        P_max_mW,
        valinit=P_init_mW,
        valfmt='%.2f'
    )

    PmW_slider.valtext.set_visible(False)
    
    ax_PmW_text = textbox_next_to_slider(ax_PmW)
    PmW_text = TextBox(ax_PmW_text,'',initial=f"{P_init_mW:.2f}",color='0.95', hovercolor='0.95')
    PmW_tick_step = 5.0  # or 10, 20, 50 depending on your range
    ui["PmW_inside_ticks"] = add_slider_ticks_inside_step(
        ax_PmW, PmW_slider,
        step=PmW_tick_step, y0=0.30, y1=0.70, lw=0.7
    )

    # ---- Pump-power slew sliders ----
    # Pnorm/ns controls how fast the pump power target is reached.
    # A large default keeps the previous near-instant behaviour unless the user slows it down.
    power_slew_min = 1e-6
    power_slew_max = 1e2
    power_slew_init = float(getattr(st, "power_slew_rate", 100.0))

    ax_Pslew = plt.axes([slider_x, det_slider_bottom-slider_spacing*6, slider_width, slider_height])
    Pslew_slider = Slider(
        ax_Pslew,
        "Pump slew (norm/ns)",
        power_slew_min,
        power_slew_max,
        valinit=power_slew_init,
        valfmt="%.3f"
    )
    Pslew_slider.valtext.set_visible(False)
    ax_Pslew_text = textbox_next_to_slider(ax_Pslew)
    Pslew_text = TextBox(ax_Pslew_text, "", initial=f"{power_slew_init:.3f}", color='0.95', hovercolor='0.95')
    ui["Pslew_inside_ticks"] = add_slider_ticks_inside_step(ax_Pslew, Pslew_slider, step=10.0, y0=0.30, y1=0.70, lw=0.7)

    # Physical equivalent in mW/ns, linked to the normalized power-rate slider.
    power_slew_mW_min = 1e3 * Pnorm_to_W(power_slew_min, st.P_in_norm_factor)
    power_slew_mW_max = 1e3 * Pnorm_to_W(power_slew_max, st.P_in_norm_factor)
    power_slew_mW_init = 1e3 * Pnorm_to_W(power_slew_init, st.P_in_norm_factor)

    ax_Pslew_mW = plt.axes([slider_x, det_slider_bottom-slider_spacing*7, slider_width, slider_height])
    Pslew_mW_slider = Slider(
        ax_Pslew_mW,
        "Pump slew (mW/ns)",
        power_slew_mW_min,
        power_slew_mW_max,
        valinit=power_slew_mW_init,
        valfmt="%.3f"
    )
    Pslew_mW_slider.valtext.set_visible(False)
    ax_Pslew_mW_text = textbox_next_to_slider(ax_Pslew_mW)
    Pslew_mW_text = TextBox(ax_Pslew_mW_text, "", initial=f"{power_slew_mW_init:.3f}", color='0.95', hovercolor='0.95')
    Pslew_mW_tick_step = max(power_slew_mW_max / 10.0, 1e-12)
    ui["Pslew_mW_inside_ticks"] = add_slider_ticks_inside_step(
        ax_Pslew_mW, Pslew_mW_slider,
        step=Pslew_mW_tick_step, y0=0.30, y1=0.70, lw=0.7
    )



    syncing = {"flag": False}
    

    def on_Pnorm_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        Pn = float(Pnorm_slider.val)
        st.set_targets(P_norm=Pn)
    
        # sync mW slider
        PmW = 1e3 * Pnorm_to_W(Pn, st.P_in_norm_factor)
        PmW_slider.set_val(PmW)
        Pnorm_text.set_val(f"{Pn:.2f}")
        PmW_text.set_val(f"{PmW:.2f}")
    
        syncing["flag"] = False
        
    def on_Pnorm_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        Pn = float(text)
        
        #passing the input
        st.set_targets(P_norm=Pn)
        
        #updating sliders/boxes
        Pnorm_slider.set_val(Pn)
        PmW = 1e3 * Pnorm_to_W(Pn, st.P_in_norm_factor)
        PmW_slider.set_val(PmW)
        PmW_text.set_val(f"{PmW:.2f}")
        
        
        syncing["flag"] = False

    def on_PmW_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        PmW = float(PmW_slider.val)
        P_W = PmW * 1e-3
        Pn = W_to_Pnorm(P_W, st.P_in_norm_factor)
    

        st.set_targets(P_norm=Pn)
        
        Pnorm_slider.set_val(Pn)
        Pnorm_text.set_val(f"{Pn:.2f}")
        PmW_text.set_val(f"{PmW:.2f}")
    
        syncing["flag"] = False
        
    def on_PmW_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        PmW = float(text)
        P_W = PmW * 1e-3
        Pn = W_to_Pnorm(P_W, st.P_in_norm_factor)
        
        #passing the input
        st.set_targets(P_norm=Pn)
        
        #updating sliders/boxes
        Pnorm_slider.set_val(Pn)
        Pnorm_text.set_val(f"{Pn:.2f}")
        PmW_slider.set_val(PmW)
        
        
        syncing["flag"] = False


    def on_Pslew_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True

        rate_norm_ns = max(float(Pslew_slider.val), 0.0)
        st.set_targets(power_slew=rate_norm_ns)

        rate_mW_ns = 1e3 * Pnorm_to_W(rate_norm_ns, st.P_in_norm_factor)
        Pslew_mW_slider.set_val(rate_mW_ns)
        Pslew_text.set_val(f"{rate_norm_ns:.3f}")
        Pslew_mW_text.set_val(f"{rate_mW_ns:.3f}")

        syncing["flag"] = False

    def on_Pslew_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True

        rate_norm_ns = max(float(text), 0.0)
        st.set_targets(power_slew=rate_norm_ns)

        rate_mW_ns = 1e3 * Pnorm_to_W(rate_norm_ns, st.P_in_norm_factor)
        Pslew_slider.set_val(rate_norm_ns)
        Pslew_mW_slider.set_val(rate_mW_ns)
        Pslew_mW_text.set_val(f"{rate_mW_ns:.3f}")

        syncing["flag"] = False

    def on_Pslew_mW_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True

        rate_mW_ns = max(float(Pslew_mW_slider.val), 0.0)
        rate_norm_ns = W_to_Pnorm(rate_mW_ns * 1e-3, st.P_in_norm_factor)
        st.set_targets(power_slew=rate_norm_ns)

        Pslew_slider.set_val(rate_norm_ns)
        Pslew_text.set_val(f"{rate_norm_ns:.3f}")
        Pslew_mW_text.set_val(f"{rate_mW_ns:.3f}")

        syncing["flag"] = False

    def on_Pslew_mW_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True

        rate_mW_ns = max(float(text), 0.0)
        rate_norm_ns = W_to_Pnorm(rate_mW_ns * 1e-3, st.P_in_norm_factor)
        st.set_targets(power_slew=rate_norm_ns)

        Pslew_slider.set_val(rate_norm_ns)
        Pslew_text.set_val(f"{rate_norm_ns:.3f}")
        Pslew_mW_slider.set_val(rate_mW_ns)

        syncing["flag"] = False

    Pnorm_slider.on_changed(on_Pnorm_changed)
    Pnorm_text.on_submit(on_Pnorm_text_submit)
    PmW_slider.on_changed(on_PmW_changed)
    PmW_text.on_submit(on_PmW_text_submit)
    Pslew_slider.on_changed(on_Pslew_changed)
    Pslew_text.on_submit(on_Pslew_text_submit)
    Pslew_mW_slider.on_changed(on_Pslew_mW_changed)
    Pslew_mW_text.on_submit(on_Pslew_mW_text_submit)

    
    def on_det_DV_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        DV = float(det_slider.val)

    
        # drive the target continuously (smooth via st.detuning_slew in step())
        st.set_targets(detuning=DV)
    
        # keep the GHz slider synced
        det_GHz = DV_to_detuning_GHz(DV, st.kappa_avg)
        slider_detGHz.set_val(det_GHz)
        
        #updating textboxes
        det_text_DV.set_val(f"{DV:.2f}")
        det_text_GHz.set_val(f"{det_GHz:.2f}")
    
        syncing["flag"] = False
        
    
    
    def on_det_DV_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        DV = float(text)
        
        #passing the input
        st.set_targets(detuning=DV)
        
        #updating sliders/boxes
        det_slider.set_val(DV)
        det_GHz = DV_to_detuning_GHz(DV, st.kappa_avg)
        slider_detGHz.set_val(det_GHz)
        det_text_GHz.set_val(f"{det_GHz:.2f}")
        
        syncing["flag"] = False
        
        
             
    def on_detGHz_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        det_GHz = float(slider_detGHz.val)
        DV = detuning_GHz_to_DV(det_GHz, st.kappa_avg)
        
        
        #other sliders
        det_text_DV.set_val(f"{DV:.2f}")
        det_text_GHz.set_val(f"{det_GHz:.2f}")
        det_slider.set_val(DV)
        
        
        st.set_targets(detuning=DV)
    
        syncing["flag"] = False
   
    
    def on_det_GHz_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        det_GHz = float(text)
        
        #passing the input
        DV = detuning_GHz_to_DV(det_GHz, st.kappa_avg)
        st.set_targets(detuning=DV)
        
        #updating sliders/boxes
        det_slider.set_val(DV)
        slider_detGHz.set_val(det_GHz)
        det_text_DV.set_val(f"{DV:.2f}")
        
        
        syncing["flag"] = False
        
    det_slider.on_changed(on_det_DV_changed)
    det_text_DV.on_submit(on_det_DV_text_submit)
    slider_detGHz.on_changed(on_detGHz_changed)    
    det_text_GHz.on_submit(on_det_GHz_text_submit)
    
    def slew_DVns_to_GHzns(slew_DVns, kappa_avg):
        # detuning_GHz = DV * kappa/(2π) * 1e-9
        # so derivative: (GHz/ns) = (DV/ns) * kappa/(2π) * 1e-9
        return slew_DVns * kappa_avg / (2*np.pi) * 1e-9
    
    def slew_GHzns_to_DVns(slew_GHzns, kappa_avg):
        return slew_GHzns * 1e9 * (2*np.pi) / kappa_avg

    
    def on_slew_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        slew_DVns = float(slider_slew.val)
        st.detuning_slew_rate = slew_DVns
        text_slew.set_val(f"{slew_DVns:.2f}")
    
        # --- NEW: sync GHz/ns slider + textbox ---
        slew_GHzns = slew_DVns_to_GHzns(slew_DVns, st.kappa_avg)
        slider_slewGHz.set_val(slew_GHzns)
        text_slewGHz.set_val(f"{slew_GHzns:.3f}")
    
        syncing["flag"] = False

        
    def on_slewGHz_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        slew_GHzns = float(slider_slewGHz.val)
        slew_DVns = slew_GHzns_to_DVns(slew_GHzns, st.kappa_avg)
    
        st.detuning_slew_rate = slew_DVns
    
        # sync DV/ns slider + both textboxes
        slider_slew.set_val(slew_DVns)
        text_slew.set_val(f"{slew_DVns:.2f}")
        text_slewGHz.set_val(f"{slew_GHzns:.3f}")
    
        syncing["flag"] = False
    
    
    def on_slewGHz_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        slew_GHzns = float(text)
        slew_DVns = slew_GHzns_to_DVns(slew_GHzns, st.kappa_avg)
    
        st.detuning_slew_rate = slew_DVns
    
        slider_slew.set_val(slew_DVns)
        slider_slewGHz.set_val(slew_GHzns)
        text_slew.set_val(f"{slew_DVns:.2f}")
    
        syncing["flag"] = False

    def on_slew_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        slew = float(text)
        st.detuning_slew_rate = slew
        slider_slew.set_val(slew)
    
        syncing["flag"] = False

    slider_slew.on_changed(on_slew_changed)
    text_slew.on_submit(on_slew_text_submit)
    
    # ---- Physical slew slider in GHz/ns (derived from DV/ns limits) ----
    slew_min_GHzns = slew_DVns_to_GHzns(slew_min, st.kappa_avg)
    slew_max_GHzns = slew_DVns_to_GHzns(slew_max, st.kappa_avg)
    slew_init_GHzns = slew_DVns_to_GHzns(st.detuning_slew_rate, st.kappa_avg)
    
    ax_slewGHz = plt.axes([slider_x, det_slider_bottom - slider_spacing*3,slider_width,slider_height])
    slider_slewGHz = Slider(
        ax_slewGHz,
        "Detuning slew (GHz/ns)",
        slew_min_GHzns, slew_max_GHzns,
        valinit=slew_init_GHzns,
        valfmt="%.3f"
    )
    # --- Inside ticks for slew in GHz/ns ---
    slewGHz_tick_step = 0.5  # pick what "one tick" means for you (0.1, 0.5, 1.0 ...)
    ui["slewGHz_inside_ticks"] = add_slider_ticks_inside_step(
        ax_slewGHz, slider_slewGHz,
        step=slewGHz_tick_step, y0=0.30, y1=0.70, lw=0.7
    )

    slider_slewGHz.valtext.set_visible(False)
    
    ax_slewGHz_text = textbox_next_to_slider(ax_slewGHz)
    text_slewGHz = TextBox(ax_slewGHz_text, "", initial=f"{slew_init_GHzns:.3f}",color='0.95', hovercolor='0.95')
    
    slider_slewGHz.on_changed(on_slewGHz_changed)
    text_slewGHz.on_submit(on_slewGHz_text_submit)
    
    # --- RESPONSIVE SIDE PANELS ---
    # One typography system is shared by fields, variable labels, buttons and
    # bottom controls so the interface reads as a finished application.
    UI_FONT_SIZE = 8.2
    UI_SECTION_FONT_SIZE = 9.2

    # Left side: physical/system parameters. Right side: run, recipe, save/export, axes.
    PANEL = dict(
        left_x=LCOL["x"], left_w=LCOL["w"],
        right_x=RCOL["x"], right_w=RCOL["w"],
        h=0.0275,
        btn_h=0.0290,
        title_fs=UI_FONT_SIZE,
        face="0.95",
    )

    def _panel_fontsize(base):
        return base * _font_scale()

    def add_panel_textbox(key, label_text, initial_val, y, side="left", w=None, h=None):
        x = PANEL["left_x"] if side == "left" else PANEL["right_x"]
        width = PANEL["left_w"] if side == "left" else PANEL["right_w"]
        if w is not None:
            width = w
        if h is None:
            h = PANEL["h"]
        ax = plt.axes([x, y, width, h])
        ax.set_title(label_text, fontsize=_panel_fontsize(PANEL["title_fs"]), pad=1.8)
        tb = TextBox(ax, "", initial=initial_val, color=PANEL["face"], hovercolor=PANEL["face"])
        try:
            tb.text_disp.set_fontsize(_panel_fontsize(UI_FONT_SIZE))
            tb.text_disp.set_clip_on(True)
            tb.text_disp.set_clip_path(ax.patch)
            tb.text_disp.set_horizontalalignment("left")
        except Exception:
            pass
        ax.set_navigate(False)
        ax.set_zorder(2000)
        ax.patch.set_alpha(1.0)
        ui[f"ax_{key}"] = ax
        ui[f"txt_{key}"] = tb
        return ax, tb

    def add_panel_button(key, label, y, side="right", color="0.90", hover="0.82", fs=7.0, weight="normal", x=None, w=None):
        if x is None:
            x = PANEL["left_x"] if side == "left" else PANEL["right_x"]
        if w is None:
            w = PANEL["left_w"] if side == "left" else PANEL["right_w"]
        ax = plt.axes([x, y, w, PANEL["btn_h"]])
        btn = Button(ax, label)
        btn.color = color
        btn.hovercolor = hover
        ax.set_facecolor(color)
        btn.label.set_fontsize(_panel_fontsize(UI_FONT_SIZE))
        btn.label.set_fontweight(weight)
        ax.set_navigate(False)
        ax.set_zorder(2000)
        ui[f"ax_{key}"] = ax
        ui[f"btn_{key}"] = btn
        return ax, btn

    # Left and right sidebars use the same width and the same vertical span as
    # the central graph block. Controls are distributed on a common vertical
    # grid to give the interface a balanced, finished appearance.
    PANEL_TOP = 0.895
    PANEL_BOTTOM = 0.335

    def distributed_y_positions(count, top=PANEL_TOP, bottom=PANEL_BOTTOM):
        """Return evenly distributed lower-left y positions for panel controls."""
        if count <= 1:
            return [top]
        return list(np.linspace(top, bottom, count))

    # Left column: one preset field, one import button and ten parameter fields.
    fig.text(
        PANEL["left_x"], 0.946, "System / resonator",
        ha="left", va="bottom", fontsize=UI_SECTION_FONT_SIZE, fontweight="bold"
    )
    left_y = distributed_y_positions(12)

    add_panel_textbox(
        "startup", "Startup configuration",
        getattr(st, "startup_config_name", "default_startup_wilson2019_GaP"),
        left_y[0], side="left",
    )
    add_panel_button(
        "import_startup", "Import preset", left_y[1], side="left",
        color="0.90", hover="0.82", fs=6.8
    )

    compact_fields = [
        ("modes",  "Number of modes",       f"{st.number_modes}"),
        ("fsr",    "FSR [Hz]",               f"{st.fsr:.2e}"),
        ("lambda", "Pump λ [nm]",            f"{st.wvl_pump*1e9:.1f}"),
        ("q",      "Q factor",               f"{st.Q:.2e}"),
        ("aeff",   "Aeff [um2]",             f"{st.Aeff*1e12:.2f}"),
        ("n2",     "n2 [m2/W]",              f"{st.n2:.2e}"),
        ("eta",    "Eta (coupling coeff)",   f"{st.eta:.3f}"),
        ("dint",   "Dispersion file name",   st.dint_file_path),
        ("pump",   "Pump input file name",   st.input_field_file),
        ("noise_level", "Noise level",       f"{getattr(st, 'noise_level', 1.0):.3g}"),
    ]
    for y, (key, lab, val) in zip(left_y[2:], compact_fields):
        add_panel_textbox(key, lab, val, y, side="left")

    # Right column: recipe, live controls, export controls and axis update.
    # The complete stack uses the same top and bottom limits as the left panel.
    fig.text(
        PANEL["right_x"], 0.946, "Run",
        ha="left", va="bottom", fontsize=UI_SECTION_FONT_SIZE, fontweight="bold"
    )
    # The first two rows use a slightly larger gap because a Matplotlib
    # TextBox title is drawn above the recipe field. This prevents the recipe
    # label from touching or overlapping the Run button, while all remaining
    # controls retain equal spacing.
    right_y = [0.900, 0.850] + list(np.linspace(0.810, PANEL_BOTTOM, 15))

    add_panel_button("run", "Run", right_y[0], side="right", color="0.88", hover="0.78", fs=7.2, weight="bold")
    add_panel_textbox("recipe", "Recipe file name", "pycombs_recipe_wilson2019_GaP_simulated_soliton.txt", right_y[1], side="right")
    add_panel_button("load_recipe", "Run recipe", right_y[2], side="right", color="0.88", hover="0.78", fs=7.0)

    # Recipe status uses the exact same Button construction, height, border,
    # background and row spacing as every other control in this column.
    ax_recipe_status, btn_recipe_status = add_panel_button(
        "recipe_status",
        "Recipe status: idle",
        right_y[3],
        side="right",
        color="0.93",
        hover="0.93",
        fs=7.0,
        weight="normal",
    )
    # This is a display-only status box; it intentionally has no click callback.
    btn_recipe_status.label.set_horizontalalignment("center")
    btn_recipe_status.label.set_verticalalignment("center")
    btn_recipe_status.label._pycombs_recipe_sidebar = True
    btn_recipe_status.label.set_animated(True)

    # Keep the old below-plot status artist hidden. The live status is displayed
    # in the matched button-style box directly below Run recipe.
    recipe_status_text.set_visible(False)
    recipe_status_text_sidebar = btn_recipe_status.label
    recipe_runner.status_artist = recipe_status_text_sidebar
    recipe_runner.set_status("Recipe status: idle")
    ui["recipe_status_text_sidebar"] = recipe_status_text_sidebar
    ui["ax_recipe_status"] = ax_recipe_status

    add_panel_button("noise", "Noise off", right_y[4], side="right", color="0.93", hover="0.84", fs=7.0)
    add_panel_button("eo_comb", "EO comb OFF", right_y[5], side="right", color="0.91", hover="0.82", fs=7.0)
    add_panel_button("apply_live", "Apply live", right_y[6], side="right", color="0.88", hover="0.78", fs=7.0)
    add_panel_button("apply", "Apply reset", right_y[7], side="right", color="0.86", hover="0.76", fs=7.0)

    # Export heading occupies a dedicated row between Apply reset and Save plot.
    fig.text(
        PANEL["right_x"], right_y[8] + 0.5 * PANEL["btn_h"], "Export",
        ha="left", va="center", fontsize=UI_SECTION_FONT_SIZE, fontweight="bold"
    )
    add_panel_button("save_plot", "Save plot", right_y[9], side="right", color="0.90", hover="0.82")
    add_panel_button("save_data", "Save data", right_y[10], side="right", color="0.90", hover="0.82")
    add_panel_button("save_pump", "Save pulse", right_y[11], side="right", color="0.90", hover="0.82")
    add_panel_button("save_both", "Save all", right_y[12], side="right", color="0.84", hover="0.74", weight="bold")
    add_panel_button("record", "Start recording", right_y[13], side="right", color="0.90", hover="0.82")

    SIDEBAR_KEYS = ["startup", "modes", "fsr", "lambda", "q", "aeff", "n2", "eta", "dint", "pump", "noise_level", "recipe"]
    for k in SIDEBAR_KEYS:
        axk = ui.get(f"ax_{k}")
        tbk = ui.get(f"txt_{k}")
        if axk is not None:
            axk.set_navigate(False)
            axk.set_zorder(2000)
            axk.patch.set_alpha(1.0)
        if tbk is not None:
            tbk.set_active(True)
            try:
                tbk.text_disp.set_fontsize(_panel_fontsize(UI_FONT_SIZE))
                tbk.text_disp.set_clip_on(True)
                tbk.text_disp.set_clip_path(tbk.ax.patch)
                tbk.text_disp.set_horizontalalignment("left")
            except Exception:
                pass

    # --- CLEAN AXIS CONTROL ---
    # Manual axis-entry boxes and the Auto axes toggle were removed to keep
    # the right sidebar uncluttered. This single button performs a one-shot
    # fit to the currently plotted data; the animation remains blit-friendly.
    AXIS_UI = dict(auto=True)

    # Axes heading also occupies a dedicated row below Start recording.
    fig.text(
        PANEL["right_x"], right_y[15] + 0.5 * PANEL["btn_h"], "Axes",
        ha="left", va="center", fontsize=UI_SECTION_FONT_SIZE, fontweight="bold"
    )
    add_panel_button(
        "axis_fit", "Update axes", right_y[16], side="right",
        color="0.88", hover="0.78", fs=7.2, weight="bold"
    )

    def update_pulse_view_button_label():
        # Spectral map button removed; this compatibility hook is intentionally a no-op.
        pass

    def set_pulse_view(mode):
        # The temporal profile and spectral evolution are now always shown;
        # the old spectral-map toggle button has been removed.
        pulse_view["mode"] = "temporal"
        pulse_line.set_visible(True)
        spectral_map_image.set_visible(True)
        ax_spectral_cbar.set_visible(True)
        ax_pulse.set_title("Temporal profile", fontsize=13, fontweight="bold")
        ax_pulse.set_xlabel("Fast time τ (ps)")
        ax_pulse.set_ylabel("Intracavity intensity\n(arb. units)")
        ax_pulse.set_xlim(float(t_ps[0]), float(t_ps[-1]))
        ax_map.set_title("Spectral evolution", fontsize=13, fontweight="bold")
        ax_map.set_xlabel("Mode number")
        ax_map.set_ylabel("Time (ns)")
        ax_map.set_xlim(float(mode_axis[0]), float(mode_axis[-1]))
        ax_map.set_ylim(0.0, WINDOW_NS)
        apply_manual_axis_boxes()
        update_pulse_view_button_label()
        fig.canvas.draw_idle()
        if hasattr(fig, "_blit_cache"):
            fig._blit_cache.clear()

    def on_pulse_view_clicked(event):
        set_pulse_view("temporal")

    update_pulse_view_button_label()

    # Compact all slider labels/value boxes so the bottom control bank does not collide with plots.
    for _sl in [det_slider, slider_detGHz, slider_slew, slider_slewGHz, Pnorm_slider, PmW_slider, Pslew_slider, Pslew_mW_slider]:
        try:
            _sl.label.set_fontsize(UI_FONT_SIZE)
            _sl.valtext.set_fontsize(UI_FONT_SIZE)
        except Exception:
            pass
    for _tb in [det_text_DV, det_text_GHz, text_slew, text_slewGHz, Pnorm_text, PmW_text, Pslew_text, Pslew_mW_text]:
        try:
            _tb.text_disp.set_fontsize(UI_FONT_SIZE)
            _tb.text_disp.set_clip_on(True)
            _tb.text_disp.set_clip_path(_tb.ax.patch)
        except Exception:
            pass

    fig.canvas.draw_idle()

    def noise_is_fully_enabled():
        return (
            bool(getattr(st, "noise_switch", False))
            and bool(getattr(st, "pump_noise_enabled", False))
            and bool(getattr(st, "cavity_noise_enabled", False))
            and bool(getattr(st, "startup_pump_noise_enabled", False))
            and bool(getattr(st, "startup_cavity_noise_enabled", False))
            and bool(getattr(st, "pump_refresh_noise_enabled", False))
        )

    def update_noise_button_label():
        ui["btn_noise"].label.set_text("Noise on" if noise_is_fully_enabled() else "Noise off")
        fig.canvas.draw_idle()

    def set_all_noise(enabled):
        enabled = bool(enabled)
        st.noise_switch = enabled
        st.pump_noise_enabled = enabled
        st.cavity_noise_enabled = enabled
        st.startup_pump_noise_enabled = enabled
        st.startup_cavity_noise_enabled = enabled
        st.pump_refresh_noise_enabled = enabled

        # Apply pump-noise changes immediately, not only at the next refresh.
        # Cavity noise is naturally controlled at startup/reinitialization.
        if hasattr(st, "fE_in_o"):
            if enabled:
                st.fE_in = st.fE_in_o + st._complex_noise(
                    st.pump_noise_amp, st.number_modes, kind="pump"
                )
            else:
                st.fE_in = st.fE_in_o.copy()

            st.input_pump_half_step = st.fE_in * (st.tal_step / 2)

    update_noise_button_label()
    fig.canvas.draw_idle()

    # EO-comb button label is initialized after its handler is defined below.

    # ---- Dashboard recording state and handlers ----
    recording = {
        "active": False,
        "folder": None,
        "mp4_path": None,
        "frame": 0,
        "busy": False,
        "stopping": False,
        "last_capture_time": 0.0,
        "session_id": 0,
    }

    def update_record_button_label():
        if recording["active"]:
            ui["btn_record"].label.set_text("Stop recording")
            ui["ax_record"].set_facecolor("0.80")
            ui["btn_record"].color = "0.80"
            ui["btn_record"].hovercolor = "0.70"
        else:
            ui["btn_record"].label.set_text("Start recording")
            ui["ax_record"].set_facecolor("0.90")
            ui["btn_record"].color = "0.90"
            ui["btn_record"].hovercolor = "0.80"
        fig.canvas.draw_idle()
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    def start_recording():
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        out_root = os.path.join(base_dir, "pycombs saved recordings")
        os.makedirs(out_root, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(out_root, f"pycombs_recording_{stamp}")
        os.makedirs(folder, exist_ok=True)
        mp4_path = os.path.join(folder, f"pycombs_recording_{stamp}.mp4")
        new_session_id = int(recording.get("session_id", 0)) + 1
        recording.update(active=True, folder=folder, mp4_path=mp4_path, frame=0,
                         busy=False, stopping=False, last_capture_time=0.0,
                         session_id=new_session_id)
        update_record_button_label()
        print(f"[Recording] Started SCREEN recording. Frames folder: {folder}")
        print(f"[Recording] Keep the pycombs window visible; this records the actual window pixels.")
        print(f"[Recording] MP4 will be created when you press Stop rec: {mp4_path}")
        # Capture one frame immediately, even if the simulation is paused.
        record_current_frame(force=True)

    def stop_recording():
        if not recording["active"]:
            return
        folder = recording.get("folder")
        mp4_path = recording.get("mp4_path")
        nframes = int(recording.get("frame", 0))
        # Stop new captures immediately, then wait for any capture already in
        # progress to finish before we list/delete frames. This prevents a final
        # frame_000001-style PNG from appearing after cleanup.
        recording["active"] = False
        recording["stopping"] = True
        t_wait0 = time.time()
        while recording.get("busy", False) and (time.time() - t_wait0) < 2.0:
            plt.pause(0.01)
        update_record_button_label()

        if not folder or nframes <= 0:
            print("[Recording] Stopped, but no frames were captured.")
            recording.update(folder=None, mp4_path=None, frame=0, busy=False, stopping=False)
            return

        frame_files = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(".png")
        ]
        if not frame_files:
            print(f"[Recording] No PNG frames found in: {folder}")
            recording.update(folder=None, mp4_path=None, frame=0, busy=False, stopping=False)
            return

        fps = max(1, int(round(1000.0 / float(interval_ms))))
        made_video = False

        try:
            import imageio.v2 as imageio
            # Force the FFMPEG backend. Without format="FFMPEG", some Spyder/Anaconda
            # setups accidentally route MP4 writing through the TIFF writer, causing:
            # "TiffWriter.write() got an unexpected keyword argument 'fps'".
            writer = imageio.get_writer(
                mp4_path,
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                quality=7,
                macro_block_size=2,
            )
            for fn in frame_files:
                writer.append_data(imageio.imread(fn))
            writer.close()
            made_video = True
        except Exception as e_imageio:
            print("[Recording] imageio/FFMPEG MP4 build unavailable:", repr(e_imageio))

        if not made_video:
            try:
                import subprocess
                pattern = os.path.join(folder, "frame_%06d.png")

                # First try the bundled ffmpeg from imageio-ffmpeg. This does not
                # require ffmpeg to be on Windows PATH.
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_exe = "ffmpeg"

                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-framerate", str(fps),
                    "-i", pattern,
                    "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264",
                    mp4_path,
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                made_video = (result.returncode == 0 and os.path.isfile(mp4_path))
                if not made_video:
                    print("[Recording] ffmpeg MP4 build failed:", result.stderr[-800:])
            except Exception as e_ffmpeg:
                print("[Recording] ffmpeg MP4 build unavailable:", repr(e_ffmpeg))

        if made_video and os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 0:
            print(f"[Recording] Saved MP4: {mp4_path}")

            # Clean up temporary PNG frames only after the MP4 has been
            # successfully written and verified to be non-empty. If MP4
            # encoding fails, the frames are intentionally kept as backup.
            # Re-scan the folder at cleanup time, not just the earlier frame list,
            # so any last frame created during the stop-click is also removed.
            deleted = 0
            remaining = []
            for cleanup_attempt in range(8):
                remaining = [
                    name for name in os.listdir(folder)
                    if name.lower().startswith("frame_") and name.lower().endswith(".png")
                ]
                if not remaining:
                    break
                for name in remaining:
                    fn = os.path.join(folder, name)
                    try:
                        os.remove(fn)
                        deleted += 1
                    except (PermissionError, OSError):
                        pass
                if cleanup_attempt < 7:
                    plt.pause(0.05)

            remaining = [
                name for name in os.listdir(folder)
                if name.lower().startswith("frame_") and name.lower().endswith(".png")
            ]
            if remaining:
                print(f"[Recording] WARNING: temporary frames still locked: {remaining}")
            else:
                print(f"[Recording] Deleted {deleted} temporary PNG frames.")
                print(f"[Recording] Folder now contains the MP4 only: {folder}")
        else:
            print("[Recording] Saved PNG frames only. To make MP4 automatically, install imageio/ffmpeg or add ffmpeg to PATH.")
            print("[Recording] Frames folder:", folder)

        recording.update(folder=None, mp4_path=None, frame=0, busy=False, stopping=False)

    def on_record_clicked(_event):
        if recording["active"]:
            stop_recording()
        else:
            start_recording()

    def _get_screen_capture_bbox():
        """Return the GUI window bounding box in screen pixels for PIL.ImageGrab.

        This records what is actually visible on screen instead of reading the
        Matplotlib canvas buffer. That avoids the blit=True problem where the
        saved frames can miss animated artists.
        """
        try:
            window = getattr(fig.canvas.manager, "window", None)
            if window is None:
                return None

            # Qt backends: frameGeometry includes the toolbar and title bar.
            if hasattr(window, "frameGeometry"):
                geom = window.frameGeometry()
                return (int(geom.x()), int(geom.y()), int(geom.x() + geom.width()), int(geom.y() + geom.height()))

            # Tk backends fallback.
            if all(hasattr(window, name) for name in ("winfo_rootx", "winfo_rooty", "winfo_width", "winfo_height")):
                x = int(window.winfo_rootx())
                y = int(window.winfo_rooty())
                w = int(window.winfo_width())
                h = int(window.winfo_height())
                return (x, y, x + w, y + h)
        except Exception:
            return None
        return None

    def record_current_frame(force=False):
        if (not recording["active"]) or recording.get("stopping", False) or recording.get("busy", False):
            return

        # Take local copies so a Stop-rec click cannot clear the folder/path
        # halfway through this capture call. This avoids the harmless but noisy
        # "expected str, bytes or os.PathLike object, not NoneType" message.
        folder = recording.get("folder")
        session_id = int(recording.get("session_id", 0))
        if not folder:
            return

        # Limit capture rate so the simulation stays fast. The live simulation
        # still uses blit=True; the recorder only screenshots the visible window.
        now = time.time()
        min_dt = max(0.001, float(interval_ms) / 1000.0)
        if (not force) and (now - recording.get("last_capture_time", 0.0) < 1.0 * min_dt):
            return

        recording["busy"] = True
        try:
            fig.canvas.flush_events()

            from PIL import ImageGrab
            bbox = _get_screen_capture_bbox()
            if bbox is None:
                # Last-resort fallback: full screen. This is less elegant, but it
                # still records what you see if the backend cannot report geometry.
                img = ImageGrab.grab()
            else:
                img = ImageGrab.grab(bbox=bbox)

            if (
                not recording.get("active", False)
                or recording.get("stopping", False)
                or int(recording.get("session_id", 0)) != session_id
                or recording.get("folder") != folder
            ):
                return

            width, height = img.size
            even_width = width - (width % 2)
            even_height = height - (height % 2)
            if even_width != width or even_height != height:
                img = img.crop((0, 0, even_width, even_height))

            recording["frame"] += 1
            frame_idx = int(recording["frame"])
            recording["last_capture_time"] = now
            frame_path = os.path.join(folder, f"frame_{frame_idx:06d}.png")
            img.save(frame_path)
        except Exception as e:
            print("[Recording] Screen capture failed:", repr(e))
            print("[Recording] Make sure the pycombs window is visible and not minimized/covered.")
        finally:
            recording["busy"] = False

    def on_close(_event):
        stop_recording()

    fig.canvas.mpl_connect("close_event", on_close)

    drag_state = {"active": False}

    running = {"flag": False}   # or False if you want it to start paused
     
    def on_press(event):
        if event.inaxes == ax_det:
            drag_state["active"] = True
    
    def on_release(event):
        if drag_state["active"]:
            drag_state["active"] = False
            st.set_targets(detuning=det_slider.val)  # commit once on release
            
    anim_ref = {"ani": None}  # defined BEFORE on_run_clicked

    def on_run_clicked(_event):
        running["flag"] = not running["flag"]
        ui["btn_run"].label.set_text("PAUSE" if running["flag"] else "RUN")
        ui["ax_run"].set_facecolor("0.95" if running["flag"] else "0.78")

        # In Spyder/Qt, a previous blit/backend hiccup can leave the animation
        # timer stopped even though the Run button toggles correctly.  Explicitly
        # restart the timer whenever the user presses Run.
        try:
            if anim_ref.get("ani") is not None and anim_ref["ani"].event_source is not None:
                anim_ref["ani"].event_source.start()
        except Exception as e:
            print(f"[Run] Could not restart animation timer: {e}")

        print("[Run] Simulation running." if running["flag"] else "[Run] Simulation paused.")
        fig.canvas.draw_idle()

    def on_noise_clicked(_event):
        set_all_noise(not noise_is_fully_enabled())
        update_noise_button_label()
        print(
            "Noise enabled: pump, cavity, startup, and refresh noise are ON."
            if noise_is_fully_enabled()
            else "Noise disabled: pump, cavity, startup, and refresh noise are OFF."
        )

    def update_eo_button_label():
        """Reflect EO-comb status in the compact main-window button."""
        enabled = bool(getattr(st, "eo_enabled", False))
        ui["btn_eo_comb"].label.set_text("EO comb ON" if enabled else "EO comb OFF")
        ui["ax_eo_comb"].set_facecolor("0.78" if enabled else "0.91")
        ui["btn_eo_comb"].color = "0.78" if enabled else "0.91"
        ui["btn_eo_comb"].hovercolor = "0.70" if enabled else "0.82"
        fig.canvas.draw_idle()

    def open_eo_comb_window(_event=None):
        """
        First-version EO-comb control panel.
        It intentionally lives in a separate Matplotlib window so the main sidebar
        stays compact.
        """
        # Do NOT pause the main simulation when opening the EO window.
        # Earlier versions saved was_running and forced running["flag"] = False here,
        # which made the solver look frozen after applying EO settings.
        cfg = st.get_eo_config()
        eo_fig = plt.figure(figsize=(5.3, 6.4))
        eo_fig.canvas.manager.set_window_title("pycombs EO comb control")
        eo_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        eo_ui = {}

        eo_fig.text(0.06, 0.955, "EO comb control", fontsize=13, fontweight="bold")
        eo_fig.text(
            0.06, 0.925,
            "Generated pump: CW seed → PM/IM modulation → Kerr-cavity input",
            fontsize=8,
        )

        def add_eo_textbox(key, label, value, y, x=0.52, w=0.34):
            eo_fig.text(0.08, y + 0.007, label, fontsize=8, ha="left", va="center")
            ax = eo_fig.add_axes([x, y, w, 0.035])
            tb = TextBox(ax, "", initial=str(value), color="0.95", hovercolor="0.95")
            try:
                tb.text_disp.set_fontsize(8)
            except Exception:
                pass
            eo_ui[key] = tb
            return tb

        def add_eo_button(key, label, x, y, w=0.22, color="0.88"):
            ax = eo_fig.add_axes([x, y, w, 0.043])
            btn = Button(ax, label)
            btn.color = color
            btn.hovercolor = "0.78"
            ax.set_facecolor(color)
            btn.label.set_fontsize(8)
            eo_ui[key] = btn
            eo_ui[f"ax_{key}"] = ax
            return btn

        y0 = 0.865
        dy = 0.052

        # Master and PM controls
        add_eo_textbox("enabled", "EO enabled [0/1]", int(cfg["enabled"]), y0)
        add_eo_textbox("rf_spacing_mu", "RF spacing [modes / μ]", f'{cfg["rf_spacing_mu"]:.6g}', y0 - dy)
        add_eo_textbox("pm_rf_power_dBm", "PM RF power [dBm]", f'{cfg["pm_rf_power_dBm"]:.6g}', y0 - 2*dy)
        add_eo_textbox("pm_vpi_V", "PM Vπ [V]", f'{cfg["pm_vpi_V"]:.6g}', y0 - 3*dy)
        add_eo_textbox("num_pm", "Number of PMs", f'{cfg["num_pm"]}', y0 - 4*dy)
        add_eo_textbox("pm_loss_dB", "Total PM loss [dB]", f'{cfg["pm_loss_dB"]:.6g}', y0 - 5*dy)
        add_eo_textbox("pm_phase_rad", "PM RF phase [rad]", f'{cfg["pm_phase_rad"]:.6g}', y0 - 6*dy)

        eo_fig.text(0.06, y0 - 7.0*dy + 0.02, "Intensity modulator", fontsize=9, fontweight="bold")
        add_eo_textbox("im_enabled", "IM enabled [0/1]", int(cfg["im_enabled"]), y0 - 8*dy)
        add_eo_textbox("im_rf_power_dBm", "IM RF power [dBm]", f'{cfg["im_rf_power_dBm"]:.6g}', y0 - 9*dy)
        add_eo_textbox("im_vpi_V", "IM Vπ [V]", f'{cfg["im_vpi_V"]:.6g}', y0 - 10*dy)
        add_eo_textbox("im_bias", "IM bias [π rad]", f'{cfg["im_bias"]:.6g}', y0 - 11*dy)
        add_eo_textbox("im_loss_dB", "IM loss [dB]", f'{cfg["im_loss_dB"]:.6g}', y0 - 12*dy)
        add_eo_textbox("im_phase_rad", "IM RF phase [rad]", f'{cfg["im_phase_rad"]:.6g}', y0 - 13*dy)
        add_eo_textbox("rf_impedance_ohm", "RF impedance [Ω]", f'{cfg["rf_impedance_ohm"]:.6g}', y0 - 14*dy)

        status = eo_fig.text(0.06, 0.105, "", fontsize=8, ha="left", va="center")

        def read_bool(key):
            return bool(int(float(eo_ui[key].text.strip())))

        def read_float(key):
            return float(eo_ui[key].text.strip())

        def read_int(key):
            return int(float(eo_ui[key].text.strip()))

        def apply_from_window(enabled_override=None):
            try:
                new_cfg = dict(
                    enabled=read_bool("enabled") if enabled_override is None else bool(enabled_override),
                    rf_spacing_mu=read_float("rf_spacing_mu"),
                    pm_rf_power_dBm=read_float("pm_rf_power_dBm"),
                    pm_vpi_V=read_float("pm_vpi_V"),
                    num_pm=read_int("num_pm"),
                    pm_loss_dB=read_float("pm_loss_dB"),
                    pm_phase_rad=read_float("pm_phase_rad"),
                    im_enabled=read_bool("im_enabled"),
                    im_rf_power_dBm=read_float("im_rf_power_dBm"),
                    im_vpi_V=read_float("im_vpi_V"),
                    im_bias=read_float("im_bias"),
                    im_loss_dB=read_float("im_loss_dB"),
                    im_phase_rad=read_float("im_phase_rad"),
                    rf_impedance_ohm=read_float("rf_impedance_ohm"),
                )

                st.set_eo_config(**new_cfg)
                st.apply_eo_comb_pump(enabled=new_cfg["enabled"], rebuild=True)
                sync_sidebar_from_state(ui, st)
                update_eo_button_label()

                # Refresh current plot artists enough that the change is visible while paused.
                spec_now = st.get_output_spectrum_dBm()
                stem_sc.set_offsets(np.column_stack([st.frq_grid * 1e-12, spec_now]))
                stem_lc.set_segments([((x, baseline_state["y"]), (x, y)) for x, y in zip(st.frq_grid * 1e-12, spec_now)])
                fig.canvas.draw_idle()
                # Keep animation timer alive after applying EO settings, especially in Spyder/Qt.
                try:
                    if anim_ref.get("ani") is not None and anim_ref["ani"].event_source is not None:
                        anim_ref["ani"].event_source.start()
                except Exception as e:
                    print(f"[EO comb] Could not restart animation timer after EO apply: {e}")

                status.set_text(
                    f"Applied. beta_PM={st.eo_last_beta_pm:.3g}, "
                    f"beta_IM={st.eo_last_beta_im:.3g}, "
                    f"trans={100*st.eo_last_transmission:.2f}%"
                )
                eo_fig.canvas.draw_idle()
            except Exception as e:
                status.set_text(f"EO apply error: {e}")
                eo_fig.canvas.draw_idle()
                print(f"[EO comb] Apply error: {e}")

        def on_apply_eo(_evt):
            # Apply EO should mean: enable the generated EO-comb pump now.
            # The textbox is updated too, so the window state and simulation state agree.
            eo_ui["enabled"].set_val("1")
            apply_from_window(enabled_override=True)
            try:
                if anim_ref.get("ani") is not None and anim_ref["ani"].event_source is not None:
                    anim_ref["ani"].event_source.start()
            except Exception as e:
                print(f"[EO comb] Could not restart animation timer after Apply EO: {e}")

        def on_disable_eo(_evt):
            eo_ui["enabled"].set_val("0")
            apply_from_window(enabled_override=False)

        def on_close_eo(_evt):
            # Closing the EO window should not change the run/pause state.
            plt.close(eo_fig)

        add_eo_button("apply", "Apply EO", 0.08, 0.035, w=0.24, color="0.82").on_clicked(on_apply_eo)
        add_eo_button("disable", "Disable EO", 0.38, 0.035, w=0.24, color="0.90").on_clicked(on_disable_eo)
        add_eo_button("close", "Close", 0.68, 0.035, w=0.20, color="0.90").on_clicked(on_close_eo)

        # Closing the EO window should not change the run/pause state.
        eo_fig.canvas.mpl_connect("close_event", lambda evt: None)
        update_eo_button_label()
        plt.show(block=False)

    def save_spectrum_snapshot(_event=None, out_dir=None, ts=None, flash=True):
        """
        Save a snapshot of:
          1) THROUGH-PORT physical spectrum (dBm + W)  <-- matches GUI plot
          2) Intracavity temporal pulse intensity (normalized)  <-- same as before
        """
        try:
            # ---------------------------
            # 1) Through-port spectrum
            # ---------------------------
            # This is exactly what your GUI plots
            spec_through_dBm = st.get_output_spectrum_dBm()
    
            # Also compute linear through-port power [W] per mode,
            # using the same physics used inside get_output_spectrum_dBm():
            # s_out = s_in - sqrt(kappa_ex) * a_mu_phys
            # P_out = hbar * omega_mu * |s_out|^2
            A_mu_norm = st.spectrum_E_t_fast_norm / st.number_modes
            a_mu_phys = st.E_amp_2_norm_factor * A_mu_norm
    
            P_in_phys_W = st.P_norm * st.P_in_norm_factor
    
            s_in = np.zeros_like(a_mu_phys, dtype=complex)
            s_in[st.pump_idx] = np.sqrt(max(P_in_phys_W, 0.0) / (st.hbar * st.ome_pump))
    
            s_out = s_in - np.sqrt(st.kappa_ex) * a_mu_phys
            P_through_W = st.hbar * st.ome_grid * (np.abs(s_out) ** 2)
            P_through_W = np.maximum(P_through_W, 1e-30)  # avoid log/zeros
    
            # ---------------------------
            # Output folder
            # ---------------------------
            if out_dir is None:
                try:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                except NameError:
                    base_dir = os.getcwd()
                out_dir = os.path.join(base_dir, "pycombs saved data")

            os.makedirs(out_dir, exist_ok=True)

            if ts is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"spectrum_{ts}_j{st.j:09d}_DV{st.DV:+.4f}.csv"
            fpath = os.path.join(out_dir, fname)
    
            data = np.column_stack([
                st.mu.astype(int),
                st.frq_grid,
                st.frq_grid * 1e-12,
                st.wvl_grid,
                st.wvl_grid * 1e9,
                spec_through_dBm,
                P_through_W,
            ])
    
            header = (
                "Snapshot of LLE THROUGH-PORT spectrum\n"
                f"timestamp={ts}, j={st.j}, DV={st.DV}, P_in_W={P_in_phys_W}, kappa_ex={st.kappa_ex}\n"
                "columns: mu, f_Hz, f_THz, lambda_m, lambda_nm, P_through_dBm, P_through_W"
            )
    
            np.savetxt(fpath, data, delimiter=",", header=header, comments="# ")
            print(f"[Snapshot] Saved THROUGH-PORT spectrum to: {fpath}")
    
            # ---------------------------
            # 2) Temporal pulse snapshot (same as before)
            # ---------------------------
            tname = f"pulse_{ts}_j{st.j:09d}_DV{st.DV:+.4f}.csv"
            tpath = os.path.join(out_dir, tname)
    
            I_t = np.abs(st.E_t_fast_norm) ** 2
            tdata = np.column_stack([st.tau_ps, I_t])
    
            theader = (
                "Snapshot of intracavity temporal pulse\n"
                f"timestamp={ts}, j={st.j}, DV={st.DV}, S={st.S}\n"
                "columns: tau_ps, |E(tau)|^2_norm"
            )
    
            np.savetxt(tpath, tdata, delimiter=",", header=theader, comments="# ")
            print(f"[Snapshot] Saved pulse to: {tpath}")
    
            # On success:
            if flash:
                flash_button(ui["btn_save_data"], "Saved!", "Save data")

            return fpath, tpath
    
        except Exception as e:
            print("[Snapshot] FAILED:", repr(e))

            

        
    def save_complex_pump_file(_event=None, export_source="through_port", out_dir=None, ts=None, flash=True):
        """Export the current comb as a reloadable complex pump-input file."""
        try:
            if out_dir is None:
                try:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                except NameError:
                    base_dir = os.getcwd()
                out_dir = os.path.join(base_dir, "pycombs saved pulses")

            os.makedirs(out_dir, exist_ok=True)

            if export_source == "intracavity":
                E_mu = np.asarray(st.spectrum_E_t_fast_norm, dtype=complex).copy()
                source_label = "intracavity_spectrum_E_t_fast_norm"
            else:
                # Physical cascaded-pump choice: through-port/bus comb field.
                A_mu_norm = st.spectrum_E_t_fast_norm / st.number_modes
                a_mu_phys = st.E_amp_2_norm_factor * A_mu_norm

                P_in_phys_W = st.P_norm * st.P_in_norm_factor
                s_in = np.zeros_like(a_mu_phys, dtype=complex)
                s_in[st.pump_idx] = np.sqrt(max(P_in_phys_W, 0.0) / (st.hbar * st.ome_pump))

                E_mu = s_in - np.sqrt(st.kappa_ex) * a_mu_phys
                source_label = "through_port_s_out"

            if not np.any(np.abs(E_mu) > 0):
                print("[Pump export] FAILED: complex field is all zeros.")
                return False

            if ts is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"pulse_complex_{ts}_j{st.j:09d}_DV{st.DV:+.4f}.txt"
            fpath = os.path.join(out_dir, fname)

            with open(fpath, "w", encoding="utf-8") as f:
                f.write("# pycombs complex pump file generated from current comb\n")
                f.write("# AUTHORITATIVE_DOMAIN: frequency\n")
                f.write("# NORMALIZATION: mean_time_power_1\n")
                f.write(f"# SOURCE_FIELD: {source_label}\n")
                f.write(f"# timestamp: {ts}\n")
                f.write(f"# j: {st.j}\n")
                f.write(f"# DV: {st.DV:.12g}\n")
                f.write(f"# P_norm: {st.P_norm:.12g}\n")
                f.write(f"# P_in_phys_W: {st.P_norm * st.P_in_norm_factor:.12g}\n")
                f.write(f"# number_modes: {st.number_modes}\n")
                f.write(f"# fsr_hz: {st.fsr:.12g}\n")
                f.write(f"# pump_wavelength_nm: {st.wvl_pump * 1e9:.12g}\n")
                f.write("\n[FREQ]\n")
                for mu_i, val in zip(st.mu.astype(int), E_mu):
                    f.write(f"{mu_i:d}, {val.real:.17e}, {val.imag:.17e}\n")

            print(f"[Pump export] Saved reloadable complex pump file to: {fpath}")
            print("[Pump export] To use it: copy this path into the Pump input file name box, then click Apply & Reset.")
            if flash:
                if "btn_save_pulse" in ui:
                    flash_button(ui["btn_save_pulse"], "Saved!", "Save pulse")
                else:
                    flash_button(ui["btn_save_pump"], "Saved!", "Save pulse")
            return fpath

        except Exception as e:
            print("[Pump export] FAILED:", repr(e))
            return False


    plot_save_state = {"busy": False, "last_click_time": 0.0}

    def save_visual_plot(out_dir=None, ts=None):
        """Save exactly one current GUI dashboard image as a high-resolution PNG."""
        try:
            if out_dir is None:
                try:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                except NameError:
                    base_dir = os.getcwd()
                out_dir = os.path.join(base_dir, "pycombs saved plots")
            os.makedirs(out_dir, exist_ok=True)
            if ts is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"pycombs_gui_{ts}_n{st.number_modes}.png"
            fpath = os.path.join(out_dir, fname)
            # bbox_inches='tight' is critical to include the sidebar in the image
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {fpath}")
            return fpath
        except Exception as e:
            print(f"Plot save error: {e}")
            return False

    def save_raw_data(out_dir=None, ts=None, flash=True):
        """Calls your existing snapshot logic to save CSV files."""
        try:
            return save_spectrum_snapshot(out_dir=out_dir, ts=ts, flash=flash)
        except Exception as e:
            print(f"Data save error: {e}")
            return False
    
    def on_save_plot_clicked(event):
        now = time.monotonic()
        if plot_save_state["busy"] or (now - plot_save_state["last_click_time"] < 0.75):
            return
        plot_save_state["busy"] = True
        plot_save_state["last_click_time"] = now
        try:
            if save_visual_plot():
                flash_button(ui['btn_save_plot'], "Saved!", "Save plot")
        finally:
            plot_save_state["busy"] = False

    def on_save_data_clicked(event):
        if save_raw_data():
            flash_button(ui['btn_save_data'], "Saved!", "Save data")

    def on_save_pump_clicked(event):
        save_complex_pump_file()


    def on_save_both_clicked(event):
        """Save plot, data CSVs, and reloadable complex pulse/pump file into one folder."""
        try:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                base_dir = os.getcwd()

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(
                base_dir,
                "pycombs saved all",
                f"pycombs_all_{ts}_j{st.j:09d}_DV{st.DV:+.4f}"
            )
            os.makedirs(out_dir, exist_ok=True)

            plot_path = save_visual_plot(out_dir=out_dir, ts=ts)
            spectrum_path, pulse_csv_path = save_raw_data(out_dir=out_dir, ts=ts, flash=False)
            complex_pulse_path = save_complex_pump_file(out_dir=out_dir, ts=ts, flash=False)

            summary_path = os.path.join(out_dir, "readme_saved_all.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("pycombs Save all export\n")
                f.write(f"timestamp: {ts}\n")
                f.write(f"j: {st.j}\n")
                f.write(f"DV: {st.DV}\n")
                f.write(f"P_norm: {st.P_norm}\n")
                f.write(f"number_modes: {st.number_modes}\n")
                f.write(f"fsr_hz: {st.fsr}\n")
                f.write(f"pump_wavelength_nm: {st.wvl_pump * 1e9}\n")
                f.write("\nFiles saved in this folder:\n")
                f.write(f"plot_png: {os.path.basename(plot_path)}\n")
                f.write(f"spectrum_csv: {os.path.basename(spectrum_path)}\n")
                f.write(f"pulse_csv: {os.path.basename(pulse_csv_path)}\n")
                f.write(f"complex_pulse_pump_txt: {os.path.basename(complex_pulse_path)}\n")

            print(f"[Save all] Saved plot, data, and complex pulse file to folder: {out_dir}")
            flash_button(ui['btn_save_both'], "Saved!", "Save all")

        except Exception as e:
            print("[Save all] FAILED:", repr(e))

    
    def import_startup_preset(event):
        try:
            startup_name = ui["txt_startup"].text.strip()
            if not startup_name:
                print("[Startup] Startup preset box is empty.")
                return
            if startup_name not in STARTUP_PRESETS:
                print(f'[Startup] Startup preset not found: "{startup_name}"')
                return

            st.startup_config_name = startup_name
            apply_named_startup_preset(st, startup_name)

            syncing["flag"] = True
            sync_sidebar_from_state(ui, st)
            syncing["flag"] = False

            apply_new_params(None)

            print(f"[Startup] Imported startup preset and applied full runtime state: {startup_name}")
            fig.canvas.draw_idle()

        except Exception as e:
            syncing["flag"] = False
            print(f"[Startup] Import error: {e}")

    def apply_new_params(event):
        # 1. Temporarily pause the simulation
        was_running = running["flag"]
        running["flag"] = False

        try:
            startup_name = ui['txt_startup'].text.strip()
            st.startup_config_name = startup_name

            # 2. Get new values from the current text boxes
            new_n = int(ui['txt_modes'].text)
            st.wvl_pump = float(ui['txt_lambda'].text) * 1e-9
            st.fsr = float(ui['txt_fsr'].text)
            st.Q = float(ui['txt_q'].text)
            st.Aeff = float(ui['txt_aeff'].text) * 1e-12
            st.n2 = float(ui['txt_n2'].text)
            st.eta = float(ui['txt_eta'].text)
            st.dint_file_path = ui['txt_dint'].text.strip()
            st.input_field_file = ui['txt_pump'].text.strip()
            st.noise_level = max(float(ui['txt_noise_level'].text), 0.0)

            # update pump physics
            st.frq_pump = st.c / st.wvl_pump
            st.ome_pump = 2*np.pi * st.frq_pump

            # 3. Complete re-gridding in the LLEState
            st.reinitialize(new_n_modes=new_n)

            # reinitialize() has already restored the selected CW source and,
            # when enabled, reapplied EO modulation. Do not reload the CW here,
            # because doing so would silently clear the EO stage.
            update_noise_button_label()

            # --- Reset core sim targets to the imported/manual startup defaults ---
            runtime_defaults = get_startup_runtime_defaults(st)
            st.detuning_slew_rate = runtime_defaults["slew"]
            st.DV = runtime_defaults["DV"]
            st.target_detuning = st.DV

            # Respect authoritative physical-power presets (e.g. Wilson 2019 = 36 mW).
            if runtime_defaults.get("Pphys") is not None:
                st.P_in_phys = float(runtime_defaults["Pphys"])
                st.P_norm = float(st.P_in_phys / st.P_in_norm_factor)
            else:
                st.P_in_phys = None
                st.P_norm = runtime_defaults["Pnorm"]

            st.P_norm_target = st.P_norm
            st.S = np.sqrt(st.P_norm)
            st.S_target = st.S
            st._rebuild_input_pump()

            # --- Reset GUI widgets (avoid feedback loops) ---
            syncing["flag"] = True

            # Detuning sliders/text
            det_slider.set_val(st.DV)
            det_text_DV.set_val(f"{st.DV:.2f}")

            det_GHz = DV_to_detuning_GHz(st.DV, st.kappa_avg)
            slider_detGHz.set_val(det_GHz)
            det_text_GHz.set_val(f"{det_GHz:.2f}")

            # Slew sliders/text (DV/ns and GHz/ns)
            slider_slew.set_val(st.detuning_slew_rate)
            text_slew.set_val(f"{st.detuning_slew_rate:.2f}")

            slew_GHzns = slew_DVns_to_GHzns(st.detuning_slew_rate, st.kappa_avg)
            slider_slewGHz.set_val(slew_GHzns)
            text_slewGHz.set_val(f"{slew_GHzns:.3f}")

            # Power sliders/text (norm and mW)
            Pnorm_slider.set_val(st.P_norm)
            Pnorm_text.set_val(f"{st.P_norm:.2f}")

            PmW = 1e3 * Pnorm_to_W(st.P_norm, st.P_in_norm_factor)
            PmW_slider.set_val(PmW)
            PmW_text.set_val(f"{PmW:.2f}")

            # Power-slew sliders/text (norm/ns and mW/ns)
            Pslew = float(getattr(st, "power_slew_rate", 100.0))
            Pslew_slider.set_val(Pslew)
            Pslew_text.set_val(f"{Pslew:.3f}")
            Pslew_mW = 1e3 * Pnorm_to_W(Pslew, st.P_in_norm_factor)
            Pslew_mW_slider.set_val(Pslew_mW)
            Pslew_mW_text.set_val(f"{Pslew_mW:.3f}")
            # Fewer ticks for mW (choose a coarser step)
            PmW_tick_step = 5.0
            ui["PmW_inside_ticks"] = add_slider_ticks_inside_step(ax_PmW, PmW_slider, step=PmW_tick_step, y0=0.30, y1=0.70, lw=0.7)

            sync_sidebar_from_state(ui, st)
            syncing["flag"] = False

            # 4. CRITICAL: recompute X axes from new N/FSR, but keep the Y limits
            # exactly as typed in the left axis boxes. Previously this reset path
            # called smart_auto_axes(), so a CW/flat restarted field made Matplotlib
            # shrink the spectrum/pulse/power Y ranges after Apply reset.
            nonlocal x_THz, t_ps
            x_THz, t_ps = sync_x_axes_to_grid(st, ax_spec, ax_pulse, ax_spec_top, pad_frac=0.01)
            apply_manual_axis_boxes()

            # 5. Update plot line/artist data shapes
            spec_line.set_data(x_THz, np.full(new_n, -100))
            stem_sc.set_offsets(np.column_stack([x_THz, np.full(new_n, -300)]))
            pulse_line.set_data(t_ps, np.zeros(new_n))

            # Clear old stem segments so spectrum truly resets visually
            stem_lc.set_segments([])

            # Put the spectrum baseline at the current manual spectrum floor.
            # This preserves custom axis-box values across Apply reset.
            spec_floor = float(ax_spec.get_ylim()[0])
            baseline_state["y"] = spec_floor
            baseline_line.set_ydata([baseline_state["y"], baseline_state["y"]])

            # Force a full redraw (blit-safe)
            fig.canvas.draw()

            # If FuncAnimation blit cache exists, clear it (prevents stale background)
            if hasattr(fig, "_blit_cache"):
                fig._blit_cache.clear()

            # 6. Reset history buffers AND clear the plotted history lines immediately
            nonlocal pow_x, y_Pcav_W, y_DV, y_Pnorm, spectral_time_hist_ns, spectral_db_hist, mode_axis
            pow_x = np.full(Nkeep, np.nan)
            y_Pcav_W = np.full(Nkeep, np.nan)
            y_DV = np.full(Nkeep, np.nan)
            y_Pnorm = np.full(Nkeep, np.nan)
            spectral_time_hist_ns = np.full(Nkeep, np.nan, dtype=float)
            spectral_db_hist = np.full((Nkeep, st.number_modes), np.nan, dtype=float)
            mode_axis = np.arange(st.number_modes) - st.pump_idx
            spectral_map_image.set_data(np.full((2, st.number_modes), np.nan))
            spectral_map_image.set_extent([float(mode_axis[0]), float(mode_axis[-1]), 0.0, WINDOW_NS])
            ax_map.set_xlim(float(mode_axis[0]), float(mode_axis[-1]))
            ax_map.set_ylim(0.0, WINDOW_NS)
            pow_i["i"] = 0

            line_Pcav_W.set_data([], [])
            line_DV.set_data([], [])
            line_Pnorm.set_data([], [])

            # restore a clean time window when paused/reset
            ax_pow.set_xlim(0.0, WINDOW_NS)

            fig.canvas.draw_idle()
            print(f"Simulation restarted with {new_n} modes.")

        except Exception as e:
            print(f"Update error: {e}")

        # 8. Restore previous run state
        running["flag"] = was_running
        
    def apply_live_params(event):
        """Apply runtime-safe sidebar changes without resetting the intracavity field."""
        was_running = running["flag"]
        running["flag"] = False

        try:
            requested_n = int(ui['txt_modes'].text)
            requested_lambda = float(ui['txt_lambda'].text) * 1e-9
            requested_fsr = float(ui['txt_fsr'].text)
            requested_Q = float(ui['txt_q'].text)
            requested_Aeff = float(ui['txt_aeff'].text) * 1e-12
            requested_n2 = float(ui['txt_n2'].text)
            requested_eta = float(ui['txt_eta'].text)

            def changed(a, b, rtol=1e-9, atol=0.0):
                return not np.isclose(float(a), float(b), rtol=rtol, atol=atol)

            blocked_changes = []
            if requested_n != st.number_modes:
                blocked_changes.append("number of modes")
            if changed(requested_lambda, st.wvl_pump):
                blocked_changes.append("pump wavelength")
            if changed(requested_fsr, st.fsr):
                blocked_changes.append("FSR")
            if changed(requested_Q, st.Q):
                blocked_changes.append("Q")
            if changed(requested_Aeff, st.Aeff):
                blocked_changes.append("Aeff")
            if changed(requested_n2, st.n2):
                blocked_changes.append("n2")
            if changed(requested_eta, st.eta):
                blocked_changes.append("eta")

            if blocked_changes:
                print("[Apply live] Not applied: " + ", ".join(blocked_changes) +
                      " require Apply reset so the grid/normalization is rebuilt cleanly.")
                return

            new_noise_level = max(float(ui['txt_noise_level'].text), 0.0)
            if new_noise_level != getattr(st, "noise_level", 1.0):
                st.set_noise_level(new_noise_level, refresh_pump=True)
                ui["txt_noise_level"].set_val(f"{st.noise_level:.3g}")
                print(f"[Apply live] Noise level set to {st.noise_level:.3g}x the default amplitude.")

            new_dint_source = ui['txt_dint'].text.strip()
            if new_dint_source and new_dint_source != getattr(st, "dint_file_path", ""):
                st.dint_file_path = new_dint_source
                st._load_dispersion_with_fallback()
                st.dint_norm = 2 * st.dint / st.kappa_avg
                print(f'[Apply live] Updated dispersion source without reset: "{new_dint_source}"')

            new_pump_source = ui['txt_pump'].text.strip()
            if new_pump_source and new_pump_source != getattr(st, "input_field_file", ""):
                old_source = getattr(st, "input_field_file", "")
                st._load_user_or_default_pump(new_pump_source)
                st._rebuild_input_pump(add_startup_noise=False)
                print(f'[Apply live] Pump changed live: "{old_source}" -> "{st.input_field_file}"')
            else:
                st._rebuild_input_pump(add_startup_noise=False)
                print("[Apply live] Rebuilt current pump without resetting the cavity field.")

            update_noise_button_label()
            sync_sidebar_from_state(ui, st)
            fig.canvas.draw_idle()

            if "btn_apply_live" in ui:
                flash_button(ui["btn_apply_live"], "Applied!", "Apply live")

        except Exception as e:
            print(f"[Apply live] Error: {e}")

        finally:
            running["flag"] = was_running

    def flash_button(btn, temp_text, final_text, delay_ms=500):
        """
        Temporarily change button label, then restore after delay.
        Non-blocking (does not freeze animation).
        """
        btn.label.set_text(temp_text)
        fig.canvas.draw_idle()
    
        timer = fig.canvas.new_timer(interval=delay_ms)
    
        def restore():
            btn.label.set_text(final_text)
            fig.canvas.draw_idle()
    
        timer.add_callback(restore)
        timer.start()

        

    def _nice_upper(value, fallback=1.0):
        value = float(value) if np.isfinite(value) else float(fallback)
        if value <= 0:
            return float(fallback)
        return value * 1.20

    def _axis_float(text, fallback):
        """Read an axis box safely. Keeps animation alive if a box is blank/invalid."""
        try:
            val = float(str(text).strip())
            if np.isfinite(val):
                return val
        except Exception:
            pass
        return float(fallback)

    def _safe_set_ylim(ax, ymin, ymax, name, min_span=1e-12):
        """Set limits only if they are valid, finite and ordered."""
        if not (np.isfinite(ymin) and np.isfinite(ymax)):
            print(f"[Axes] Ignored {name}: limits must be finite numbers.")
            return False
        if ymax <= ymin + min_span:
            print(f"[Axes] Ignored {name}: Y max must be larger than Y min.")
            return False
        ax.set_ylim(float(ymin), float(ymax))
        return True

    def smart_auto_axes(y_db=None, I_t_pulse=None, Pcav_plot=None):
        """Compute and apply smart limits once. Uses physical grids for X and live data for Y.

        This function is intentionally NOT called from the animation loop,
        so blitting stays fast during normal live plotting.
        """

        # X axes: always physical/meaningful.
        ax_pow.set_xlim(0.0, WINDOW_NS)

        # Spectrum Y: robust floor plus headroom.
        if y_db is not None:
            yy = np.asarray(y_db, dtype=float)
            yy = yy[np.isfinite(yy)]
            if yy.size:
                ymax = float(np.nanmax(yy)) + 10.0
                ymin = float(np.nanpercentile(yy, 5)) - 10.0
                ymin = max(ymin, -300.0)
                if ymax <= ymin + 5.0:
                    ymax = ymin + 50.0
                ax_spec.set_ylim(ymin, ymax)
                baseline_state["y"] = ymin
                baseline_line.set_ydata([ymin, ymin])

        # Pulse Y: start at zero and pad the current maximum.
        if I_t_pulse is not None:
            pp = np.asarray(I_t_pulse, dtype=float)
            pp = pp[np.isfinite(pp)]
            if pp.size:
                ax_pulse.set_ylim(0.0, _nice_upper(np.nanmax(pp), fallback=1.0))

        # Power history Y: use current plotted window, padded.
        if Pcav_plot is not None:
            ww = np.asarray(Pcav_plot, dtype=float)
            ww = ww[np.isfinite(ww)]
            if ww.size:
                ax_pow.set_ylim(0.0, _nice_upper(np.nanmax(ww), fallback=1.0))


    def on_fit_axes_clicked(event):
        """Fit all displayed axes once from the current live data."""
        try:
            y_db_now = st.get_output_spectrum_dBm()
            I_t_now = np.abs(st.E_t_fast_norm)**2
            try:
                Pcav_now = np.asarray(line_Pcav_W.get_ydata(), dtype=float)
            except Exception:
                Pcav_now = np.array([0.0])
            smart_auto_axes(y_db=y_db_now, I_t_pulse=I_t_now, Pcav_plot=Pcav_now)
            fig.canvas.draw()
            print("[Axes] Fitted axes once from current live data.")
        except Exception as e:
            print(f"[Axes] Could not fit axes: {e}")

    def on_noise_level_submit(text):
        if syncing["flag"]:
            return
        try:
            level = max(float(str(text).strip()), 0.0)
            st.set_noise_level(level, refresh_pump=True)
            ui["txt_noise_level"].set_val(f"{st.noise_level:.3g}")
            print(f"[Noise] Noise level set to {st.noise_level:.3g}x the default amplitude.")
        except Exception as e:
            print(f"[Noise] Invalid noise level; keeping {getattr(st, 'noise_level', 1.0):.3g}. Details: {e}")
            ui["txt_noise_level"].set_val(f"{getattr(st, 'noise_level', 1.0):.3g}")

    def on_load_recipe_clicked(event=None):
        try:
            recipe_runner.load(ui["txt_recipe"].text)
            running["flag"] = True
            ui["btn_run"].label.set_text("PAUSE")
            fig.canvas.draw_idle()
        except Exception as e:
            recipe_runner.fail(str(e))

    def sync_recipe_driven_controls_from_state(force=False):
        """Keep sliders and numeric boxes synced when a recipe changes targets."""
        if syncing["flag"]:
            return
        if (not force) and (not getattr(recipe_runner, "ui_sync_needed", False)):
            return
        syncing["flag"] = True
        try:
            DV_target = float(getattr(st, "target_detuning", getattr(st, "DV", det_slider.val)))
            det_GHz = DV_to_detuning_GHz(DV_target, st.kappa_avg)
            det_slider.set_val(DV_target)
            slider_detGHz.set_val(det_GHz)
            det_text_DV.set_val(f"{DV_target:.3f}")
            det_text_GHz.set_val(f"{det_GHz:.3f}")

            Pn_target = float(getattr(st, "P_norm_target", getattr(st, "P_norm", Pnorm_slider.val)))
            PmW = 1e3 * Pnorm_to_W(Pn_target, st.P_in_norm_factor)
            Pnorm_slider.set_val(Pn_target)
            PmW_slider.set_val(PmW)
            Pnorm_text.set_val(f"{Pn_target:.2f}")
            PmW_text.set_val(f"{PmW:.2f}")

            slew_DVns = float(getattr(st, "detuning_slew_rate", slider_slew.val))
            slew_GHzns = slew_DVns_to_GHzns(slew_DVns, st.kappa_avg)
            slider_slew.set_val(slew_DVns)
            slider_slewGHz.set_val(slew_GHzns)
            text_slew.set_val(f"{slew_DVns:.3f}")
            text_slewGHz.set_val(f"{slew_GHzns:.3f}")

            power_slew = float(getattr(st, "power_slew_rate", Pslew_slider.val))
            power_slew_mW = 1e3 * Pnorm_to_W(power_slew, st.P_in_norm_factor)
            Pslew_slider.set_val(power_slew)
            Pslew_mW_slider.set_val(power_slew_mW)
            Pslew_text.set_val(f"{power_slew:.3f}")
            Pslew_mW_text.set_val(f"{power_slew_mW:.3f}")
        finally:
            recipe_runner.ui_sync_needed = False
            syncing["flag"] = False

    # Bind recipe actions after all GUI callbacks/helpers exist.
    # The lambda wrappers keep the recipe runner independent from Matplotlib event objects.
    recipe_runner.bind_actions(
        set_all_noise=set_all_noise,
        apply_live=lambda: apply_live_params(None),
        apply_reset=lambda: apply_new_params(None),
        save_plot=lambda: save_visual_plot(),
        save_data=lambda: save_raw_data(flash=False),
        save_pulse=lambda: save_complex_pump_file(flash=False),
        save_all=lambda: on_save_both_clicked(None),
    )

    # ---- Hook up sidebar buttons ----
    
    ui["btn_import_startup"].on_clicked(import_startup_preset)
    ui["btn_apply_live"].on_clicked(apply_live_params)
    ui["btn_apply"].on_clicked(apply_new_params)

    
    ui["btn_save_plot"].on_clicked(on_save_plot_clicked)
    ui["btn_save_data"].on_clicked(on_save_data_clicked)
    ui["btn_save_pump"].on_clicked(on_save_pump_clicked)
    ui["btn_save_both"].on_clicked(on_save_both_clicked)
    
    ui["btn_run"].on_clicked(on_run_clicked)
    ui["btn_noise"].on_clicked(on_noise_clicked)
    ui["btn_eo_comb"].on_clicked(open_eo_comb_window)
    update_eo_button_label()
    ui["txt_noise_level"].on_submit(on_noise_level_submit)
    ui["btn_record"].on_clicked(on_record_clicked)
    ui["btn_load_recipe"].on_clicked(on_load_recipe_clicked)

    ui["btn_axis_fit"].on_clicked(on_fit_axes_clicked)

    
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    
    # ---- Animation update ----
    def _update_body(_frame):
        if not running["flag"]:
                ax_pow.set_xlim(0.0, WINDOW_NS)
                recipe_runner.advance()
                sync_recipe_driven_controls_from_state()
                record_current_frame()
                return stem_lc, stem_sc, baseline_line, pulse_line, spectral_map_image, line_Pcav_W, line_DV, line_Pnorm, hud, recipe_status_text, recipe_status_text_sidebar

    
        # 1) advance simulation and then advance any loaded recipe
        st.step(n_steps=steps_per_frame)
        recipe_runner.advance()
        sync_recipe_driven_controls_from_state()
        
        # ---- Physical readouts ----
        det_GHz   = st.DV * st.kappa_avg / (2*np.pi) * 1e-9           # GHz
        Pin_mW    = 1e3 * (st.P_norm * st.P_in_norm_factor)            # mW

        
        # ---- Update temporal pulse ----
        I_t = np.abs(st.E_t_fast_norm)**2
        
        if st.temporal_interpol:
            print("interpolating")
            ##interpolating the temporal fast time to make temporal pulse profile graph prittier
            interp_func = interp1d(t_ps, I_t, kind='cubic')
            t_fine = np.linspace(t_ps[0], t_ps[-1], len(t_ps) * 10)
            t_pulse = t_fine
            I_t_fine = interp_func(t_fine)
            I_t_pulse = I_t_fine
        else:
            t_pulse = t_ps
            I_t_pulse = I_t
        pulse_line.set_xdata(t_pulse)
        pulse_line.set_ydata(I_t_pulse)
    
        # 2) spectrum (OSA-like, wavelength axis)
        y_db = st.get_output_spectrum_dBm()
        
        b = baseline_state["y"]
        baseline_line.set_ydata([b, b])
        
        # Keep “floor at baseline” style if you want
        y_plot = np.maximum(y_db, b)
        
        mask = y_db > b
        segments = [((x, b), (x, y)) for x, y in zip(x_THz[mask], y_plot[mask])]
        stem_lc.set_segments(segments)
        
        # dots for all modes
        stem_sc.set_offsets(np.column_stack([x_THz, y_plot]))
        spec_line.set_data(x_THz, y_db)



        # Keep main title static; update only the subtitle text
        t_phys_ns = 1e9 * (2.0 / st.kappa_avg) * (st.j * st.tal_step)
        
        # ---- Power plot (3 traces ONLY) ----
        #j_now = float(st.j)
        
        # Physical slow time [µs]
        t_phys_ns = 1e9 * (2.0 / st.kappa_avg) * (st.j * st.tal_step)
        

        # Pcav_norm = float(st.get_intracavity_mean())   # ⟨|E|²⟩ directly from field
        # Pcav_mW   = Pcav_norm_to_mW(Pcav_norm)         # consistent physical conversion
        
        # --- Physical intracavity circulating power (from modal amplitudes) ---
        
        # Normalized mode amplitudes (consistent with get_output_spectrum_dBm())
        A_mu_norm = st.spectrum_E_t_fast_norm / st.number_modes
        
        # Convert to physical intracavity mode amplitudes
        a_mu_phys = st.E_amp_2_norm_factor * A_mu_norm
        
        # Intracavity energy stored across all modes [J]
        U_J = np.sum(st.hbar * st.ome_grid * (np.abs(a_mu_phys) ** 2))
        
        # Circulating power [W] = energy / round-trip time
        Pcav_W = U_J / st.t_phys_round_trip
        Pcav_mW = 1e3 * Pcav_W
        
        # Normalized mean intensity (your original meaning)
        Pcav_norm = float(np.mean(np.abs(st.E_t_fast_norm) ** 2))


        
        DV_now    = float(st.DV)
        Pnorm_now = float(st.P_norm)
        
        slew_GHzns = slew_DVns_to_GHzns(st.detuning_slew_rate, st.kappa_avg)
        
        recipe_status = getattr(recipe_runner, "_last_status", "Recipe: idle")
        hud.set_text(
            f"t_sim: {st.j * st.tal_step:.2f} s   |   t_phys: {t_phys_ns:.2f} ns\n"
            f"Detuning: {DV_now:.2f} DV   |   {DV_to_detuning_GHz(DV_now, st.kappa_avg):.2f} GHz\n"
            f"Det. rate: {st.detuning_slew_rate:.2f} DV/ns   |   {slew_GHzns:.2f} GHz/ns\n"
            f"Pump: {Pnorm_now:.2f} norm   |   {Pin_mW:.2f} mW\n"
            f"Cavity: {Pcav_norm:.2f} norm   |   {Pcav_mW:.2f} mW\n"
            f"{recipe_status}"
        )

        k = pow_i["i"] % Nkeep
        #pow_x[k]    = j_now
        pow_x[k] = t_phys_ns
        spectral_time_hist_ns[k] = t_phys_ns
        spectral_db_hist[k, :] = np.asarray(y_db, dtype=float)
        y_Pcav_W[k] = Pcav_norm
        y_DV[k]     = DV_now
        y_Pnorm[k]  = Pnorm_now
        pow_i["i"] += 1

        # unwrap ring buffer
        if pow_i["i"] < Nkeep:
            x_plot    = pow_x[:pow_i["i"]]
            Pcav_plot = y_Pcav_W[:pow_i["i"]]
            DV_plot   = y_DV[:pow_i["i"]]
            Pn_plot   = y_Pnorm[:pow_i["i"]]
        else:
            order     = np.arange(k + 1, k + 1 + Nkeep) % Nkeep
            x_plot    = pow_x[order]
            Pcav_plot = y_Pcav_W[order]
            DV_plot   = y_DV[order]
            Pn_plot   = y_Pnorm[order]

        

        # Rebase displayed history so the visible axis always runs from 0 to 10 ns
        if np.isfinite(x_plot).any():
            xmax = float(np.nanmax(x_plot))
            x_start = max(0.0, xmax - WINDOW_NS)
            x_plot_disp = x_plot - x_start
        else:
            x_plot_disp = x_plot

        # ---- Spectral evolution map below the intracavity power plot ----
        if pow_i["i"] < Nkeep:
            spec_times = spectral_time_hist_ns[:pow_i["i"]]
            spec_data = spectral_db_hist[:pow_i["i"], :]
        else:
            spec_times = spectral_time_hist_ns[order]
            spec_data = spectral_db_hist[order, :]

        if np.isfinite(spec_times).any():
            spec_xmax = float(np.nanmax(spec_times))
            spec_start = max(0.0, spec_xmax - WINDOW_NS)
            valid = np.isfinite(spec_times) & (spec_times >= spec_start)
            spec_data = spec_data[valid, :]
            if spec_data.size > 0:
                spectral_map_image.set_data(np.clip(spec_data, -80, 0))
                spectral_map_image.set_extent([float(mode_axis[0]), float(mode_axis[-1]), 0.0, WINDOW_NS])
                ax_map.set_xlim(float(mode_axis[0]), float(mode_axis[-1]))
                ax_map.set_ylim(0.0, WINDOW_NS)

        # set data
        line_Pcav_W.set_data(x_plot_disp, Pcav_plot)
        line_DV.set_data(x_plot_disp, DV_plot)
        line_Pnorm.set_data(x_plot_disp, Pn_plot)

        # Axis limits are intentionally not updated here.
        # Use the left-panel "Update axes" button for a one-shot redraw,
        # keeping normal animation blitting fast.

        record_current_frame()

        return stem_lc, stem_sc, baseline_line, pulse_line, spectral_map_image, line_Pcav_W, line_DV, line_Pnorm, hud, recipe_status_text, recipe_status_text_sidebar

    def update(_frame):
        try:
            return _update_body(_frame)
        except Exception as e:
            print(f"[Update] Animation update error kept alive: {e}")
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass
            return stem_lc, stem_sc, baseline_line, pulse_line, spectral_map_image, line_Pcav_W, line_DV, line_Pnorm, hud, recipe_status_text, recipe_status_text_sidebar

    # Keep the few non-grid artists aligned after manual window resizing.
    fig.canvas.mpl_connect("resize_event", reposition_spectral_colorbar)
    fig.canvas.mpl_connect("resize_event", apply_responsive_fonts)
    reposition_spectral_colorbar()
    apply_responsive_fonts()

    ani = animation.FuncAnimation(fig, update, interval=interval_ms, blit=True, cache_frame_data=False)

    fig._ani = ani  # keep alive in Spyder
    fig._ui = ui 
    
    anim_ref["ani"] = ani
    

    global APP
    APP = pycombsApp()
    APP.fig = fig
    APP.ui = ui
    APP.st = st
    APP.ani = ani


    plt.show()




# ============================================================
# EMBEDDED DEFAULTS
# ============================================================

DEFAULT_DINT_HERR_MGF2_CSV = '''Wavelength [um],Frequency [THz],Dint/2pi [GHz] 
1.6117799427830857,186.0008615582743,0.32
1.611474976888244,186.0360615582743,0.316808
1.6111701263771472,186.0712615582743,0.313632
1.6108653911843245,186.1064615582743,0.310472
1.610560771244355,186.1416615582743,0.307328
1.610256266491867,186.1768615582743,0.3042
1.6099518768615382,186.2120615582743,0.301088
1.6096476022880952,186.24726155827432,0.297992
1.6093434427063151,186.28246155827432,0.294912
1.6090393980510234,186.3176615582743,0.291848
1.608735468257095,186.3528615582743,0.2888
1.6084316532594538,186.3880615582743,0.285768
1.6081279529930737,186.4232615582743,0.282752
1.6078243673929764,186.4584615582743,0.279752
1.607520896394234,186.4936615582743,0.276768
1.6072175399319664,186.52886155827431,0.2738
1.6069142979413438,186.56406155827432,0.270848
1.6066111703575838,186.59926155827432,0.267912
1.606308157115954,186.6344615582743,0.264992
1.6060052581517708,186.6696615582743,0.262088
1.6057024734003984,186.7048615582743,0.2592
1.6053998027972505,186.7400615582743,0.256328
1.6050972462777895,186.7752615582743,0.253472
1.6047948037775266,186.8104615582743,0.250632
1.6044924752320209,186.84566155827432,0.247808
1.6041902605768805,186.88086155827432,0.245
1.603888159747762,186.91606155827432,0.242208
1.6035861726803706,186.9512615582743,0.239432
1.6032842993104595,186.9864615582743,0.236672
1.6029825395738306,187.0216615582743,0.233928
1.602680893406334,187.0568615582743,0.2312
1.602379360743868,187.0920615582743,0.228488
1.6020779415223794,187.1272615582743,0.225792
1.601776635677863,187.16246155827432,0.223112
1.6014754431463618,187.19766155827432,0.220448
1.6011743638639666,187.23286155827432,0.2178
1.6008733977668168,187.2680615582743,0.215168
1.6005725447910992,187.3032615582743,0.212552
1.6002718048730493,187.3384615582743,0.209952
1.5999711779489498,187.3736615582743,0.207368
1.5996706639551317,187.4088615582743,0.2048
1.5993702628279731,187.4440615582743,0.202248
1.5990699745039016,187.47926155827432,0.199712
1.5987697989193905,187.51446155827432,0.197192
1.598469736010962,187.5496615582743,0.194688
1.5981697857151855,187.5848615582743,0.1922
1.5978699479686782,187.6200615582743,0.189728
1.597570222708105,187.6552615582743,0.187272
1.5972706098701779,187.6904615582743,0.184832
1.5969711093916565,187.7256615582743,0.182408
1.5966717212093484,187.7608615582743,0.18
1.5963724452601074,187.79606155827432,0.177608
1.5960732814808356,187.83126155827432,0.175232
1.5957742298084823,187.8664615582743,0.172872
1.5954752901800433,187.9016615582743,0.170528
1.5951764625325628,187.9368615582743,0.1682
1.5948777468031312,187.9720615582743,0.165888
1.5945791429288863,188.0072615582743,0.163592
1.594280650847013,188.0424615582743,0.161312
1.5939822704947435,188.07766155827431,0.159048
1.593684001809356,188.11286155827432,0.1568
1.5933858447281772,188.14806155827432,0.154568
1.5930877991885792,188.1832615582743,0.152352
1.592789865127982,188.2184615582743,0.150152
1.5924920424838516,188.2536615582743,0.147968
1.5921943311937015,188.2888615582743,0.1458
1.5918967311950911,188.3240615582743,0.143648
1.5915992424256273,188.3592615582743,0.141512
1.5913018648229633,188.39446155827432,0.139392
1.591004598324799,188.42966155827432,0.137288
1.59070744286888,188.46486155827432,0.1352
1.5904103983929996,188.5000615582743,0.133128
1.590113464834997,188.5352615582743,0.131072
1.589816642132758,188.5704615582743,0.129032
1.5895199302242147,188.6056615582743,0.127008
1.589223329047345,188.6408615582743,0.125
1.5889268385401738,188.6760615582743,0.123008
1.588630458640772,188.71126155827432,0.121032
1.5883341892872567,188.74646155827432,0.119072
1.5880380304177915,188.78166155827432,0.117128
1.587741981970585,188.8168615582743,0.1152
1.587446043883893,188.8520615582743,0.113288
1.5871502160960171,188.8872615582743,0.111392
1.5868544985453046,188.9224615582743,0.109512
1.5865588911701491,188.9576615582743,0.107648
1.5862633939089894,188.9928615582743,0.1058
1.585968006700311,189.02806155827432,0.103968
1.5856727294826447,189.06326155827432,0.102152
1.5853775621945672,189.0984615582743,0.100352
1.585082504774701,189.1336615582743,0.098568
1.584787557161714,189.1688615582743,0.0968
1.58449271929432,189.2040615582743,0.095048
1.5841979911112787,189.2392615582743,0.093312
1.5839033725513947,189.2744615582743,0.091592
1.5836088635535186,189.3096615582743,0.089888
1.5833144640565462,189.34486155827432,0.0882
1.5830201739994185,189.38006155827432,0.086528
1.5827259933211228,189.4152615582743,0.084872
1.582431921960691,189.4504615582743,0.083232
1.5821379598572,189.4856615582743,0.081608
1.581844106949773,189.5208615582743,0.08
1.5815503631775776,189.5560615582743,0.078408
1.581256728479827,189.5912615582743,0.076832
1.580963202795779,189.62646155827431,0.075272
1.580669786064737,189.66166155827432,0.073728
1.5803764782260494,189.69686155827432,0.0722
1.5800832792191093,189.7320615582743,0.070688
1.5797901889833554,189.7672615582743,0.069192
1.5794972074582705,189.8024615582743,0.067712
1.5792043345833828,189.8376615582743,0.066248
1.5789115702982652,189.8728615582743,0.0648
1.5786189145425353,189.9080615582743,0.063368
1.578326367255856,189.94326155827432,0.061952
1.5780339283779343,189.97846155827432,0.060552
1.577741597848522,190.01366155827432,0.059168
1.5774493756074157,190.0488615582743,0.0578
1.5771572615944565,190.0840615582743,0.056448
1.57686525574953,190.1192615582743,0.055112
1.5765733580125663,190.1544615582743,0.053792
1.5762815683235405,190.1896615582743,0.052488
1.575989886622471,190.2248615582743,0.0512
1.575698312849422,190.26006155827432,0.049928
1.5754068469445008,190.29526155827432,0.048672
1.5751154888478598,190.33046155827432,0.047432
1.574824238499695,190.3656615582743,0.046208
1.5745330958402475,190.4008615582743,0.045
1.5742420608098016,190.4360615582743,0.043808
1.5739511333486869,190.4712615582743,0.042632
1.5736603133972755,190.5064615582743,0.041472
1.5733696008959854,190.5416615582743,0.040328
1.5730789957852773,190.57686155827432,0.0392
1.5727884980056566,190.61206155827432,0.038088
1.5724981074976718,190.6472615582743,0.036992
1.5722078242019164,190.6824615582743,0.035912
1.5719176480590267,190.7176615582743,0.034848
1.5716275790096836,190.7528615582743,0.0338
1.5713376169946114,190.7880615582743,0.032768
1.5710477619545784,190.8232615582743,0.031752
1.5707580138303963,190.8584615582743,0.030752
1.5704683725629207,190.89366155827432,0.029768
1.5701788380930504,190.92886155827432,0.0288
1.569889410361728,190.9640615582743,0.027848
1.5696000893099404,190.9992615582743,0.026912
1.5693108748787163,191.0344615582743,0.025992
1.56902176700913,191.0696615582743,0.025088
1.5687327656422974,191.1048615582743,0.0242
1.5684438707193782,191.1400615582743,0.023328
1.5681550821815764,191.17526155827431,0.022472
1.567866399970138,191.21046155827432,0.021632
1.567577824026353,191.24566155827432,0.020808
1.5672893542915547,191.2808615582743,0.02
1.567000990707119,191.3160615582743,0.019208
1.5667127332144652,191.3512615582743,0.018432
1.5664245817550562,191.3864615582743,0.017672
1.5661365362703974,191.4216615582743,0.016928
1.565848596702037,191.4568615582743,0.0162
1.5655607629915667,191.49206155827432,0.015488
1.565273035080621,191.52726155827432,0.014792
1.5649854129108773,191.56246155827432,0.014112
1.5646978964240559,191.5976615582743,0.013448
1.5644104855619194,191.6328615582743,0.0128
1.564123180266274,191.6680615582743,0.012168
1.563835980478968,191.7032615582743,0.011552
1.5635488861418931,191.7384615582743,0.010952
1.563261897196983,191.7736615582743,0.010368
1.562975013586214,191.80886155827432,0.0098
1.5626882352516054,191.84406155827432,0.009248
1.5624015621352187,191.87926155827432,0.008712
1.5621149941791586,191.9144615582743,0.008192
1.561828531325571,191.9496615582743,0.007688
1.561542173516646,191.9848615582743,0.0072
1.561255920694614,192.0200615582743,0.006728
1.5609697728017493,192.0552615582743,0.006272
1.5606837297803682,192.0904615582743,0.005832
1.5603977915728289,192.12566155827432,0.005408
1.5601119581215321,192.16086155827432,0.005
1.559826229368921,192.1960615582743,0.004608
1.5595406052574796,192.2312615582743,0.004232
1.5592550857297358,192.2664615582743,0.003872
1.558969670728259,192.3016615582743,0.003528
1.5586843601956597,192.3368615582743,0.0032
1.5583991540745918,192.3720615582743,0.002888
1.5581140523077504,192.4072615582743,0.002592
1.5578290548378722,192.44246155827432,0.002312
1.5575441616077363,192.47766155827432,0.002048
1.5572593725601642,192.5128615582743,0.0018
1.556974687638018,192.5480615582743,0.001568
1.5566901067842023,192.5832615582743,0.001352
1.5564056299416633,192.6184615582743,0.001152
1.556121257053389,192.6536615582743,0.000968
1.5558369880624088,192.6888615582743,0.0008
1.555552822911794,192.72406155827431,0.000648
1.5552687615446574,192.75926155827432,0.000512
1.5549848039041532,192.79446155827432,0.000392
1.5547009499334772,192.8296615582743,0.000288
1.5544171995758667,192.8648615582743,0.0002
1.5541335527746005,192.9000615582743,0.000128
1.5538500094729986,192.9352615582743,7.2e-05
1.5535665696144225,192.9704615582743,3.2e-05
1.553283233142275,193.0056615582743,8e-06
1.553,193.04086155827432,0.0
1.552716870131083,193.07606155827432,8e-06
1.5524338434790506,193.11126155827432,3.2e-05
1.5521509199874701,193.1464615582743,7.2e-05
1.5518680995999505,193.1816615582743,0.000128
1.551585382260142,193.2168615582743,0.0002
1.5513027679117353,193.2520615582743,0.000288
1.5510202564984623,193.2872615582743,0.000392
1.550737847964096,193.3224615582743,0.000512
1.5504555422524504,193.35766155827432,0.000648
1.5501733393073804,193.39286155827432,0.0008
1.5498912390727815,193.42806155827432,0.000968
1.54960924149259,193.4632615582743,0.001152
1.5493273465107835,193.4984615582743,0.001352
1.5490455540713801,193.5336615582743,0.001568
1.5487638641184387,193.5688615582743,0.0018
1.5484822765960582,193.6040615582743,0.002048
1.5482007914483793,193.6392615582743,0.002312
1.5479194086195822,193.67446155827432,0.002592
1.5476381280538887,193.70966155827432,0.002888
1.5473569496955606,193.7448615582743,0.0032
1.5470758734888999,193.7800615582743,0.003528
1.5467948993782492,193.8152615582743,0.003872
1.5465140273079925,193.8504615582743,0.004232
1.546233257222553,193.8856615582743,0.004608
1.5459525890663943,193.9208615582743,0.005
1.5456720227840213,193.9560615582743,0.005408
1.545391558319978,193.99126155827432,0.005832
1.5451111956188497,194.02646155827432,0.006272
1.544830934625261,194.0616615582743,0.006728
1.5445507752838774,194.0968615582743,0.0072
1.544270717539404,194.1320615582743,0.007688
1.5439907613365862,194.1672615582743,0.008192
1.5437109066202095,194.2024615582743,0.008712
1.5434311533350993,194.2376615582743,0.009248
1.5431515014261215,194.27286155827431,0.0098
1.5428719508381807,194.30806155827432,0.010368
1.542592501516223,194.34326155827432,0.010952
1.5423131534052335,194.3784615582743,0.011552
1.5420339064502369,194.4136615582743,0.012168
1.5417547605962985,194.4488615582743,0.0128
1.5414757157885228,194.4840615582743,0.013448
1.5411967719720538,194.5192615582743,0.014112
1.5409179290920763,194.5544615582743,0.014792
1.5406391870938134,194.58966155827432,0.015488
1.540360545922529,194.62486155827432,0.0162
1.5400820055235254,194.66006155827432,0.016928
1.5398035658421456,194.6952615582743,0.017672
1.5395252268237716,194.7304615582743,0.018432
1.539246988413825,194.7656615582743,0.019208
1.5389688505577663,194.8008615582743,0.02
1.5386908132010964,194.8360615582743,0.020808
1.538412876289355,194.8712615582743,0.021632
1.5381350397681208,194.90646155827432,0.022472
1.537857303583013,194.94166155827432,0.023328
1.5375796676796882,194.97686155827432,0.0242
1.5373021320038442,195.0120615582743,0.025088
1.5370246965012166,195.0472615582743,0.025992
1.536747361117581,195.0824615582743,0.026912
1.5364701257987519,195.1176615582743,0.027848
1.5361929904905822,195.1528615582743,0.0288
1.535915955138965,195.1880615582743,0.029768
1.5356390196898317,195.22326155827432,0.030752
1.5353621840891531,195.25846155827432,0.031752
1.5350854482829386,195.2936615582743,0.032768
1.5348088122172363,195.3288615582743,0.0338
1.534532275838134,195.3640615582743,0.034848
1.5342558390917578,195.3992615582743,0.035912
1.5339795019242726,195.4344615582743,0.036992
1.533703264281882,195.4696615582743,0.038088
1.5334271261108288,195.5048615582743,0.0392
1.533151087357394,195.54006155827432,0.040328
1.5328751479678975,195.57526155827432,0.041472
1.5325993078886981,195.6104615582743,0.042632
1.5323235670661928,195.6456615582743,0.043808
1.5320479254468173,195.6808615582743,0.045
1.5317723829770455,195.7160615582743,0.046208
1.5314969396033908,195.7512615582743,0.047432
1.531221595272404,195.7864615582743,0.048672
1.5309463499306748,195.82166155827431,0.049928
1.530671203524831,195.85686155827432,0.0512
1.5303961560015398,195.89206155827432,0.052488
1.530121207307505,195.9272615582743,0.053792
1.52984635738947,195.9624615582743,0.055112
1.529571606194216,195.9976615582743,0.056448
1.5292969536685626,196.0328615582743,0.0578
1.5290223997593675,196.0680615582743,0.059168
1.5287479444135266,196.1032615582743,0.060552
1.5284735875779736,196.13846155827432,0.061952
1.5281993291996807,196.17366155827432,0.063368
1.527925169225658,196.20886155827432,0.0648
1.5276511076029538,196.2440615582743,0.066248
1.5273771442786541,196.2792615582743,0.067712
1.5271032791998824,196.3144615582743,0.069192
1.5268295123138018,196.3496615582743,0.070688
1.5265558435676114,196.3848615582743,0.0722
1.5262822729085488,196.4200615582743,0.073728
1.52600880028389,196.45526155827432,0.075272
1.525735425640948,196.49046155827432,0.076832
1.525462148927074,196.52566155827432,0.078408
1.5251889700896568,196.5608615582743,0.08
'''

EMBEDDED_DINT_WILSON2019_CSV = r'''Wavelength [um], Frequency [THz], Dint/2pi [GHz]
2.120000, 141.411537, -70.000000
2.117840, 141.555786, -64.348888
2.115684, 141.700035, -58.970115
2.113532, 141.844285, -53.853410
2.111385, 141.988534, -48.988614
2.109242, 142.132783, -44.365683
2.107104, 142.277033, -39.974686
2.104970, 142.421282, -35.805802
2.102840, 142.565532, -31.849324
2.100714, 142.709781, -28.095651
2.098593, 142.854030, -24.535295
2.096476, 142.998280, -21.158870
2.094363, 143.142529, -17.957101
2.092255, 143.286778, -14.920815
2.090151, 143.431028, -12.040946
2.088051, 143.575277, -9.308530
2.085955, 143.719526, -6.714705
2.083863, 143.863776, -4.250710
2.081776, 144.008025, -1.907886
2.079693, 144.152274, 0.322328
2.077614, 144.296524, 2.448396
2.075539, 144.440773, 4.478682
2.073468, 144.585022, 6.421455
2.071402, 144.729272, 8.284889
2.069339, 144.873521, 10.077062
2.067281, 145.017770, 11.805960
2.065227, 145.162020, 13.479476
2.063176, 145.306269, 15.105410
2.061130, 145.450518, 16.691473
2.059088, 145.594768, 18.245286
2.057050, 145.739017, 19.774379
2.055016, 145.883266, 21.286195
2.052986, 146.027516, 22.788090
2.050960, 146.171765, 24.287334
2.048938, 146.316014, 25.790913
2.046920, 146.460264, 27.301704
2.044906, 146.604513, 28.818795
2.042896, 146.748762, 30.341139
2.040890, 146.893012, 31.867703
2.038888, 147.037261, 33.397463
2.036889, 147.181510, 34.929408
2.034895, 147.325760, 36.462541
2.032905, 147.470009, 37.995874
2.030918, 147.614258, 39.528430
2.028935, 147.758508, 41.059245
2.026957, 147.902757, 42.587367
2.024982, 148.047006, 44.111854
2.023010, 148.191256, 45.631773
2.021043, 148.335505, 47.146206
2.019080, 148.479754, 48.654243
2.017120, 148.624004, 50.154985
2.015164, 148.768253, 51.647546
2.013212, 148.912502, 53.131047
2.011264, 149.056752, 54.604622
2.009319, 149.201001, 56.067414
2.007379, 149.345250, 57.518577
2.005442, 149.489500, 58.957274
2.003508, 149.633749, 60.382680
2.001579, 149.777998, 61.793978
1.999653, 149.922248, 63.190361
1.997731, 150.066497, 64.571034
1.995812, 150.210746, 65.935209
1.993898, 150.354996, 67.282108
1.991986, 150.499245, 68.610963
1.990079, 150.643495, 69.921016
1.988175, 150.787744, 71.211518
1.986275, 150.931993, 72.481727
1.984379, 151.076243, 73.730912
1.982486, 151.220492, 74.958353
1.980596, 151.364741, 76.163334
1.978711, 151.508991, 77.345153
1.976829, 151.653240, 78.503112
1.974950, 151.797489, 79.636525
1.973075, 151.941739, 80.744714
1.971204, 152.085988, 81.827009
1.969336, 152.230237, 82.882747
1.967471, 152.374487, 83.911276
1.965611, 152.518736, 84.911951
1.963753, 152.662985, 85.884134
1.961900, 152.807235, 86.827197
1.960049, 152.951484, 87.740520
1.958203, 153.095733, 88.623488
1.956359, 153.239983, 89.475498
1.954519, 153.384232, 90.295952
1.952683, 153.528481, 91.084260
1.950850, 153.672731, 91.839840
1.949021, 153.816980, 92.562118
1.947194, 153.961229, 93.250528
1.945372, 154.105479, 93.904508
1.943553, 154.249728, 94.523509
1.941737, 154.393977, 95.106983
1.939924, 154.538227, 95.654394
1.938115, 154.682476, 96.165211
1.936309, 154.826725, 96.638910
1.934507, 154.970975, 97.074974
1.932708, 155.115224, 97.472895
1.930913, 155.259473, 97.832168
1.929120, 155.403723, 98.152325
1.927331, 155.547972, 98.433538
1.925546, 155.692221, 98.676631
1.923763, 155.836471, 98.882447
1.921984, 155.980720, 99.051822
1.920208, 156.124969, 99.185583
1.918436, 156.269219, 99.284551
1.916667, 156.413468, 99.349536
1.914901, 156.557717, 99.381342
1.913138, 156.701967, 99.380766
1.911378, 156.846216, 99.348594
1.909622, 156.990465, 99.285608
1.907869, 157.134715, 99.192581
1.906119, 157.278964, 99.070277
1.904373, 157.423213, 98.919454
1.902629, 157.567463, 98.740863
1.900889, 157.711712, 98.535245
1.899152, 157.855961, 98.303338
1.897418, 158.000211, 98.045869
1.895687, 158.144460, 97.763559
1.893960, 158.288709, 97.457122
1.892235, 158.432959, 97.127266
1.890514, 158.577208, 96.774690
1.888796, 158.721457, 96.400088
1.887081, 158.865707, 96.004145
1.885369, 159.009956, 95.587541
1.883660, 159.154206, 95.150950
1.881955, 159.298455, 94.695036
1.880252, 159.442704, 94.220459
1.878552, 159.586954, 93.727872
1.876856, 159.731203, 93.217921
1.875163, 159.875452, 92.691246
1.873472, 160.019702, 92.148481
1.871785, 160.163951, 91.590252
1.870101, 160.308200, 91.017181
1.868419, 160.452450, 90.429881
1.866741, 160.596699, 89.828962
1.865066, 160.740948, 89.215025
1.863394, 160.885198, 88.588666
1.861724, 161.029447, 87.950477
1.860058, 161.173696, 87.301041
1.858395, 161.317946, 86.640936
1.856735, 161.462195, 85.970734
1.855077, 161.606444, 85.291003
1.853423, 161.750694, 84.602304
1.851772, 161.894943, 83.905191
1.850123, 162.039192, 83.200214
1.848478, 162.183442, 82.487916
1.846835, 162.327691, 81.768837
1.845195, 162.471940, 81.043509
1.843558, 162.616190, 80.312459
1.841925, 162.760439, 79.576209
1.840294, 162.904688, 78.835276
1.838666, 163.048938, 78.090172
1.837040, 163.193187, 77.341401
1.835418, 163.337436, 76.589466
1.833798, 163.481686, 75.834860
1.832182, 163.625935, 75.078076
1.830568, 163.770184, 74.319597
1.828957, 163.914434, 73.559906
1.827349, 164.058683, 72.799475
1.825744, 164.202932, 72.038778
1.824141, 164.347182, 71.278277
1.822541, 164.491431, 70.518435
1.820945, 164.635680, 69.759706
1.819351, 164.779930, 69.002542
1.817759, 164.924179, 68.247389
1.816171, 165.068428, 67.494688
1.814585, 165.212678, 66.744876
1.813002, 165.356927, 65.998386
1.811422, 165.501176, 65.255644
1.809844, 165.645426, 64.517074
1.808270, 165.789675, 63.783095
1.806698, 165.933924, 63.054120
1.805129, 166.078174, 62.330560
1.803562, 166.222423, 61.612819
1.801998, 166.366672, 60.901299
1.800437, 166.510922, 60.196395
1.798879, 166.655171, 59.498472
1.797323, 166.799420, 58.807604
1.795770, 166.943670, 58.123709
1.794220, 167.087919, 57.446702
1.792672, 167.232169, 56.776499
1.791127, 167.376418, 56.113019
1.789585, 167.520667, 55.456179
1.788045, 167.664917, 54.805897
1.786508, 167.809166, 54.162092
1.784974, 167.953415, 53.524685
1.783442, 168.097665, 52.893596
1.781913, 168.241914, 52.268746
1.780387, 168.386163, 51.650056
1.778863, 168.530413, 51.037450
1.777341, 168.674662, 50.430851
1.775823, 168.818911, 49.830182
1.774307, 168.963161, 49.235368
1.772793, 169.107410, 48.646333
1.771282, 169.251659, 48.063004
1.769774, 169.395909, 47.485306
1.768268, 169.540158, 46.913167
1.766765, 169.684407, 46.346514
1.765264, 169.828657, 45.785275
1.763766, 169.972906, 45.229378
1.762271, 170.117155, 44.678754
1.760778, 170.261405, 44.133332
1.759287, 170.405654, 43.593042
1.757799, 170.549903, 43.057815
1.756314, 170.694153, 42.527583
1.754831, 170.838402, 42.002278
1.753350, 170.982651, 41.481833
1.751872, 171.126901, 40.966182
1.750397, 171.271150, 40.455257
1.748924, 171.415399, 39.948994
1.747453, 171.559649, 39.447327
1.745985, 171.703898, 38.950192
1.744520, 171.848147, 38.457525
1.743056, 171.992397, 37.969263
1.741596, 172.136646, 37.485342
1.740138, 172.280895, 37.005701
1.738682, 172.425145, 36.530277
1.737228, 172.569394, 36.059009
1.735778, 172.713643, 35.591836
1.734329, 172.857893, 35.128698
1.732883, 173.002142, 34.669536
1.731439, 173.146391, 34.214290
1.729998, 173.290641, 33.762900
1.728559, 173.434890, 33.315310
1.727123, 173.579139, 32.871460
1.725689, 173.723389, 32.431294
1.724257, 173.867638, 31.994756
1.722827, 174.011887, 31.561787
1.721400, 174.156137, 31.132334
1.719976, 174.300386, 30.706340
1.718554, 174.444635, 30.283750
1.717134, 174.588885, 29.864510
1.715716, 174.733134, 29.448566
1.714301, 174.877383, 29.035864
1.712888, 175.021633, 28.626351
1.711477, 175.165882, 28.219975
1.710069, 175.310132, 27.816684
1.708663, 175.454381, 27.416425
1.707260, 175.598630, 27.019148
1.705858, 175.742880, 26.624801
1.704459, 175.887129, 26.233334
1.703063, 176.031378, 25.844697
1.701668, 176.175628, 25.458840
1.700276, 176.319877, 25.075715
1.698886, 176.464126, 24.695277
1.697499, 176.608376, 24.317521
1.696113, 176.752625, 23.942457
1.694730, 176.896874, 23.570096
1.693349, 177.041124, 23.200448
1.691971, 177.185373, 22.833523
1.690594, 177.329622, 22.469332
1.689220, 177.473872, 22.107884
1.687848, 177.618121, 21.749188
1.686479, 177.762370, 21.393255
1.685111, 177.906620, 21.040094
1.683746, 178.050869, 20.689715
1.682383, 178.195118, 20.342125
1.681022, 178.339368, 19.997335
1.679664, 178.483617, 19.655352
1.678307, 178.627866, 19.316186
1.676953, 178.772116, 18.979846
1.675601, 178.916365, 18.646339
1.674251, 179.060614, 18.315674
1.672904, 179.204864, 17.987858
1.671558, 179.349113, 17.662901
1.670215, 179.493362, 17.340809
1.668874, 179.637612, 17.021590
1.667535, 179.781861, 16.705253
1.666198, 179.926110, 16.391803
1.664863, 180.070360, 16.081250
1.663530, 180.214609, 15.773598
1.662200, 180.358858, 15.468857
1.660871, 180.503108, 15.167032
1.659545, 180.647357, 14.868130
1.658221, 180.791606, 14.572159
1.656899, 180.935856, 14.279123
1.655579, 181.080105, 13.989031
1.654261, 181.224354, 13.701887
1.652946, 181.368604, 13.417699
1.651632, 181.512853, 13.136472
1.650321, 181.657102, 12.858212
1.649011, 181.801352, 12.582924
1.647704, 181.945601, 12.310615
1.646399, 182.089850, 12.041290
1.645095, 182.234100, 11.774955
1.643794, 182.378349, 11.511614
1.642495, 182.522598, 11.251272
1.641198, 182.666848, 10.993936
1.639903, 182.811097, 10.739610
1.638610, 182.955346, 10.488298
1.637319, 183.099596, 10.240005
1.636030, 183.243845, 9.994737
1.634743, 183.388095, 9.752497
1.633458, 183.532344, 9.513290
1.632176, 183.676593, 9.277121
1.630895, 183.820843, 9.043993
1.629616, 183.965092, 8.813910
1.628339, 184.109341, 8.586877
1.627064, 184.253591, 8.362897
1.625792, 184.397840, 8.141975
1.624521, 184.542089, 7.924114
1.623252, 184.686339, 7.709317
1.621985, 184.830588, 7.497587
1.620720, 184.974837, 7.288930
1.619457, 185.119087, 7.083347
1.618196, 185.263336, 6.880842
1.616937, 185.407585, 6.681418
1.615680, 185.551835, 6.485077
1.614425, 185.696084, 6.291824
1.613172, 185.840333, 6.101661
1.611921, 185.984583, 5.914590
1.610672, 186.128832, 5.730614
1.609424, 186.273081, 5.549736
1.608179, 186.417331, 5.371958
1.606936, 186.561580, 5.197282
1.605694, 186.705829, 5.025711
1.604455, 186.850079, 4.857248
1.603217, 186.994328, 4.691893
1.601981, 187.138577, 4.529650
1.600747, 187.282827, 4.370520
1.599515, 187.427076, 4.214505
1.598285, 187.571325, 4.061607
1.597057, 187.715575, 3.911827
1.595831, 187.859824, 3.765168
1.594606, 188.004073, 3.621630
1.593384, 188.148323, 3.481216
1.592163, 188.292572, 3.343926
1.590944, 188.436821, 3.209762
1.589727, 188.581071, 3.078725
1.588512, 188.725320, 2.950817
1.587299, 188.869569, 2.826038
1.586088, 189.013819, 2.704389
1.584878, 189.158068, 2.585872
1.583670, 189.302317, 2.470487
1.582464, 189.446567, 2.358236
1.581260, 189.590816, 2.249118
1.580058, 189.735065, 2.143134
1.578858, 189.879315, 2.040286
1.577659, 190.023564, 1.940573
1.576463, 190.167813, 1.843996
1.575268, 190.312063, 1.750556
1.574075, 190.456312, 1.660252
1.572883, 190.600561, 1.573085
1.571694, 190.744811, 1.489055
1.570506, 190.889060, 1.408163
1.569320, 191.033309, 1.330408
1.568136, 191.177559, 1.255790
1.566954, 191.321808, 1.184310
1.565773, 191.466057, 1.115966
1.564595, 191.610307, 1.050760
1.563418, 191.754556, 0.988690
1.562242, 191.898806, 0.929757
1.561069, 192.043055, 0.873959
1.559897, 192.187304, 0.821297
1.558727, 192.331554, 0.771770
1.557559, 192.475803, 0.725378
1.556393, 192.620052, 0.682119
1.555228, 192.764302, 0.641993
1.554065, 192.908551, 0.605000
1.552904, 193.052800, 0.571139
1.551744, 193.197050, 0.540408
1.550587, 193.341299, 0.512808
1.549431, 193.485548, 0.488336
1.548276, 193.629798, 0.466995
1.547124, 193.774047, 0.448789
1.545973, 193.918296, 0.433722
1.544824, 194.062546, 0.421798
1.543676, 194.206795, 0.413020
1.542531, 194.351044, 0.407394
1.541387, 194.495294, 0.404922
1.540244, 194.639543, 0.405608
1.539104, 194.783792, 0.409455
1.537965, 194.928042, 0.416468
1.536827, 195.072291, 0.426649
1.535692, 195.216540, 0.440002
1.534558, 195.360790, 0.456529
1.533426, 195.505039, 0.476235
1.532295, 195.649288, 0.499122
1.531166, 195.793538, 0.525192
1.530039, 195.937787, 0.554449
1.528913, 196.082036, 0.586896
1.527789, 196.226286, 0.622535
1.526667, 196.370535, 0.661368
1.525547, 196.514784, 0.703398
1.524428, 196.659034, 0.748628
1.523310, 196.803283, 0.797060
1.522195, 196.947532, 0.848695
1.521080, 197.091782, 0.903537
1.519968, 197.236031, 0.961587
1.518857, 197.380280, 1.022847
1.517748, 197.524530, 1.087319
1.516640, 197.668779, 1.155006
1.515534, 197.813028, 1.225907
1.514430, 197.957278, 1.300027
1.513327, 198.101527, 1.377365
1.512226, 198.245776, 1.457923
1.511127, 198.390026, 1.541704
1.510029, 198.534275, 1.628708
1.508932, 198.678524, 1.718936
1.507838, 198.822774, 1.812390
1.506744, 198.967023, 1.909071
1.505653, 199.111272, 2.008980
1.504563, 199.255522, 2.112118
1.503474, 199.399771, 2.218486
1.502388, 199.544020, 2.328085
1.501302, 199.688270, 2.440915
1.500219, 199.832519, 2.556978
1.499136, 199.976769, 2.676273
1.498056, 200.121018, 2.798802
1.496977, 200.265267, 2.924566
1.495899, 200.409517, 3.053564
1.494823, 200.553766, 3.185796
1.493749, 200.698015, 3.321265
1.492676, 200.842265, 3.459968
1.491605, 200.986514, 3.601908
1.490535, 201.130763, 3.747083
1.489467, 201.275013, 3.895494
1.488400, 201.419262, 4.047141
1.487335, 201.563511, 4.202025
1.486271, 201.707761, 4.360144
1.485209, 201.852010, 4.521498
1.484149, 201.996259, 4.686089
1.483089, 202.140509, 4.853914
1.482032, 202.284758, 5.024974
1.480976, 202.429007, 5.199269
1.479921, 202.573257, 5.376797
1.478868, 202.717506, 5.557559
1.477817, 202.861755, 5.741554
1.476766, 203.006005, 5.928781
1.475718, 203.150254, 6.119240
1.474671, 203.294503, 6.312930
1.473625, 203.438753, 6.509849
1.472581, 203.583002, 6.709998
1.471538, 203.727251, 6.913376
1.470497, 203.871501, 7.119980
1.469457, 204.015750, 7.329811
1.468419, 204.159999, 7.542868
1.467382, 204.304249, 7.759148
1.466347, 204.448498, 7.978652
1.465313, 204.592747, 8.201377
1.464281, 204.736997, 8.427324
1.463250, 204.881246, 8.656489
1.462220, 205.025495, 8.888873
1.461192, 205.169745, 9.124473
1.460166, 205.313994, 9.363288
1.459141, 205.458243, 9.605317
1.458117, 205.602493, 9.850559
1.457095, 205.746742, 10.099010
1.456074, 205.890991, 10.350671
1.455054, 206.035241, 10.605539
1.454036, 206.179490, 10.863613
1.453020, 206.323739, 11.124890
1.452005, 206.467989, 11.389369
1.450991, 206.612238, 11.657049
1.449978, 206.756487, 11.927927
1.448968, 206.900737, 12.202001
1.447958, 207.044986, 12.479269
1.446950, 207.189235, 12.759730
1.445943, 207.333485, 13.043381
1.444938, 207.477734, 13.330221
1.443934, 207.621983, 13.620246
1.442932, 207.766233, 13.913455
1.441930, 207.910482, 14.209846
1.440931, 208.054732, 14.509417
1.439932, 208.198981, 14.812164
1.438935, 208.343230, 15.118087
1.437940, 208.487480, 15.427181
1.436946, 208.631729, 15.739446
1.435953, 208.775978, 16.054878
1.434961, 208.920228, 16.373476
1.433971, 209.064477, 16.695236
1.432983, 209.208726, 17.020156
1.431995, 209.352976, 17.348234
1.431009, 209.497225, 17.679466
1.430025, 209.641474, 18.013850
1.429041, 209.785724, 18.351384
1.428059, 209.929973, 18.692065
1.427079, 210.074222, 19.035890
1.426100, 210.218472, 19.382856
1.425122, 210.362721, 19.732960
1.424145, 210.506970, 20.086200
1.423170, 210.651220, 20.442573
1.422196, 210.795469, 20.802075
1.421223, 210.939718, 21.164704
1.420252, 211.083968, 21.530457
1.419282, 211.228217, 21.899330
1.418314, 211.372466, 22.271321
1.417346, 211.516716, 22.646427
1.416380, 211.660965, 23.024645
1.415416, 211.805214, 23.405970
1.414453, 211.949464, 23.790402
1.413491, 212.093713, 24.177935
1.412530, 212.237962, 24.568567
1.411570, 212.382212, 24.962294
1.410612, 212.526461, 25.359114
1.409656, 212.670710, 25.759023
1.408700, 212.814960, 26.162017
1.407746, 212.959209, 26.568094
1.406793, 213.103458, 26.977250
1.405841, 213.247708, 27.389481
1.404891, 213.391957, 27.804785
1.403942, 213.536206, 28.223157
1.402994, 213.680456, 28.644594
1.402048, 213.824705, 29.069093
1.401103, 213.968954, 29.496649
1.400159, 214.113204, 29.927261
1.399216, 214.257453, 30.360914
1.398275, 214.401702, 30.797539
1.397334, 214.545952, 31.237040
1.396396, 214.690201, 31.679324
1.395458, 214.834450, 32.124297
1.394522, 214.978700, 32.571868
1.393587, 215.122949, 33.021943
1.392653, 215.267198, 33.474433
1.391720, 215.411448, 33.929247
1.390789, 215.555697, 34.386294
1.389859, 215.699946, 34.845487
1.388930, 215.844196, 35.306737
1.388002, 215.988445, 35.769955
1.387076, 216.132695, 36.235055
1.386151, 216.276944, 36.701951
1.385227, 216.421193, 37.170557
1.384304, 216.565443, 37.640788
1.383383, 216.709692, 38.112559
1.382463, 216.853941, 38.585786
1.381544, 216.998191, 39.060387
1.380626, 217.142440, 39.536280
1.379709, 217.286689, 40.013381
1.378794, 217.430939, 40.491611
1.377880, 217.575188, 40.970888
1.376967, 217.719437, 41.451132
1.376055, 217.863687, 41.932264
1.375145, 218.007936, 42.414205
1.374235, 218.152185, 42.896878
1.373327, 218.296435, 43.380204
1.372420, 218.440684, 43.864107
1.371515, 218.584933, 44.348510
1.370610, 218.729183, 44.833337
1.369707, 218.873432, 45.318514
1.368805, 219.017681, 45.803965
1.367904, 219.161931, 46.289618
1.367004, 219.306180, 46.775397
1.366106, 219.450429, 47.261231
1.365208, 219.594679, 47.747047
1.364312, 219.738928, 48.232774
1.363417, 219.883177, 48.718339
1.362523, 220.027427, 49.203674
1.361630, 220.171676, 49.688706
1.360739, 220.315925, 50.173368
1.359849, 220.460175, 50.657590
1.358959, 220.604424, 51.141304
1.358071, 220.748673, 51.624442
1.357185, 220.892923, 52.106936
1.356299, 221.037172, 52.588719
1.355414, 221.181421, 53.069727
1.354531, 221.325671, 53.549891
1.353649, 221.469920, 54.029148
1.352768, 221.614169, 54.507433
1.351888, 221.758419, 54.984682
1.351009, 221.902668, 55.460830
1.350131, 222.046917, 55.935815
1.349255, 222.191167, 56.409574
1.348379, 222.335416, 56.882044
1.347505, 222.479665, 57.353165
1.346632, 222.623915, 57.822875
1.345760, 222.768164, 58.291113
1.344889, 222.912413, 58.757819
1.344019, 223.056663, 59.222933
1.343151, 223.200912, 59.686397
1.342283, 223.345161, 60.148151
1.341417, 223.489411, 60.608138
1.340552, 223.633660, 61.066298
1.339687, 223.777909, 61.522576
1.338824, 223.922159, 61.976914
1.337963, 224.066408, 62.429256
1.337102, 224.210658, 62.879545
1.336242, 224.354907, 63.327727
1.335383, 224.499156, 63.773747
1.334526, 224.643406, 64.217549
1.333670, 224.787655, 64.659080
1.332814, 224.931904, 65.098286
1.331960, 225.076154, 65.535113
1.331107, 225.220403, 65.969510
1.330255, 225.364652, 66.401424
1.329404, 225.508902, 66.830802
1.328554, 225.653151, 67.257593
1.327706, 225.797400, 67.681747
1.326858, 225.941650, 68.103212
1.326011, 226.085899, 68.521939
1.325166, 226.230148, 68.937877
1.324321, 226.374398, 69.350977
1.323478, 226.518647, 69.761191
1.322636, 226.662896, 70.168470
1.321795, 226.807146, 70.572765
1.320954, 226.951395, 70.974029
1.320115, 227.095644, 71.372214
1.319277, 227.239894, 71.767274
1.318440, 227.384143, 72.159163
1.317605, 227.528392, 72.547833
1.316770, 227.672642, 72.933240
1.315936, 227.816891, 73.315338
1.315103, 227.961140, 73.694081
1.314272, 228.105390, 74.069426
1.313441, 228.249639, 74.441329
1.312612, 228.393888, 74.809745
1.311783, 228.538138, 75.174630
1.310956, 228.682387, 75.535943
1.310129, 228.826636, 75.893641
1.309304, 228.970886, 76.247680
1.308480, 229.115135, 76.598020
1.307656, 229.259384, 76.944618
1.306834, 229.403634, 77.287433
1.306013, 229.547883, 77.626425
1.305193, 229.692132, 77.961553
1.304373, 229.836382, 78.292777
1.303555, 229.980631, 78.620057
1.302738, 230.124880, 78.943354
1.301922, 230.269130, 79.262629
1.301107, 230.413379, 79.577842
1.300293, 230.557628, 79.888957
1.299480, 230.701878, 80.195936
1.298668, 230.846127, 80.498780
1.297857, 230.990376, 80.797511
1.297047, 231.134626, 81.092152
1.296238, 231.278875, 81.382727
1.295430, 231.423124, 81.669258
1.294623, 231.567374, 81.951769
1.293817, 231.711623, 82.230281
1.293012, 231.855872, 82.504818
1.292208, 232.000122, 82.775402
1.291405, 232.144371, 83.042055
1.290603, 232.288620, 83.304799
1.289802, 232.432870, 83.563658
1.289002, 232.577119, 83.818652
1.288203, 232.721369, 84.069804
1.287405, 232.865618, 84.317135
1.286608, 233.009867, 84.560667
1.285812, 233.154117, 84.800422
1.285017, 233.298366, 85.036422
1.284223, 233.442615, 85.268687
1.283430, 233.586865, 85.497238
1.282638, 233.731114, 85.722098
1.281847, 233.875363, 85.943287
1.281057, 234.019613, 86.160826
1.280268, 234.163862, 86.374735
1.279480, 234.308111, 86.585036
1.278692, 234.452361, 86.791750
1.277906, 234.596610, 86.994895
1.277121, 234.740859, 87.194494
1.276337, 234.885109, 87.390567
1.275553, 235.029358, 87.583133
1.274771, 235.173607, 87.772213
1.273989, 235.317857, 87.957826
1.273209, 235.462106, 88.139993
1.272429, 235.606355, 88.318733
1.271651, 235.750605, 88.494067
1.270873, 235.894854, 88.666013
1.270097, 236.039103, 88.834592
1.269321, 236.183353, 88.999822
1.268546, 236.327602, 89.161723
1.267772, 236.471851, 89.320314
1.266999, 236.616101, 89.475615
1.266227, 236.760350, 89.627643
1.265456, 236.904599, 89.776419
1.264686, 237.048849, 89.921961
1.263917, 237.193098, 90.064288
1.263149, 237.337347, 90.203418
1.262382, 237.481597, 90.339371
1.261616, 237.625846, 90.472163
1.260850, 237.770095, 90.601814
1.260086, 237.914345, 90.728343
1.259322, 238.058594, 90.851766
1.258560, 238.202843, 90.972103
1.257798, 238.347093, 91.089371
1.257037, 238.491342, 91.203588
1.256277, 238.635591, 91.314772
1.255518, 238.779841, 91.422941
1.254760, 238.924090, 91.528112
1.254003, 239.068339, 91.630304
1.253247, 239.212589, 91.729532
1.252492, 239.356838, 91.825815
1.251737, 239.501087, 91.919170
1.250984, 239.645337, 92.009614
1.250231, 239.789586, 92.097165
1.249480, 239.933835, 92.181839
1.248729, 240.078085, 92.263653
1.247979, 240.222334, 92.342624
1.247230, 240.366583, 92.418769
1.246482, 240.510833, 92.492104
1.245735, 240.655082, 92.562647
1.244989, 240.799332, 92.630413
1.244243, 240.943581, 92.695419
1.243499, 241.087830, 92.757682
1.242755, 241.232080, 92.817217
1.242013, 241.376329, 92.874041
1.241271, 241.520578, 92.928171
1.240530, 241.664828, 92.979621
1.239790, 241.809077, 93.028408
1.239051, 241.953326, 93.074548
1.238313, 242.097576, 93.118056
1.237575, 242.241825, 93.158949
1.236839, 242.386074, 93.197241
1.236103, 242.530324, 93.232950
1.235368, 242.674573, 93.266088
1.234634, 242.818822, 93.296674
1.233901, 242.963072, 93.324721
1.233169, 243.107321, 93.350244
1.232438, 243.251570, 93.373260
1.231708, 243.395820, 93.393783
1.230978, 243.540069, 93.411828
1.230249, 243.684318, 93.427410
1.229521, 243.828568, 93.440544
1.228795, 243.972817, 93.451244
1.228068, 244.117066, 93.459526
1.227343, 244.261316, 93.465404
1.226619, 244.405565, 93.468893
1.225895, 244.549814, 93.470006
1.225173, 244.694064, 93.468759
1.224451, 244.838313, 93.465166
1.223730, 244.982562, 93.459242
1.223010, 245.126812, 93.450999
1.222290, 245.271061, 93.440453
1.221572, 245.415310, 93.427617
1.220854, 245.559560, 93.412506
1.220138, 245.703809, 93.395133
1.219422, 245.848058, 93.375513
1.218707, 245.992308, 93.353658
1.217992, 246.136557, 93.329584
1.217279, 246.280806, 93.303338
1.216566, 246.425056, 93.274828
1.215855, 246.569305, 93.244174
1.215144, 246.713554, 93.211354
1.214434, 246.857804, 93.176381
1.213725, 247.002053, 93.139269
1.213016, 247.146302, 93.100031
1.212309, 247.290552, 93.058680
1.211602, 247.434801, 93.015229
1.210896, 247.579050, 92.969691
1.210191, 247.723300, 92.922079
1.209487, 247.867549, 92.872406
1.208783, 248.011798, 92.820685
1.208080, 248.156048, 92.766929
1.207379, 248.300297, 92.711150
1.206678, 248.444546, 92.653361
1.205977, 248.588796, 92.593575
1.205278, 248.733045, 92.531804
1.204579, 248.877295, 92.468060
1.203882, 249.021544, 92.402356
1.203185, 249.165793, 92.334705
1.202488, 249.310043, 92.265119
1.201793, 249.454292, 92.193609
1.201099, 249.598541, 92.120188
1.200405, 249.742791, 92.044869
1.199712, 249.887040, 91.967662
1.199020, 250.031289, 91.888564
1.198328, 250.175539, 91.807553
1.197638, 250.319788, 91.724604
1.196948, 250.464037, 91.639695
1.196259, 250.608287, 91.552803
1.195571, 250.752536, 91.463904
1.194884, 250.896785, 91.372975
1.194197, 251.041035, 91.279994
1.193511, 251.185284, 91.184939
1.192826, 251.329533, 91.087787
1.192142, 251.473783, 90.988515
1.191459, 251.618032, 90.887103
1.190776, 251.762281, 90.783529
1.190094, 251.906531, 90.677770
1.189413, 252.050780, 90.569805
1.188733, 252.195029, 90.459614
1.188053, 252.339279, 90.347175
1.187374, 252.483528, 90.232467
1.186696, 252.627777, 90.115469
1.186019, 252.772027, 89.996162
1.185343, 252.916276, 89.874524
1.184667, 253.060525, 89.750536
1.183992, 253.204775, 89.624176
1.183318, 253.349024, 89.495427
1.182645, 253.493273, 89.364267
1.181972, 253.637523, 89.230677
1.181300, 253.781772, 89.094638
1.180629, 253.926021, 88.956131
1.179959, 254.070271, 88.815136
1.179289, 254.214520, 88.671634
1.178620, 254.358769, 88.525608
1.177952, 254.503019, 88.377037
1.177285, 254.647268, 88.225905
1.176618, 254.791517, 88.072192
1.175952, 254.935767, 87.915880
1.175287, 255.080016, 87.756952
1.174623, 255.224265, 87.595389
1.173959, 255.368515, 87.431174
1.173296, 255.512764, 87.264290
1.172634, 255.657013, 87.094719
1.171972, 255.801263, 86.922443
1.171311, 255.945512, 86.747446
1.170651, 256.089761, 86.569711
1.169992, 256.234011, 86.389221
1.169334, 256.378260, 86.205959
1.168677, 256.522509, 86.019909
1.168021, 256.666759, 85.831054
1.167365, 256.811008, 85.639379
1.166711, 256.955258, 85.444867
1.166057, 257.099507, 85.247502
1.165405, 257.243756, 85.047269
1.164753, 257.388006, 84.844152
1.164102, 257.532255, 84.638135
1.163453, 257.676504, 84.429202
1.162804, 257.820754, 84.217340
1.162156, 257.965003, 84.002533
1.161510, 258.109252, 83.784765
1.160864, 258.253502, 83.564021
1.160220, 258.397751, 83.340288
1.159576, 258.542000, 83.113551
1.158934, 258.686250, 82.883795
1.158292, 258.830499, 82.651006
1.157652, 258.974748, 82.415169
1.157013, 259.118998, 82.176271
1.156375, 259.263247, 81.934298
1.155737, 259.407496, 81.689235
1.155101, 259.551746, 81.441070
1.154466, 259.695995, 81.189789
1.153832, 259.840244, 80.935378
1.153199, 259.984494, 80.677824
1.152567, 260.128743, 80.417115
1.151936, 260.272992, 80.153236
1.151306, 260.417242, 79.886175
1.150678, 260.561491, 79.615919
1.150050, 260.705740, 79.342455
1.149424, 260.849990, 79.065772
1.148799, 260.994239, 78.785856
1.148175, 261.138488, 78.502696
1.147552, 261.282738, 78.216278
1.146931, 261.426987, 77.926591
1.146311, 261.571236, 77.633624
1.145692, 261.715486, 77.337363
1.145074, 261.859735, 77.037798
1.144457, 262.003984, 76.734917
1.143841, 262.148234, 76.428708
1.143226, 262.292483, 76.119160
1.142613, 262.436732, 75.806262
1.142000, 262.580982, 75.490002
1.141389, 262.725231, 75.170370
1.140779, 262.869480, 74.847354
1.140170, 263.013730, 74.520944
1.139562, 263.157979, 74.191129
1.138956, 263.302228, 73.857898
1.138350, 263.446478, 73.521241
1.137746, 263.590727, 73.181148
1.137143, 263.734976, 72.837607
1.136542, 263.879226, 72.490610
1.135941, 264.023475, 72.140145
1.135342, 264.167724, 71.786204
1.134744, 264.311974, 71.428775
1.134148, 264.456223, 71.067850
1.133552, 264.600472, 70.703418
1.132958, 264.744722, 70.335470
1.132365, 264.888971, 69.963997
1.131773, 265.033220, 69.588989
1.131182, 265.177470, 69.210437
1.130593, 265.321719, 68.828332
1.130005, 265.465969, 68.442665
1.129417, 265.610218, 68.053426
1.128831, 265.754467, 67.660607
1.128247, 265.898717, 67.264200
1.127663, 266.042966, 66.864194
1.127081, 266.187215, 66.460583
1.126500, 266.331465, 66.053357
1.125920, 266.475714, 65.642508
1.125341, 266.619963, 65.228028
1.124764, 266.764213, 64.809908
1.124188, 266.908462, 64.388140
1.123613, 267.052711, 63.962717
1.123039, 267.196961, 63.533631
1.122467, 267.341210, 63.100872
1.121896, 267.485459, 62.664435
1.121326, 267.629709, 62.224311
1.120757, 267.773958, 61.780493
1.120190, 267.918207, 61.332972
1.119623, 268.062457, 60.881743
1.119058, 268.206706, 60.426797
1.118495, 268.350955, 59.968127
1.117932, 268.495205, 59.505727
1.117371, 268.639454, 59.039588
1.116811, 268.783703, 58.569705
1.116252, 268.927953, 58.096070
1.115694, 269.072202, 57.618676
1.115138, 269.216451, 57.137518
1.114582, 269.360701, 56.652587
1.114028, 269.504950, 56.163878
1.113475, 269.649199, 55.671384
1.112923, 269.793449, 55.175099
1.112373, 269.937698, 54.675017
1.111823, 270.081947, 54.171131
1.111275, 270.226197, 53.663435
1.110728, 270.370446, 53.151924
1.110182, 270.514695, 52.636590
1.109638, 270.658945, 52.117429
1.109094, 270.803194, 51.594435
1.108552, 270.947443, 51.067601
1.108011, 271.091693, 50.536922
1.107471, 271.235942, 50.002393
1.106932, 271.380191, 49.464008
1.106395, 271.524441, 48.921761
1.105858, 271.668690, 48.375648
1.105323, 271.812939, 47.825663
1.104790, 271.957189, 47.271801
1.104257, 272.101438, 46.714056
1.103726, 272.245687, 46.152424
1.103195, 272.389937, 45.586900
1.102667, 272.534186, 45.017478
1.102139, 272.678435, 44.444093
1.101612, 272.822685, 43.866414
1.101087, 272.966934, 43.284038
1.100562, 273.111183, 42.696565
1.100039, 273.255433, 42.103597
1.099517, 273.399682, 41.504741
1.098996, 273.543932, 40.899603
1.098476, 273.688181, 40.287795
1.097957, 273.832430, 39.668929
1.097439, 273.976680, 39.042621
1.096923, 274.120929, 38.408488
1.096407, 274.265178, 37.766152
1.095893, 274.409428, 37.115234
1.095380, 274.553677, 36.455360
1.094868, 274.697926, 35.786159
1.094358, 274.842176, 35.107260
1.093849, 274.986425, 34.418295
1.093341, 275.130674, 33.718901
1.092834, 275.274924, 33.008714
1.092329, 275.419173, 32.287374
1.091825, 275.563422, 31.554523
1.091321, 275.707672, 30.809807
1.090820, 275.851921, 30.052871
1.090319, 275.996170, 29.283366
1.089820, 276.140420, 28.500942
1.089322, 276.284669, 27.705254
1.088825, 276.428918, 26.895958
1.088329, 276.573168, 26.072712
1.087834, 276.717417, 25.235176
1.087341, 276.861666, 24.383015
1.086848, 277.005916, 23.515893
1.086357, 277.150165, 22.633477
1.085867, 277.294414, 21.735437
1.085378, 277.438664, 20.821446
1.084891, 277.582913, 19.891177
1.084405, 277.727162, 18.944306
1.083920, 277.871412, 17.980512
1.083436, 278.015661, 16.999476
1.082954, 278.159910, 16.000881
1.082472, 278.304160, 14.984410
1.081993, 278.448409, 13.949752
1.081514, 278.592658, 12.896595
1.081036, 278.736908, 11.824632
1.080560, 278.881157, 10.733554
1.080085, 279.025406, 9.623058
1.079611, 279.169656, 8.492841
1.079138, 279.313905, 7.342603
1.078667, 279.458154, 6.172045
1.078197, 279.602404, 4.980871
1.077729, 279.746653, 3.768787
1.077261, 279.890902, 2.535500
1.076796, 280.035152, 1.280721
1.076331, 280.179401, 0.004161
1.075868, 280.323650, -1.294466
1.075406, 280.467900, -2.615445
1.074945, 280.612149, -3.959058
1.074485, 280.756398, -5.325585
1.074026, 280.900648, -6.715304
1.073569, 281.044897, -8.128493
1.073113, 281.189146, -9.565425
1.072659, 281.333396, -11.026374
1.072205, 281.477645, -12.511609
1.071753, 281.621895, -14.021401
1.071302, 281.766144, -15.556015
1.070852, 281.910393, -17.115718
1.070404, 282.054643, -18.700771
1.069956, 282.198892, -20.311437
1.069510, 282.343141, -21.947975
1.069065, 282.487391, -23.610642
1.068621, 282.631640, -25.299696
1.068179, 282.775889, -27.015388
1.067738, 282.920139, -28.757973
1.067298, 283.064388, -30.527700
1.066860, 283.208637, -32.324818
1.066422, 283.352887, -34.149574
1.065986, 283.497136, -36.002213
1.065552, 283.641385, -37.882978
1.065118, 283.785635, -39.792110
1.064686, 283.929884, -41.729851
1.064256, 284.074133, -43.696438
1.063826, 284.218383, -45.692107
1.063398, 284.362632, -47.717093
1.062972, 284.506881, -49.771628
1.062546, 284.651131, -51.855946
1.062123, 284.795380, -53.970274
1.061700, 284.939629, -56.114841
1.061279, 285.083879, -58.289874
1.060860, 285.228128, -60.495596
1.060442, 285.372377, -62.732231
1.060025, 285.516627, -65.000000'''

EMBEDDED_DINT_COEN2012_CSV = r'''Wavelength [um], Frequency [THz], Dint/2pi [GHz]
2.120000, 141.411537, -70.000000
2.117840, 141.555786, -64.348888
2.115684, 141.700035, -58.970115
2.113532, 141.844285, -53.853410
2.111385, 141.988534, -48.988614
2.109242, 142.132783, -44.365683
2.107104, 142.277033, -39.974686
2.104970, 142.421282, -35.805802
2.102840, 142.565532, -31.849324
2.100714, 142.709781, -28.095651
2.098593, 142.854030, -24.535295
2.096476, 142.998280, -21.158870
2.094363, 143.142529, -17.957101
2.092255, 143.286778, -14.920815
2.090151, 143.431028, -12.040946
2.088051, 143.575277, -9.308530
2.085955, 143.719526, -6.714705
2.083863, 143.863776, -4.250710
2.081776, 144.008025, -1.907886
2.079693, 144.152274, 0.322328
2.077614, 144.296524, 2.448396
2.075539, 144.440773, 4.478682
2.073468, 144.585022, 6.421455
2.071402, 144.729272, 8.284889
2.069339, 144.873521, 10.077062
2.067281, 145.017770, 11.805960
2.065227, 145.162020, 13.479476
2.063176, 145.306269, 15.105410
2.061130, 145.450518, 16.691473
2.059088, 145.594768, 18.245286
2.057050, 145.739017, 19.774379
2.055016, 145.883266, 21.286195
2.052986, 146.027516, 22.788090
2.050960, 146.171765, 24.287334
2.048938, 146.316014, 25.790913
2.046920, 146.460264, 27.301704
2.044906, 146.604513, 28.818795
2.042896, 146.748762, 30.341139
2.040890, 146.893012, 31.867703
2.038888, 147.037261, 33.397463
2.036889, 147.181510, 34.929408
2.034895, 147.325760, 36.462541
2.032905, 147.470009, 37.995874
2.030918, 147.614258, 39.528430
2.028935, 147.758508, 41.059245
2.026957, 147.902757, 42.587367
2.024982, 148.047006, 44.111854
2.023010, 148.191256, 45.631773
2.021043, 148.335505, 47.146206
2.019080, 148.479754, 48.654243
2.017120, 148.624004, 50.154985
2.015164, 148.768253, 51.647546
2.013212, 148.912502, 53.131047
2.011264, 149.056752, 54.604622
2.009319, 149.201001, 56.067414
2.007379, 149.345250, 57.518577
2.005442, 149.489500, 58.957274
2.003508, 149.633749, 60.382680
2.001579, 149.777998, 61.793978
1.999653, 149.922248, 63.190361
1.997731, 150.066497, 64.571034
1.995812, 150.210746, 65.935209
1.993898, 150.354996, 67.282108
1.991986, 150.499245, 68.610963
1.990079, 150.643495, 69.921016
1.988175, 150.787744, 71.211518
1.986275, 150.931993, 72.481727
1.984379, 151.076243, 73.730912
1.982486, 151.220492, 74.958353
1.980596, 151.364741, 76.163334
1.978711, 151.508991, 77.345153
1.976829, 151.653240, 78.503112
1.974950, 151.797489, 79.636525
1.973075, 151.941739, 80.744714
1.971204, 152.085988, 81.827009
1.969336, 152.230237, 82.882747
1.967471, 152.374487, 83.911276
1.965611, 152.518736, 84.911951
1.963753, 152.662985, 85.884134
1.961900, 152.807235, 86.827197
1.960049, 152.951484, 87.740520
1.958203, 153.095733, 88.623488
1.956359, 153.239983, 89.475498
1.954519, 153.384232, 90.295952
1.952683, 153.528481, 91.084260
1.950850, 153.672731, 91.839840
1.949021, 153.816980, 92.562118
1.947194, 153.961229, 93.250528
1.945372, 154.105479, 93.904508
1.943553, 154.249728, 94.523509
1.941737, 154.393977, 95.106983
1.939924, 154.538227, 95.654394
1.938115, 154.682476, 96.165211
1.936309, 154.826725, 96.638910
1.934507, 154.970975, 97.074974
1.932708, 155.115224, 97.472895
1.930913, 155.259473, 97.832168
1.929120, 155.403723, 98.152325
1.927331, 155.547972, 98.433538
1.925546, 155.692221, 98.676631
1.923763, 155.836471, 98.882447
1.921984, 155.980720, 99.051822
1.920208, 156.124969, 99.185583
1.918436, 156.269219, 99.284551
1.916667, 156.413468, 99.349536
1.914901, 156.557717, 99.381342
1.913138, 156.701967, 99.380766
1.911378, 156.846216, 99.348594
1.909622, 156.990465, 99.285608
1.907869, 157.134715, 99.192581
1.906119, 157.278964, 99.070277
1.904373, 157.423213, 98.919454
1.902629, 157.567463, 98.740863
1.900889, 157.711712, 98.535245
1.899152, 157.855961, 98.303338
1.897418, 158.000211, 98.045869
1.895687, 158.144460, 97.763559
1.893960, 158.288709, 97.457122
1.892235, 158.432959, 97.127266
1.890514, 158.577208, 96.774690
1.888796, 158.721457, 96.400088
1.887081, 158.865707, 96.004145
1.885369, 159.009956, 95.587541
1.883660, 159.154206, 95.150950
1.881955, 159.298455, 94.695036
1.880252, 159.442704, 94.220459
1.878552, 159.586954, 93.727872
1.876856, 159.731203, 93.217921
1.875163, 159.875452, 92.691246
1.873472, 160.019702, 92.148481
1.871785, 160.163951, 91.590252
1.870101, 160.308200, 91.017181
1.868419, 160.452450, 90.429881
1.866741, 160.596699, 89.828962
1.865066, 160.740948, 89.215025
1.863394, 160.885198, 88.588666
1.861724, 161.029447, 87.950477
1.860058, 161.173696, 87.301041
1.858395, 161.317946, 86.640936
1.856735, 161.462195, 85.970734
1.855077, 161.606444, 85.291003
1.853423, 161.750694, 84.602304
1.851772, 161.894943, 83.905191
1.850123, 162.039192, 83.200214
1.848478, 162.183442, 82.487916
1.846835, 162.327691, 81.768837
1.845195, 162.471940, 81.043509
1.843558, 162.616190, 80.312459
1.841925, 162.760439, 79.576209
1.840294, 162.904688, 78.835276
1.838666, 163.048938, 78.090172
1.837040, 163.193187, 77.341401
1.835418, 163.337436, 76.589466
1.833798, 163.481686, 75.834860
1.832182, 163.625935, 75.078076
1.830568, 163.770184, 74.319597
1.828957, 163.914434, 73.559906
1.827349, 164.058683, 72.799475
1.825744, 164.202932, 72.038778
1.824141, 164.347182, 71.278277
1.822541, 164.491431, 70.518435
1.820945, 164.635680, 69.759706
1.819351, 164.779930, 69.002542
1.817759, 164.924179, 68.247389
1.816171, 165.068428, 67.494688
1.814585, 165.212678, 66.744876
1.813002, 165.356927, 65.998386
1.811422, 165.501176, 65.255644
1.809844, 165.645426, 64.517074
1.808270, 165.789675, 63.783095
1.806698, 165.933924, 63.054120
1.805129, 166.078174, 62.330560
1.803562, 166.222423, 61.612819
1.801998, 166.366672, 60.901299
1.800437, 166.510922, 60.196395
1.798879, 166.655171, 59.498472
1.797323, 166.799420, 58.807604
1.795770, 166.943670, 58.123709
1.794220, 167.087919, 57.446702
1.792672, 167.232169, 56.776499
1.791127, 167.376418, 56.113019
1.789585, 167.520667, 55.456179
1.788045, 167.664917, 54.805897
1.786508, 167.809166, 54.162092
1.784974, 167.953415, 53.524685
1.783442, 168.097665, 52.893596
1.781913, 168.241914, 52.268746
1.780387, 168.386163, 51.650056
1.778863, 168.530413, 51.037450
1.777341, 168.674662, 50.430851
1.775823, 168.818911, 49.830182
1.774307, 168.963161, 49.235368
1.772793, 169.107410, 48.646333
1.771282, 169.251659, 48.063004
1.769774, 169.395909, 47.485306
1.768268, 169.540158, 46.913167
1.766765, 169.684407, 46.346514
1.765264, 169.828657, 45.785275
1.763766, 169.972906, 45.229378
1.762271, 170.117155, 44.678754
1.760778, 170.261405, 44.133332
1.759287, 170.405654, 43.593042
1.757799, 170.549903, 43.057815
1.756314, 170.694153, 42.527583
1.754831, 170.838402, 42.002278
1.753350, 170.982651, 41.481833
1.751872, 171.126901, 40.966182
1.750397, 171.271150, 40.455257
1.748924, 171.415399, 39.948994
1.747453, 171.559649, 39.447327
1.745985, 171.703898, 38.950192
1.744520, 171.848147, 38.457525
1.743056, 171.992397, 37.969263
1.741596, 172.136646, 37.485342
1.740138, 172.280895, 37.005701
1.738682, 172.425145, 36.530277
1.737228, 172.569394, 36.059009
1.735778, 172.713643, 35.591836
1.734329, 172.857893, 35.128698
1.732883, 173.002142, 34.669536
1.731439, 173.146391, 34.214290
1.729998, 173.290641, 33.762900
1.728559, 173.434890, 33.315310
1.727123, 173.579139, 32.871460
1.725689, 173.723389, 32.431294
1.724257, 173.867638, 31.994756
1.722827, 174.011887, 31.561787
1.721400, 174.156137, 31.132334
1.719976, 174.300386, 30.706340
1.718554, 174.444635, 30.283750
1.717134, 174.588885, 29.864510
1.715716, 174.733134, 29.448566
1.714301, 174.877383, 29.035864
1.712888, 175.021633, 28.626351
1.711477, 175.165882, 28.219975
1.710069, 175.310132, 27.816684
1.708663, 175.454381, 27.416425
1.707260, 175.598630, 27.019148
1.705858, 175.742880, 26.624801
1.704459, 175.887129, 26.233334
1.703063, 176.031378, 25.844697
1.701668, 176.175628, 25.458840
1.700276, 176.319877, 25.075715
1.698886, 176.464126, 24.695277
1.697499, 176.608376, 24.317521
1.696113, 176.752625, 23.942457
1.694730, 176.896874, 23.570096
1.693349, 177.041124, 23.200448
1.691971, 177.185373, 22.833523
1.690594, 177.329622, 22.469332
1.689220, 177.473872, 22.107884
1.687848, 177.618121, 21.749188
1.686479, 177.762370, 21.393255
1.685111, 177.906620, 21.040094
1.683746, 178.050869, 20.689715
1.682383, 178.195118, 20.342125
1.681022, 178.339368, 19.997335
1.679664, 178.483617, 19.655352
1.678307, 178.627866, 19.316186
1.676953, 178.772116, 18.979846
1.675601, 178.916365, 18.646339
1.674251, 179.060614, 18.315674
1.672904, 179.204864, 17.987858
1.671558, 179.349113, 17.662901
1.670215, 179.493362, 17.340809
1.668874, 179.637612, 17.021590
1.667535, 179.781861, 16.705253
1.666198, 179.926110, 16.391803
1.664863, 180.070360, 16.081250
1.663530, 180.214609, 15.773598
1.662200, 180.358858, 15.468857
1.660871, 180.503108, 15.167032
1.659545, 180.647357, 14.868130
1.658221, 180.791606, 14.572159
1.656899, 180.935856, 14.279123
1.655579, 181.080105, 13.989031
1.654261, 181.224354, 13.701887
1.652946, 181.368604, 13.417699
1.651632, 181.512853, 13.136472
1.650321, 181.657102, 12.858212
1.649011, 181.801352, 12.582924
1.647704, 181.945601, 12.310615
1.646399, 182.089850, 12.041290
1.645095, 182.234100, 11.774955
1.643794, 182.378349, 11.511614
1.642495, 182.522598, 11.251272
1.641198, 182.666848, 10.993936
1.639903, 182.811097, 10.739610
1.638610, 182.955346, 10.488298
1.637319, 183.099596, 10.240005
1.636030, 183.243845, 9.994737
1.634743, 183.388095, 9.752497
1.633458, 183.532344, 9.513290
1.632176, 183.676593, 9.277121
1.630895, 183.820843, 9.043993
1.629616, 183.965092, 8.813910
1.628339, 184.109341, 8.586877
1.627064, 184.253591, 8.362897
1.625792, 184.397840, 8.141975
1.624521, 184.542089, 7.924114
1.623252, 184.686339, 7.709317
1.621985, 184.830588, 7.497587
1.620720, 184.974837, 7.288930
1.619457, 185.119087, 7.083347
1.618196, 185.263336, 6.880842
1.616937, 185.407585, 6.681418
1.615680, 185.551835, 6.485077
1.614425, 185.696084, 6.291824
1.613172, 185.840333, 6.101661
1.611921, 185.984583, 5.914590
1.610672, 186.128832, 5.730614
1.609424, 186.273081, 5.549736
1.608179, 186.417331, 5.371958
1.606936, 186.561580, 5.197282
1.605694, 186.705829, 5.025711
1.604455, 186.850079, 4.857248
1.603217, 186.994328, 4.691893
1.601981, 187.138577, 4.529650
1.600747, 187.282827, 4.370520
1.599515, 187.427076, 4.214505
1.598285, 187.571325, 4.061607
1.597057, 187.715575, 3.911827
1.595831, 187.859824, 3.765168
1.594606, 188.004073, 3.621630
1.593384, 188.148323, 3.481216
1.592163, 188.292572, 3.343926
1.590944, 188.436821, 3.209762
1.589727, 188.581071, 3.078725
1.588512, 188.725320, 2.950817
1.587299, 188.869569, 2.826038
1.586088, 189.013819, 2.704389
1.584878, 189.158068, 2.585872
1.583670, 189.302317, 2.470487
1.582464, 189.446567, 2.358236
1.581260, 189.590816, 2.249118
1.580058, 189.735065, 2.143134
1.578858, 189.879315, 2.040286
1.577659, 190.023564, 1.940573
1.576463, 190.167813, 1.843996
1.575268, 190.312063, 1.750556
1.574075, 190.456312, 1.660252
1.572883, 190.600561, 1.573085
1.571694, 190.744811, 1.489055
1.570506, 190.889060, 1.408163
1.569320, 191.033309, 1.330408
1.568136, 191.177559, 1.255790
1.566954, 191.321808, 1.184310
1.565773, 191.466057, 1.115966
1.564595, 191.610307, 1.050760
1.563418, 191.754556, 0.988690
1.562242, 191.898806, 0.929757
1.561069, 192.043055, 0.873959
1.559897, 192.187304, 0.821297
1.558727, 192.331554, 0.771770
1.557559, 192.475803, 0.725378
1.556393, 192.620052, 0.682119
1.555228, 192.764302, 0.641993
1.554065, 192.908551, 0.605000
1.552904, 193.052800, 0.571139
1.551744, 193.197050, 0.540408
1.550587, 193.341299, 0.512808
1.549431, 193.485548, 0.488336
1.548276, 193.629798, 0.466995
1.547124, 193.774047, 0.448789
1.545973, 193.918296, 0.433722
1.544824, 194.062546, 0.421798
1.543676, 194.206795, 0.413020
1.542531, 194.351044, 0.407394
1.541387, 194.495294, 0.404922
1.540244, 194.639543, 0.405608
1.539104, 194.783792, 0.409455
1.537965, 194.928042, 0.416468
1.536827, 195.072291, 0.426649
1.535692, 195.216540, 0.440002
1.534558, 195.360790, 0.456529
1.533426, 195.505039, 0.476235
1.532295, 195.649288, 0.499122
1.531166, 195.793538, 0.525192
1.530039, 195.937787, 0.554449
1.528913, 196.082036, 0.586896
1.527789, 196.226286, 0.622535
1.526667, 196.370535, 0.661368
1.525547, 196.514784, 0.703398
1.524428, 196.659034, 0.748628
1.523310, 196.803283, 0.797060
1.522195, 196.947532, 0.848695
1.521080, 197.091782, 0.903537
1.519968, 197.236031, 0.961587
1.518857, 197.380280, 1.022847
1.517748, 197.524530, 1.087319
1.516640, 197.668779, 1.155006
1.515534, 197.813028, 1.225907
1.514430, 197.957278, 1.300027
1.513327, 198.101527, 1.377365
1.512226, 198.245776, 1.457923
1.511127, 198.390026, 1.541704
1.510029, 198.534275, 1.628708
1.508932, 198.678524, 1.718936
1.507838, 198.822774, 1.812390
1.506744, 198.967023, 1.909071
1.505653, 199.111272, 2.008980
1.504563, 199.255522, 2.112118
1.503474, 199.399771, 2.218486
1.502388, 199.544020, 2.328085
1.501302, 199.688270, 2.440915
1.500219, 199.832519, 2.556978
1.499136, 199.976769, 2.676273
1.498056, 200.121018, 2.798802
1.496977, 200.265267, 2.924566
1.495899, 200.409517, 3.053564
1.494823, 200.553766, 3.185796
1.493749, 200.698015, 3.321265
1.492676, 200.842265, 3.459968
1.491605, 200.986514, 3.601908
1.490535, 201.130763, 3.747083
1.489467, 201.275013, 3.895494
1.488400, 201.419262, 4.047141
1.487335, 201.563511, 4.202025
1.486271, 201.707761, 4.360144
1.485209, 201.852010, 4.521498
1.484149, 201.996259, 4.686089
1.483089, 202.140509, 4.853914
1.482032, 202.284758, 5.024974
1.480976, 202.429007, 5.199269
1.479921, 202.573257, 5.376797
1.478868, 202.717506, 5.557559
1.477817, 202.861755, 5.741554
1.476766, 203.006005, 5.928781
1.475718, 203.150254, 6.119240
1.474671, 203.294503, 6.312930
1.473625, 203.438753, 6.509849
1.472581, 203.583002, 6.709998
1.471538, 203.727251, 6.913376
1.470497, 203.871501, 7.119980
1.469457, 204.015750, 7.329811
1.468419, 204.159999, 7.542868
1.467382, 204.304249, 7.759148
1.466347, 204.448498, 7.978652
1.465313, 204.592747, 8.201377
1.464281, 204.736997, 8.427324
1.463250, 204.881246, 8.656489
1.462220, 205.025495, 8.888873
1.461192, 205.169745, 9.124473
1.460166, 205.313994, 9.363288
1.459141, 205.458243, 9.605317
1.458117, 205.602493, 9.850559
1.457095, 205.746742, 10.099010
1.456074, 205.890991, 10.350671
1.455054, 206.035241, 10.605539
1.454036, 206.179490, 10.863613
1.453020, 206.323739, 11.124890
1.452005, 206.467989, 11.389369
1.450991, 206.612238, 11.657049
1.449978, 206.756487, 11.927927
1.448968, 206.900737, 12.202001
1.447958, 207.044986, 12.479269
1.446950, 207.189235, 12.759730
1.445943, 207.333485, 13.043381
1.444938, 207.477734, 13.330221
1.443934, 207.621983, 13.620246
1.442932, 207.766233, 13.913455
1.441930, 207.910482, 14.209846
1.440931, 208.054732, 14.509417
1.439932, 208.198981, 14.812164
1.438935, 208.343230, 15.118087
1.437940, 208.487480, 15.427181
1.436946, 208.631729, 15.739446
1.435953, 208.775978, 16.054878
1.434961, 208.920228, 16.373476
1.433971, 209.064477, 16.695236
1.432983, 209.208726, 17.020156
1.431995, 209.352976, 17.348234
1.431009, 209.497225, 17.679466
1.430025, 209.641474, 18.013850
1.429041, 209.785724, 18.351384
1.428059, 209.929973, 18.692065
1.427079, 210.074222, 19.035890
1.426100, 210.218472, 19.382856
1.425122, 210.362721, 19.732960
1.424145, 210.506970, 20.086200
1.423170, 210.651220, 20.442573
1.422196, 210.795469, 20.802075
1.421223, 210.939718, 21.164704
1.420252, 211.083968, 21.530457
1.419282, 211.228217, 21.899330
1.418314, 211.372466, 22.271321
1.417346, 211.516716, 22.646427
1.416380, 211.660965, 23.024645
1.415416, 211.805214, 23.405970
1.414453, 211.949464, 23.790402
1.413491, 212.093713, 24.177935
1.412530, 212.237962, 24.568567
1.411570, 212.382212, 24.962294
1.410612, 212.526461, 25.359114
1.409656, 212.670710, 25.759023
1.408700, 212.814960, 26.162017
1.407746, 212.959209, 26.568094
1.406793, 213.103458, 26.977250
1.405841, 213.247708, 27.389481
1.404891, 213.391957, 27.804785
1.403942, 213.536206, 28.223157
1.402994, 213.680456, 28.644594
1.402048, 213.824705, 29.069093
1.401103, 213.968954, 29.496649
1.400159, 214.113204, 29.927261
1.399216, 214.257453, 30.360914
1.398275, 214.401702, 30.797539
1.397334, 214.545952, 31.237040
1.396396, 214.690201, 31.679324
1.395458, 214.834450, 32.124297
1.394522, 214.978700, 32.571868
1.393587, 215.122949, 33.021943
1.392653, 215.267198, 33.474433
1.391720, 215.411448, 33.929247
1.390789, 215.555697, 34.386294
1.389859, 215.699946, 34.845487
1.388930, 215.844196, 35.306737
1.388002, 215.988445, 35.769955
1.387076, 216.132695, 36.235055
1.386151, 216.276944, 36.701951
1.385227, 216.421193, 37.170557
1.384304, 216.565443, 37.640788
1.383383, 216.709692, 38.112559
1.382463, 216.853941, 38.585786
1.381544, 216.998191, 39.060387
1.380626, 217.142440, 39.536280
1.379709, 217.286689, 40.013381
1.378794, 217.430939, 40.491611
1.377880, 217.575188, 40.970888
1.376967, 217.719437, 41.451132
1.376055, 217.863687, 41.932264
1.375145, 218.007936, 42.414205
1.374235, 218.152185, 42.896878
1.373327, 218.296435, 43.380204
1.372420, 218.440684, 43.864107
1.371515, 218.584933, 44.348510
1.370610, 218.729183, 44.833337
1.369707, 218.873432, 45.318514
1.368805, 219.017681, 45.803965
1.367904, 219.161931, 46.289618
1.367004, 219.306180, 46.775397
1.366106, 219.450429, 47.261231
1.365208, 219.594679, 47.747047
1.364312, 219.738928, 48.232774
1.363417, 219.883177, 48.718339
1.362523, 220.027427, 49.203674
1.361630, 220.171676, 49.688706
1.360739, 220.315925, 50.173368
1.359849, 220.460175, 50.657590
1.358959, 220.604424, 51.141304
1.358071, 220.748673, 51.624442
1.357185, 220.892923, 52.106936
1.356299, 221.037172, 52.588719
1.355414, 221.181421, 53.069727
1.354531, 221.325671, 53.549891
1.353649, 221.469920, 54.029148
1.352768, 221.614169, 54.507433
1.351888, 221.758419, 54.984682
1.351009, 221.902668, 55.460830
1.350131, 222.046917, 55.935815
1.349255, 222.191167, 56.409574
1.348379, 222.335416, 56.882044
1.347505, 222.479665, 57.353165
1.346632, 222.623915, 57.822875
1.345760, 222.768164, 58.291113
1.344889, 222.912413, 58.757819
1.344019, 223.056663, 59.222933
1.343151, 223.200912, 59.686397
1.342283, 223.345161, 60.148151
1.341417, 223.489411, 60.608138
1.340552, 223.633660, 61.066298
1.339687, 223.777909, 61.522576
1.338824, 223.922159, 61.976914
1.337963, 224.066408, 62.429256
1.337102, 224.210658, 62.879545
1.336242, 224.354907, 63.327727
1.335383, 224.499156, 63.773747
1.334526, 224.643406, 64.217549
1.333670, 224.787655, 64.659080
1.332814, 224.931904, 65.098286
1.331960, 225.076154, 65.535113
1.331107, 225.220403, 65.969510
1.330255, 225.364652, 66.401424
1.329404, 225.508902, 66.830802
1.328554, 225.653151, 67.257593
1.327706, 225.797400, 67.681747
1.326858, 225.941650, 68.103212
1.326011, 226.085899, 68.521939
1.325166, 226.230148, 68.937877
1.324321, 226.374398, 69.350977
1.323478, 226.518647, 69.761191
1.322636, 226.662896, 70.168470
1.321795, 226.807146, 70.572765
1.320954, 226.951395, 70.974029
1.320115, 227.095644, 71.372214
1.319277, 227.239894, 71.767274
1.318440, 227.384143, 72.159163
1.317605, 227.528392, 72.547833
1.316770, 227.672642, 72.933240
1.315936, 227.816891, 73.315338
1.315103, 227.961140, 73.694081
1.314272, 228.105390, 74.069426
1.313441, 228.249639, 74.441329
1.312612, 228.393888, 74.809745
1.311783, 228.538138, 75.174630
1.310956, 228.682387, 75.535943
1.310129, 228.826636, 75.893641
1.309304, 228.970886, 76.247680
1.308480, 229.115135, 76.598020
1.307656, 229.259384, 76.944618
1.306834, 229.403634, 77.287433
1.306013, 229.547883, 77.626425
1.305193, 229.692132, 77.961553
1.304373, 229.836382, 78.292777
1.303555, 229.980631, 78.620057
1.302738, 230.124880, 78.943354
1.301922, 230.269130, 79.262629
1.301107, 230.413379, 79.577842
1.300293, 230.557628, 79.888957
1.299480, 230.701878, 80.195936
1.298668, 230.846127, 80.498780
1.297857, 230.990376, 80.797511
1.297047, 231.134626, 81.092152
1.296238, 231.278875, 81.382727
1.295430, 231.423124, 81.669258
1.294623, 231.567374, 81.951769
1.293817, 231.711623, 82.230281
1.293012, 231.855872, 82.504818
1.292208, 232.000122, 82.775402
1.291405, 232.144371, 83.042055
1.290603, 232.288620, 83.304799
1.289802, 232.432870, 83.563658
1.289002, 232.577119, 83.818652
1.288203, 232.721369, 84.069804
1.287405, 232.865618, 84.317135
1.286608, 233.009867, 84.560667
1.285812, 233.154117, 84.800422
1.285017, 233.298366, 85.036422
1.284223, 233.442615, 85.268687
1.283430, 233.586865, 85.497238
1.282638, 233.731114, 85.722098
1.281847, 233.875363, 85.943287
1.281057, 234.019613, 86.160826
1.280268, 234.163862, 86.374735
1.279480, 234.308111, 86.585036
1.278692, 234.452361, 86.791750
1.277906, 234.596610, 86.994895
1.277121, 234.740859, 87.194494
1.276337, 234.885109, 87.390567
1.275553, 235.029358, 87.583133
1.274771, 235.173607, 87.772213
1.273989, 235.317857, 87.957826
1.273209, 235.462106, 88.139993
1.272429, 235.606355, 88.318733
1.271651, 235.750605, 88.494067
1.270873, 235.894854, 88.666013
1.270097, 236.039103, 88.834592
1.269321, 236.183353, 88.999822
1.268546, 236.327602, 89.161723
1.267772, 236.471851, 89.320314
1.266999, 236.616101, 89.475615
1.266227, 236.760350, 89.627643
1.265456, 236.904599, 89.776419
1.264686, 237.048849, 89.921961
1.263917, 237.193098, 90.064288
1.263149, 237.337347, 90.203418
1.262382, 237.481597, 90.339371
1.261616, 237.625846, 90.472163
1.260850, 237.770095, 90.601814
1.260086, 237.914345, 90.728343
1.259322, 238.058594, 90.851766
1.258560, 238.202843, 90.972103
1.257798, 238.347093, 91.089371
1.257037, 238.491342, 91.203588
1.256277, 238.635591, 91.314772
1.255518, 238.779841, 91.422941
1.254760, 238.924090, 91.528112
1.254003, 239.068339, 91.630304
1.253247, 239.212589, 91.729532
1.252492, 239.356838, 91.825815
1.251737, 239.501087, 91.919170
1.250984, 239.645337, 92.009614
1.250231, 239.789586, 92.097165
1.249480, 239.933835, 92.181839
1.248729, 240.078085, 92.263653
1.247979, 240.222334, 92.342624
1.247230, 240.366583, 92.418769
1.246482, 240.510833, 92.492104
1.245735, 240.655082, 92.562647
1.244989, 240.799332, 92.630413
1.244243, 240.943581, 92.695419
1.243499, 241.087830, 92.757682
1.242755, 241.232080, 92.817217
1.242013, 241.376329, 92.874041
1.241271, 241.520578, 92.928171
1.240530, 241.664828, 92.979621
1.239790, 241.809077, 93.028408
1.239051, 241.953326, 93.074548
1.238313, 242.097576, 93.118056
1.237575, 242.241825, 93.158949
1.236839, 242.386074, 93.197241
1.236103, 242.530324, 93.232950
1.235368, 242.674573, 93.266088
1.234634, 242.818822, 93.296674
1.233901, 242.963072, 93.324721
1.233169, 243.107321, 93.350244
1.232438, 243.251570, 93.373260
1.231708, 243.395820, 93.393783
1.230978, 243.540069, 93.411828
1.230249, 243.684318, 93.427410
1.229521, 243.828568, 93.440544
1.228795, 243.972817, 93.451244
1.228068, 244.117066, 93.459526
1.227343, 244.261316, 93.465404
1.226619, 244.405565, 93.468893
1.225895, 244.549814, 93.470006
1.225173, 244.694064, 93.468759
1.224451, 244.838313, 93.465166
1.223730, 244.982562, 93.459242
1.223010, 245.126812, 93.450999
1.222290, 245.271061, 93.440453
1.221572, 245.415310, 93.427617
1.220854, 245.559560, 93.412506
1.220138, 245.703809, 93.395133
1.219422, 245.848058, 93.375513
1.218707, 245.992308, 93.353658
1.217992, 246.136557, 93.329584
1.217279, 246.280806, 93.303338
1.216566, 246.425056, 93.274828
1.215855, 246.569305, 93.244174
1.215144, 246.713554, 93.211354
1.214434, 246.857804, 93.176381
1.213725, 247.002053, 93.139269
1.213016, 247.146302, 93.100031
1.212309, 247.290552, 93.058680
1.211602, 247.434801, 93.015229
1.210896, 247.579050, 92.969691
1.210191, 247.723300, 92.922079
1.209487, 247.867549, 92.872406
1.208783, 248.011798, 92.820685
1.208080, 248.156048, 92.766929
1.207379, 248.300297, 92.711150
1.206678, 248.444546, 92.653361
1.205977, 248.588796, 92.593575
1.205278, 248.733045, 92.531804
1.204579, 248.877295, 92.468060
1.203882, 249.021544, 92.402356
1.203185, 249.165793, 92.334705
1.202488, 249.310043, 92.265119
1.201793, 249.454292, 92.193609
1.201099, 249.598541, 92.120188
1.200405, 249.742791, 92.044869
1.199712, 249.887040, 91.967662
1.199020, 250.031289, 91.888564
1.198328, 250.175539, 91.807553
1.197638, 250.319788, 91.724604
1.196948, 250.464037, 91.639695
1.196259, 250.608287, 91.552803
1.195571, 250.752536, 91.463904
1.194884, 250.896785, 91.372975
1.194197, 251.041035, 91.279994
1.193511, 251.185284, 91.184939
1.192826, 251.329533, 91.087787
1.192142, 251.473783, 90.988515
1.191459, 251.618032, 90.887103
1.190776, 251.762281, 90.783529
1.190094, 251.906531, 90.677770
1.189413, 252.050780, 90.569805
1.188733, 252.195029, 90.459614
1.188053, 252.339279, 90.347175
1.187374, 252.483528, 90.232467
1.186696, 252.627777, 90.115469
1.186019, 252.772027, 89.996162
1.185343, 252.916276, 89.874524
1.184667, 253.060525, 89.750536
1.183992, 253.204775, 89.624176
1.183318, 253.349024, 89.495427
1.182645, 253.493273, 89.364267
1.181972, 253.637523, 89.230677
1.181300, 253.781772, 89.094638
1.180629, 253.926021, 88.956131
1.179959, 254.070271, 88.815136
1.179289, 254.214520, 88.671634
1.178620, 254.358769, 88.525608
1.177952, 254.503019, 88.377037
1.177285, 254.647268, 88.225905
1.176618, 254.791517, 88.072192
1.175952, 254.935767, 87.915880
1.175287, 255.080016, 87.756952
1.174623, 255.224265, 87.595389
1.173959, 255.368515, 87.431174
1.173296, 255.512764, 87.264290
1.172634, 255.657013, 87.094719
1.171972, 255.801263, 86.922443
1.171311, 255.945512, 86.747446
1.170651, 256.089761, 86.569711
1.169992, 256.234011, 86.389221
1.169334, 256.378260, 86.205959
1.168677, 256.522509, 86.019909
1.168021, 256.666759, 85.831054
1.167365, 256.811008, 85.639379
1.166711, 256.955258, 85.444867
1.166057, 257.099507, 85.247502
1.165405, 257.243756, 85.047269
1.164753, 257.388006, 84.844152
1.164102, 257.532255, 84.638135
1.163453, 257.676504, 84.429202
1.162804, 257.820754, 84.217340
1.162156, 257.965003, 84.002533
1.161510, 258.109252, 83.784765
1.160864, 258.253502, 83.564021
1.160220, 258.397751, 83.340288
1.159576, 258.542000, 83.113551
1.158934, 258.686250, 82.883795
1.158292, 258.830499, 82.651006
1.157652, 258.974748, 82.415169
1.157013, 259.118998, 82.176271
1.156375, 259.263247, 81.934298
1.155737, 259.407496, 81.689235
1.155101, 259.551746, 81.441070
1.154466, 259.695995, 81.189789
1.153832, 259.840244, 80.935378
1.153199, 259.984494, 80.677824
1.152567, 260.128743, 80.417115
1.151936, 260.272992, 80.153236
1.151306, 260.417242, 79.886175
1.150678, 260.561491, 79.615919
1.150050, 260.705740, 79.342455
1.149424, 260.849990, 79.065772
1.148799, 260.994239, 78.785856
1.148175, 261.138488, 78.502696
1.147552, 261.282738, 78.216278
1.146931, 261.426987, 77.926591
1.146311, 261.571236, 77.633624
1.145692, 261.715486, 77.337363
1.145074, 261.859735, 77.037798
1.144457, 262.003984, 76.734917
1.143841, 262.148234, 76.428708
1.143226, 262.292483, 76.119160
1.142613, 262.436732, 75.806262
1.142000, 262.580982, 75.490002
1.141389, 262.725231, 75.170370
1.140779, 262.869480, 74.847354
1.140170, 263.013730, 74.520944
1.139562, 263.157979, 74.191129
1.138956, 263.302228, 73.857898
1.138350, 263.446478, 73.521241
1.137746, 263.590727, 73.181148
1.137143, 263.734976, 72.837607
1.136542, 263.879226, 72.490610
1.135941, 264.023475, 72.140145
1.135342, 264.167724, 71.786204
1.134744, 264.311974, 71.428775
1.134148, 264.456223, 71.067850
1.133552, 264.600472, 70.703418
1.132958, 264.744722, 70.335470
1.132365, 264.888971, 69.963997
1.131773, 265.033220, 69.588989
1.131182, 265.177470, 69.210437
1.130593, 265.321719, 68.828332
1.130005, 265.465969, 68.442665
1.129417, 265.610218, 68.053426
1.128831, 265.754467, 67.660607
1.128247, 265.898717, 67.264200
1.127663, 266.042966, 66.864194
1.127081, 266.187215, 66.460583
1.126500, 266.331465, 66.053357
1.125920, 266.475714, 65.642508
1.125341, 266.619963, 65.228028
1.124764, 266.764213, 64.809908
1.124188, 266.908462, 64.388140
1.123613, 267.052711, 63.962717
1.123039, 267.196961, 63.533631
1.122467, 267.341210, 63.100872
1.121896, 267.485459, 62.664435
1.121326, 267.629709, 62.224311
1.120757, 267.773958, 61.780493
1.120190, 267.918207, 61.332972
1.119623, 268.062457, 60.881743
1.119058, 268.206706, 60.426797
1.118495, 268.350955, 59.968127
1.117932, 268.495205, 59.505727
1.117371, 268.639454, 59.039588
1.116811, 268.783703, 58.569705
1.116252, 268.927953, 58.096070
1.115694, 269.072202, 57.618676
1.115138, 269.216451, 57.137518
1.114582, 269.360701, 56.652587
1.114028, 269.504950, 56.163878
1.113475, 269.649199, 55.671384
1.112923, 269.793449, 55.175099
1.112373, 269.937698, 54.675017
1.111823, 270.081947, 54.171131
1.111275, 270.226197, 53.663435
1.110728, 270.370446, 53.151924
1.110182, 270.514695, 52.636590
1.109638, 270.658945, 52.117429
1.109094, 270.803194, 51.594435
1.108552, 270.947443, 51.067601
1.108011, 271.091693, 50.536922
1.107471, 271.235942, 50.002393
1.106932, 271.380191, 49.464008
1.106395, 271.524441, 48.921761
1.105858, 271.668690, 48.375648
1.105323, 271.812939, 47.825663
1.104790, 271.957189, 47.271801
1.104257, 272.101438, 46.714056
1.103726, 272.245687, 46.152424
1.103195, 272.389937, 45.586900
1.102667, 272.534186, 45.017478
1.102139, 272.678435, 44.444093
1.101612, 272.822685, 43.866414
1.101087, 272.966934, 43.284038
1.100562, 273.111183, 42.696565
1.100039, 273.255433, 42.103597
1.099517, 273.399682, 41.504741
1.098996, 273.543932, 40.899603
1.098476, 273.688181, 40.287795
1.097957, 273.832430, 39.668929
1.097439, 273.976680, 39.042621
1.096923, 274.120929, 38.408488
1.096407, 274.265178, 37.766152
1.095893, 274.409428, 37.115234
1.095380, 274.553677, 36.455360
1.094868, 274.697926, 35.786159
1.094358, 274.842176, 35.107260
1.093849, 274.986425, 34.418295
1.093341, 275.130674, 33.718901
1.092834, 275.274924, 33.008714
1.092329, 275.419173, 32.287374
1.091825, 275.563422, 31.554523
1.091321, 275.707672, 30.809807
1.090820, 275.851921, 30.052871
1.090319, 275.996170, 29.283366
1.089820, 276.140420, 28.500942
1.089322, 276.284669, 27.705254
1.088825, 276.428918, 26.895958
1.088329, 276.573168, 26.072712
1.087834, 276.717417, 25.235176
1.087341, 276.861666, 24.383015
1.086848, 277.005916, 23.515893
1.086357, 277.150165, 22.633477
1.085867, 277.294414, 21.735437
1.085378, 277.438664, 20.821446
1.084891, 277.582913, 19.891177
1.084405, 277.727162, 18.944306
1.083920, 277.871412, 17.980512
1.083436, 278.015661, 16.999476
1.082954, 278.159910, 16.000881
1.082472, 278.304160, 14.984410
1.081993, 278.448409, 13.949752
1.081514, 278.592658, 12.896595
1.081036, 278.736908, 11.824632
1.080560, 278.881157, 10.733554
1.080085, 279.025406, 9.623058
1.079611, 279.169656, 8.492841
1.079138, 279.313905, 7.342603
1.078667, 279.458154, 6.172045
1.078197, 279.602404, 4.980871
1.077729, 279.746653, 3.768787
1.077261, 279.890902, 2.535500
1.076796, 280.035152, 1.280721
1.076331, 280.179401, 0.004161
1.075868, 280.323650, -1.294466
1.075406, 280.467900, -2.615445
1.074945, 280.612149, -3.959058
1.074485, 280.756398, -5.325585
1.074026, 280.900648, -6.715304
1.073569, 281.044897, -8.128493
1.073113, 281.189146, -9.565425
1.072659, 281.333396, -11.026374
1.072205, 281.477645, -12.511609
1.071753, 281.621895, -14.021401
1.071302, 281.766144, -15.556015
1.070852, 281.910393, -17.115718
1.070404, 282.054643, -18.700771
1.069956, 282.198892, -20.311437
1.069510, 282.343141, -21.947975
1.069065, 282.487391, -23.610642
1.068621, 282.631640, -25.299696
1.068179, 282.775889, -27.015388
1.067738, 282.920139, -28.757973
1.067298, 283.064388, -30.527700
1.066860, 283.208637, -32.324818
1.066422, 283.352887, -34.149574
1.065986, 283.497136, -36.002213
1.065552, 283.641385, -37.882978
1.065118, 283.785635, -39.792110
1.064686, 283.929884, -41.729851
1.064256, 284.074133, -43.696438
1.063826, 284.218383, -45.692107
1.063398, 284.362632, -47.717093
1.062972, 284.506881, -49.771628
1.062546, 284.651131, -51.855946
1.062123, 284.795380, -53.970274
1.061700, 284.939629, -56.114841
1.061279, 285.083879, -58.289874
1.060860, 285.228128, -60.495596
1.060442, 285.372377, -62.732231
1.060025, 285.516627, -65.000000'''

EMBEDDED_PUMP_PRESET_NAMES = [
    # Single-line reference pump
    "default_cw_1550",

    # Multi-CW / frequency-domain style pumps
    "default_dual_cw_mu_plus_10",
    "default_dual_cw_mu_plus_20",
    "default_comb_9_lines",
    "default_multitone_random",

    # Time-domain pulses
    "default_pulse_gaussian",
    "default_pulse_gaussian_wide",
    "default_pulse_gaussian_narrow",
    "default_pulse_flattop",
    "default_pulse_chirped_gaussian",
    "default_pulse_sech_chirped",
    "default_pulse_double",

    # Modulated/noisy pump tests
    "default_phase_modulated_cw",
    "default_cw_noise_seeded",
]


def _phase_ramp_mode(mode_mu, N):
    """Fast-time phasor that places a tone at an integer mode offset."""
    n = np.arange(N, dtype=float)
    return np.exp(1j * 2.0 * np.pi * float(mode_mu) * n / float(N))


def _gaussian_window(x, sigma):
    return np.exp(-0.5 * (x / float(sigma)) ** 2)


def _super_gaussian_window(x, width, order=8):
    return np.exp(-0.5 * np.abs(x / float(width)) ** int(order))


def build_embedded_pump_profile(preset_name, number_modes):
    """
    Return a complex fast-time pump profile for a built-in preset.

    All profiles are normalized later by the LLEState helpers to
    mean(|profile_t|^2) = 1. The GUI pump-power slider then sets the
    total normalized pump strength.
    """
    N = int(number_modes)
    n = np.arange(N, dtype=float)
    x = n - N / 2.0

    if preset_name == "default_cw_1550":
        return np.ones(N, dtype=complex)

    if preset_name == "default_dual_cw_mu_plus_10":
        line_ratio = 0.55
        phase = 0.0
        return (1.0 + line_ratio * np.exp(1j * phase) * _phase_ramp_mode(+10, N)).astype(complex)

    if preset_name == "default_dual_cw_mu_plus_20":
        line_ratio = 0.45
        phase = np.pi / 2.0
        return (1.0 + line_ratio * np.exp(1j * phase) * _phase_ramp_mode(+20, N)).astype(complex)

    if preset_name == "default_comb_9_lines":
        spacing_mu = 3
        sigma_lines = 3.0
        profile = np.zeros(N, dtype=complex)
        for k in range(-4, 5):
            amp = np.exp(-0.5 * (k / sigma_lines) ** 2)
            phase = 0.15 * k ** 2
            profile += amp * np.exp(1j * phase) * _phase_ramp_mode(k * spacing_mu, N)
        return profile.astype(complex)

    if preset_name == "default_multitone_random":
        rng = np.random.default_rng(12345)
        mode_offsets = np.array([-31, -23, -17, -9, 0, 6, 14, 22, 29])
        amps = np.array([0.25, 0.35, 0.28, 0.50, 1.00, 0.42, 0.31, 0.27, 0.20])
        phases = rng.uniform(0.0, 2.0 * np.pi, size=len(mode_offsets))
        phases[mode_offsets == 0] = 0.0
        profile = np.zeros(N, dtype=complex)
        for mu, amp, phase in zip(mode_offsets, amps, phases):
            profile += amp * np.exp(1j * phase) * _phase_ramp_mode(mu, N)
        return profile.astype(complex)

    if preset_name == "default_pulse_gaussian":
        sigma = 7.0
        return _gaussian_window(x, sigma).astype(complex)

    if preset_name == "default_pulse_gaussian_wide":
        sigma = 10.0
        return _gaussian_window(x, sigma).astype(complex)

    if preset_name == "default_pulse_gaussian_narrow":
        sigma = 4.0
        return _gaussian_window(x, sigma).astype(complex)

    if preset_name == "default_pulse_flattop":
        width = 14.0
        return _super_gaussian_window(x, width, order=10).astype(complex)

    if preset_name == "default_pulse_chirped_gaussian":
        sigma = 8.0
        chirp = 0.035
        return (_gaussian_window(x, sigma) * np.exp(1j * chirp * x ** 2)).astype(complex)

    if preset_name == "default_pulse_sech_chirped":
        T0 = 7.0
        phase_chirp = 0.03 * x ** 2
        return (1.0 / np.cosh(x / T0) * np.exp(1j * phase_chirp)).astype(complex)

    if preset_name == "default_pulse_double":
        sep = 24.0
        sigma = 5.0
        return (
            _gaussian_window(x - sep / 2.0, sigma)
            + 0.8 * _gaussian_window(x + sep / 2.0, sigma) * np.exp(1j * 0.6)
        ).astype(complex)

    if preset_name == "default_phase_modulated_cw":
        beta_pm = 1.4
        modulation_mu = 4
        return np.exp(1j * beta_pm * np.sin(2.0 * np.pi * modulation_mu * n / float(N))).astype(complex)

    if preset_name == "default_cw_noise_seeded":
        rng = np.random.default_rng(54321)
        amp_noise = 0.06 * rng.standard_normal(N)
        phase_noise = 0.12 * rng.standard_normal(N)
        return ((1.0 + amp_noise) * np.exp(1j * phase_noise)).astype(complex)

    raise ValueError(
        f"Unknown embedded pump preset: {preset_name}. "
        f"Available presets: {', '.join(EMBEDDED_PUMP_PRESET_NAMES)}"
    )
EMBEDDED_DINT_INGAP_SIO2_THI500_WID600_MODE1_CSV = r'''Wavelength [um],Frequency [THz],Dint/2pi [GHz]
2.9,103.376709655,4773.07789859
2.89300373425,103.626709655,4826.7067352
2.88604114431,103.876709655,4878.06436081
2.87911198762,104.126709655,4927.1395314
2.87221602396,104.376709655,4973.92000195
2.86535301538,104.626709655,5018.3994972
2.85852272622,104.876709655,5060.57406754
2.85172492303,105.126709655,5100.44332186
2.84495937462,105.376709655,5138.00979887
2.83822585195,105.626709655,5173.27947357
2.83152412817,105.876709655,5206.26164965
2.82485397855,106.126709655,5236.96945816
2.81821518048,106.376709655,5265.42740596
2.81160751344,106.626709655,5291.72023078
2.80503075897,106.876709655,5315.78966326
2.79848470064,107.126709655,5337.66251196
2.79196912406,107.376709655,5357.37098372
2.7854838168,107.626709655,5374.94467636
2.77902856843,107.876709655,5390.4119315
2.77260317045,108.126709655,5403.81464315
2.76620741628,108.376709655,5415.18839412
2.75984110125,108.626709655,5424.57386816
2.75350402257,108.876709655,5432.0138139
2.74719597931,109.126709655,5437.55202982
2.74091677237,109.376709655,5441.23483231
2.73466620446,109.626709655,5443.10492579
2.7284440801,109.876709655,5443.21519779
2.72225020559,110.126709655,5441.62035701
2.71608438897,110.376709655,5438.41199918
2.70994644001,110.626709655,5433.58634986
2.70383617021,110.876709655,5427.19898744
2.69775339277,111.126709655,5419.2970796
2.69169792256,111.376709655,5409.9318456
2.68566957609,111.626709655,5399.15489657
2.67966817154,111.876709655,5387.01679904
2.6736935287,112.126709655,5373.56790425
2.66774546897,112.376709655,5358.85859551
2.66182381531,112.626709655,5342.9381656
2.65592839228,112.876709655,5325.85360183
2.65005902597,113.126709655,5307.65872257
2.64421554402,113.376709655,5288.38759453
2.63839777557,113.626709655,5268.0982685
2.63260555128,113.876709655,5246.83536039
2.62683870328,114.126709655,5224.66038036
2.62109706516,114.376709655,5201.61758958
2.61538047198,114.626709655,5177.72307268
2.60968876024,114.876709655,5153.02490364
2.60402176782,115.126709655,5127.56075786
2.59837933406,115.376709655,5101.36977938
2.59276129965,115.626709655,5074.48957324
2.58716750667,115.876709655,5046.95714443
2.58159779856,116.126709655,5018.81059187
2.5760520201,116.376709655,4990.07684012
2.57053001741,116.626709655,4960.79801053
2.56503163791,116.876709655,4931.00154525
2.55955673034,117.126709655,4900.72058971
2.55410514472,117.376709655,4869.98360036
2.54867673234,117.626709655,4838.81980267
2.54327134577,117.876709655,4807.25717407
2.53788883882,118.126709655,4775.34275919
2.53252906651,118.376709655,4743.09215839
2.52719188513,118.626709655,4710.52258097
2.52187715213,118.876709655,4677.65493406
2.51658472619,119.126709655,4644.51190924
2.51131446717,119.376709655,4611.11886503
2.50606623608,119.626709655,4577.49556159
2.50083989511,119.876709655,4543.66277121
2.49563530759,120.126709655,4509.6400327
2.49045233799,120.376709655,4475.44664914
2.48529085189,120.626709655,4441.09980832
2.48015071601,120.876709655,4406.61801863
2.47503179813,121.126709655,4372.01681915
2.46993396716,121.376709655,4337.31323008
2.46485709307,121.626709655,4302.52174779
2.45980104688,121.876709655,4267.65871919
2.45476570069,122.126709655,4232.73568812
2.44975092765,122.376709655,4197.78351278
2.44475660191,122.626709655,4162.80234828
2.43978259868,122.876709655,4127.80314656
2.43482879417,123.126709655,4092.78924922
2.42989506559,123.376709655,4057.78034278
2.42498129115,123.626709655,4022.78431482
2.42008735003,123.876709655,3987.81143892
2.4152131224,124.126709655,3952.87229822
2.41035848939,124.376709655,3917.97387312
2.40552333308,124.626709655,3883.13014606
2.40070753648,124.876709655,3848.34121952
2.39591098356,125.126709655,3813.6228362
2.39113355921,125.376709655,3778.98055969
2.38637514922,125.626709655,3744.41902826
2.38163564031,125.876709655,3709.94858098
2.37691492008,126.126709655,3675.57473693
2.37221287702,126.376709655,3641.30390016
2.36752940052,126.626709655,3607.1530452
2.36286438082,126.876709655,3573.12076112
2.35821770903,127.126709655,3539.20814992
2.35358927713,127.376709655,3505.42050075
2.34897897791,127.626709655,3471.7630051
2.34438670504,127.876709655,3438.2418273
2.339812353,128.126709655,3404.85838727
2.33525581708,128.376709655,3371.61701766
2.33071699341,128.626709655,3338.52578168
2.32619577891,128.876709655,3305.58611504
2.3216920713,129.126709655,3272.79835887
2.3172057691,129.376709655,3240.1711231
2.31273677159,129.626709655,3207.70599645
2.30828497885,129.876709655,3175.40330106
2.30385029172,130.126709655,3143.26297181
2.2994326118,130.376709655,3111.30172961
2.29503184143,130.626709655,3079.50847072
2.29064788372,130.876709655,3047.88746656
2.28628064251,131.126709655,3016.45304895
2.28193002235,131.376709655,2985.20094546
2.27759592856,131.626709655,2954.12952354
2.27327826713,131.876709655,2923.24091335
2.2689769448,132.126709655,2892.53821512
2.26469186899,132.376709655,2862.01933003
2.26042294783,132.626709655,2831.68523
2.25617009014,132.876709655,2801.54207801
2.25193320541,133.126709655,2771.58964657
2.24771220384,133.376709655,2741.83321209
2.24350699627,133.626709655,2712.25683139
2.23931749422,133.876709655,2682.87924025
2.23514360988,134.126709655,2653.70225946
2.23098525607,134.376709655,2624.70623194
2.22684234628,134.626709655,2595.91226935
2.22271479462,134.876709655,2567.32169306
2.21860251585,135.126709655,2538.91310285
2.21450542537,135.376709655,2510.71621306
2.21042343918,135.626709655,2482.71200725
2.2063564739,135.876709655,2454.90174196
2.20230444679,136.126709655,2427.3087066
2.19826727568,136.376709655,2399.8916779
2.19424487903,136.626709655,2372.68463157
2.19023717589,136.876709655,2345.67503438
2.18624408588,137.126709655,2318.8719318
2.18226552923,137.376709655,2292.25405273
2.17830142674,137.626709655,2265.83950664
2.17435169979,137.876709655,2239.61905393
2.17041627031,138.126709655,2213.6006686
2.16649506082,138.376709655,2187.78031234
2.16258799438,138.626709655,2162.15785019
2.15869499461,138.876709655,2136.73255896
2.15481598568,139.126709655,2111.50455578
2.15095089231,139.376709655,2086.47508087
2.14709963975,139.626709655,2061.63983528
2.14326215379,139.876709655,2037.00127406
2.13943836074,140.126709655,2012.55837095
2.13562818744,140.376709655,1988.32508636
2.13183156127,140.626709655,1964.26603636
2.12804841009,140.876709655,1940.41336092
2.12427866229,141.126709655,1916.75599082
2.12052224678,141.376709655,1893.28950701
2.11677909294,141.626709655,1870.01616395
2.11304913068,141.876709655,1846.9340972
2.10933229037,142.126709655,1824.04954712
2.10562850291,142.376709655,1801.35138397
2.10193769964,142.626709655,1778.8457069
2.09825981242,142.876709655,1756.52422024
2.09459477356,143.126709655,1734.38819414
2.09094251585,143.376709655,1712.45381904
2.08730297254,143.626709655,1690.68596276
2.08367607738,143.876709655,1669.11678584
2.08006176452,144.126709655,1647.73708209
2.07645996862,144.376709655,1626.53530902
2.07287062476,144.626709655,1605.52141288
2.06929366848,144.876709655,1584.69582431
2.06572903577,145.126709655,1564.04345956
2.06217666304,145.376709655,1543.58020827
2.05863648715,145.626709655,1523.30026576
2.0551084454,145.876709655,1503.20053101
2.05159247551,146.126709655,1483.28170231
2.04808851563,146.376709655,1463.5385623
2.04459650431,146.626709655,1443.97469034
2.04111638056,146.876709655,1424.5994446
2.03764808377,147.126709655,1405.38864711
2.03419155375,147.376709655,1386.34303788
2.03074673073,147.626709655,1367.48167573
2.02731355532,147.876709655,1348.79397807
2.02389196856,148.126709655,1330.27869548
2.02048191186,148.376709655,1311.93520643
2.01708332705,148.626709655,1293.77665616
2.01369615633,148.876709655,1275.77679728
2.0103203423,149.126709655,1257.93757654
2.00695582793,149.376709655,1240.2720553
2.00360255659,149.626709655,1222.76162577
2.00026047202,149.876709655,1205.44459596
1.99692951833,150.126709655,1188.27825525
1.99360964,150.376709655,1171.27693479
1.99030078189,150.626709655,1154.44283803
1.98700288921,150.876709655,1137.77349725
1.98371590756,151.126709655,1121.26311678
1.98043978286,151.376709655,1104.8968333
1.97717446142,151.626709655,1088.70778745
1.97391988989,151.876709655,1072.67795249
1.97067601527,152.126709655,1056.80617957
1.96744278491,152.376709655,1041.09217169
1.96422014651,152.626709655,1025.53463803
1.96100804809,152.876709655,1010.1489369
1.95780643805,153.126709655,994.900319441
1.95461526508,153.376709655,979.806639487
1.95143447824,153.626709655,964.868690478
1.94826402691,153.876709655,950.076773675
1.94510386078,154.126709655,935.426144407
1.9419539299,154.376709655,920.938206862
1.93881418462,154.626709655,906.599456365
1.93568457561,154.876709655,892.409084802
1.93256505386,155.126709655,878.366169252
1.92945557069,155.376709655,864.464149436
1.92635607772,155.626709655,850.713299225
1.92326652688,155.876709655,837.109568348
1.92018687041,156.126709655,823.650960468
1.91711706085,156.376709655,810.334557245
1.91405705106,156.626709655,797.160084793
1.91100679418,156.876709655,784.126820627
1.90796624366,157.126709655,771.233910529
1.90493535325,157.376709655,758.479579355
1.90191407697,157.626709655,745.864785454
1.89890236916,157.876709655,733.3877944
1.89590018444,158.126709655,721.047642893
1.8929074777,158.376709655,708.843440616
1.88992420414,158.626709655,696.774313247
1.88695031922,158.876709655,684.839757894
1.88398577869,159.126709655,673.038698256
1.88103053858,159.376709655,661.370253366
1.8780845552,159.626709655,649.833592731
1.87514778511,159.876709655,638.4279806
1.87222018516,160.126709655,627.152396266
1.86930171248,160.376709655,616.006110855
1.86639232444,160.626709655,604.988263383
1.86349197869,160.876709655,594.098038961
1.86060063314,161.126709655,583.334860734
1.85771824596,161.376709655,572.699442599
1.85484477559,161.626709655,562.198302156
1.85198018071,161.876709655,551.80370394
1.84912442026,162.126709655,541.540868808
1.84627745344,162.376709655,531.401528117
1.84343923969,162.626709655,521.385116215
1.84060973871,162.876709655,511.488423668
1.83778891043,163.126709655,501.712126707
1.83497671506,163.376709655,492.055313082
1.83217311301,163.626709655,482.517225061
1.82937806495,163.876709655,473.095670981
1.82659153181,164.126709655,463.792785775
1.82381347472,164.376709655,454.617132569
1.82104385508,164.626709655,445.535414598
1.8182826345,164.876709655,436.579034523
1.81552977484,165.126709655,427.73713976
1.81278523817,165.376709655,419.008240232
1.81004898681,165.626709655,410.39225236
1.8073209833,165.876709655,401.888097511
1.80460119039,166.126709655,393.508061446
1.80188957109,166.376709655,385.225577611
1.7991860886,166.626709655,377.052661394
1.79649070634,166.876709655,368.988843531
1.79380338797,167.126709655,361.034412421
1.79112409736,167.376709655,353.178124311
1.78845279858,167.626709655,345.44214617
1.78578945594,167.876709655,337.803870239
1.78313403394,168.126709655,330.274399336
1.7804864973,168.376709655,322.843119073
1.77784681094,168.626709655,315.521798568
1.77521494001,168.876709655,308.30356883
1.77259084985,169.126709655,301.187655742
1.769974506,169.376709655,294.173405584
1.76736587422,169.626709655,287.260134179
1.76476492044,169.876709655,280.44706715
1.76217161084,170.126709655,273.73357001
1.75958591175,170.376709655,267.118962023
1.75700778973,170.626709655,260.608138275
1.75443721151,170.876709655,254.189256372
1.75187414404,171.126709655,247.867207949
1.74931855445,171.376709655,241.644407044
1.74677041005,171.626709655,235.513997947
1.74422967836,171.876709655,229.478440766
1.74169632709,172.126709655,223.537000226
1.73917032411,172.376709655,217.689075429
1.73665163751,172.626709655,211.924943703
1.73414023554,172.876709655,206.262888527
1.73163608664,173.126709655,200.692659875
1.72913915944,173.376709655,195.212097532
1.72664942275,173.626709655,189.823095253
1.72416684555,173.876709655,184.523503642
1.72169139699,174.126709655,179.312922911
1.71922304643,174.376709655,174.190720609
1.71676176338,174.626709655,169.168448474
1.71430751751,174.876709655,164.20872585
1.71186027871,175.126709655,159.349546594
1.709420017,175.376709655,154.574372133
1.70698670258,175.626709655,149.884490668
1.70456030584,175.876709655,145.279210072
1.70214079731,176.126709655,140.756116822
1.6997281477,176.376709655,136.318197948
1.69732232789,176.626709655,131.963101746
1.69492330892,176.876709655,127.690228631
1.69253106199,177.126709655,123.506217057
1.69014555847,177.376709655,119.395987651
1.68776676988,177.626709655,115.366149496
1.68539466792,177.876709655,111.416133458
1.68302922442,178.126709655,107.545367011
1.6806704114,178.376709655,103.746441058
1.67831820101,178.626709655,100.032455273
1.67597256556,178.876709655,96.3963550297
1.67363347754,179.126709655,92.8386618867
1.67130090956,179.376709655,89.3553786778
1.6689748344,179.626709655,85.9543607357
1.66665522499,179.876709655,82.6232236222
1.6643420544,180.126709655,79.3667288447
1.66203529587,180.376709655,76.1843771607
1.65973492277,180.626709655,73.0756241572
1.65744090863,180.876709655,70.0526683185
1.6551532271,181.126709655,67.0893842955
1.65287185201,181.376709655,64.1980890541
1.65059675732,181.626709655,61.3781587257
1.64832791713,181.876709655,58.6137255491
1.64606530567,182.126709655,55.9311921168
1.64380889735,182.376709655,53.3258373757
1.64155866667,182.626709655,50.7860232824
1.63931458831,182.876709655,48.3149256922
1.63707663707,183.126709655,45.9079041107
1.63484478789,183.376709655,43.5726229974
1.63261901584,183.626709655,41.3045311293
1.63039929615,183.876709655,39.1030657851
1.62818560415,184.126709655,36.9677007372
1.62597791533,184.376709655,34.8975338392
1.6237762053,184.626709655,32.8928847016
1.6215804498,184.876709655,30.9528331941
1.61939062472,185.126709655,29.0770788113
1.61720670605,185.376709655,27.2658515799
1.61502866994,185.626709655,25.5172232067
1.61285649265,185.876709655,23.8312308768
1.61069015057,186.126709655,22.2073035946
1.60852962022,186.376709655,20.644970743
1.60637487825,186.626709655,19.1447103462
1.60422590141,186.876709655,17.7041259221
1.60208266662,187.126709655,16.3318439319
1.59994515088,187.376709655,15.0029493888
1.59781333133,187.626709655,13.7413683999
1.59568718523,187.876709655,12.5374774443
1.59356668997,188.126709655,11.3928369086
1.59145182305,188.376709655,10.3059562208
1.58934256208,188.626709655,9.27636972205
1.58723888481,188.876709655,8.31274627668
1.5851407691,189.126709655,7.3964090749
1.58304819292,189.376709655,6.53596460771
1.58096113435,189.626709655,5.73092394227
1.57887957161,189.876709655,4.98089105235
1.57680348302,190.126709655,4.27556696319
1.57473284701,190.376709655,3.63403887429
1.57266764213,190.626709655,3.04618362707
1.57060784703,190.876709655,2.51144730519
1.56855344049,191.126709655,2.02947731946
1.5665044014,191.376709655,1.59975238569
1.56446070874,191.626709655,1.22201969108
1.56242234161,191.876709655,0.896030411145
1.56038927923,192.126709655,0.620954528318
1.55836150092,192.376709655,0.402004061104
1.5563389861,192.626709655,0.227601674693
1.5543217143,192.876709655,0.102803472111
1.55230966517,193.126709655,0.0272355527299
1.55030281844,193.376709655,0.000446664340627
1.54830115398,193.626709655,0.0167901591491
1.54630465172,193.876709655,0.0860993706292
1.54431329173,194.126709655,0.203042356191
1.54232705416,194.376709655,0.367086573869
1.54034591928,194.626709655,0.57784662759
1.53836986744,194.876709655,0.853723444571
1.53639887912,195.126709655,1.13783412206
1.53443293486,195.376709655,1.4862073489
1.53247201534,195.626709655,1.8797940434
1.53051610132,195.876709655,2.31806034199
1.52856517364,196.126709655,2.80024221987
1.52661921328,196.376709655,3.32619941311
1.52467820128,196.626709655,3.89558833551
1.52274211879,196.876709655,4.50800500525
1.52081094705,197.126709655,5.17021834714
1.51888466741,197.376709655,5.8673572722
1.51696326131,197.626709655,6.60655500314
1.51504671026,197.876709655,7.37959374403
1.51313499589,198.126709655,8.20110613977
1.51122809992,198.376709655,9.06338203304
1.50932600414,198.626709655,9.96610805022
1.50742869047,198.876709655,10.9086612796
1.50553614088,199.126709655,11.8906895895
1.50364833745,199.376709655,12.9118734086
1.50176526236,199.626709655,13.971722072
1.49988689786,199.876709655,15.0698766085
1.4980132263,200.126709655,16.2149528799
1.49614423011,200.376709655,17.3886049258
1.49427989182,200.626709655,18.5994864269
1.49242019403,200.876709655,19.8472010668
1.49056511944,201.126709655,21.1314256934
1.48871465083,201.376709655,22.4431970461
1.48686877107,201.626709655,23.799157282
1.48502746311,201.876709655,25.1904652793
1.48319070998,202.126709655,26.6167700348
1.48135849481,202.376709655,28.0887587869
1.4795308008,202.626709655,29.5839449329
1.47770761124,202.876709655,31.1129972269
1.47588890948,203.126709655,32.6756122086
1.47407467899,203.376709655,34.2714051439
1.4722649033,203.626709655,35.8880108368
1.47045956601,203.876709655,37.549633577
1.46865865083,204.126709655,39.2427613331
1.46686214151,204.376709655,40.9676537339
1.46507002192,204.626709655,42.7239319682
1.46328227598,204.876709655,44.5305843865
1.46149888771,205.126709655,46.3487346693
1.45971984118,205.376709655,48.1959518486
1.45794512057,205.626709655,50.0710934982
1.45617471011,205.876709655,51.9937276224
1.45440859412,206.126709655,53.9315403714
1.452646757,206.376709655,55.8983300435
1.45088918321,206.626709655,57.8852488496
1.44913585729,206.876709655,59.9036597966
1.44738676388,207.126709655,61.9553721062
1.44564188765,207.376709655,64.0347356359
1.44390121337,207.626709655,66.1413888842
1.4421647259,207.876709655,68.2584800884
1.44043241013,208.126709655,70.4186562835
1.43870425105,208.376709655,72.6051224994
1.43698023372,208.626709655,74.8175892534
1.43526034327,208.876709655,77.0557536921
1.43354456489,209.126709655,79.3191894748
1.43183288387,209.376709655,81.6076713216
1.43012528553,209.626709655,83.9208014721
1.42842175529,209.876709655,86.2804961798
1.42672227863,210.126709655,88.6419717305
1.4250268411,210.376709655,91.02722206
1.42333542831,210.626709655,93.4358039413
1.42164802595,210.876709655,95.8675168774
1.41996461978,211.126709655,98.2999655691
1.41828519561,211.376709655,100.776876809
1.41660973933,211.626709655,103.276229621
1.4149382369,211.876709655,105.800435958
1.41327067434,212.126709655,108.342889873
1.41160703773,212.376709655,110.906488719
1.40994731323,212.626709655,113.492348217
1.40829148706,212.876709655,116.097602519
1.40663954548,213.126709655,118.723080952
1.40499147486,213.376709655,121.368555569
1.4033472616,213.626709655,124.048433128
1.40170689218,213.876709655,126.732997798
1.40007035312,214.126709655,129.436592046
1.39843763104,214.376709655,132.158925171
1.39680871259,214.626709655,134.899676994
1.39518358449,214.876709655,137.652750284
1.39356223353,215.126709655,140.429505529
1.39194464657,215.376709655,143.223825566
1.3903308105,215.626709655,146.035341528
1.38872071229,215.876709655,148.848879735
1.38711433898,216.126709655,151.693960798
1.38551167766,216.376709655,154.555408527
1.38391271546,216.626709655,157.432934266
1.3823174396,216.876709655,160.326221168
1.38072583735,217.126709655,163.253745754
1.37913789603,217.376709655,166.177762885
1.37755360303,217.626709655,169.116682405
1.37597294578,217.876709655,172.067711568
1.37439591178,218.126709655,175.019494035
1.37282248859,218.376709655,178.010845817
1.37125266383,218.626709655,181.006457413
1.36968642517,218.876709655,184.0155948
1.36812376032,219.126709655,187.028344966
1.36656465707,219.376709655,190.063566863
1.36500910327,219.626709655,193.111462188
1.36345708679,219.876709655,196.171707718
1.3619085956,220.126709655,199.243994237
1.36036361768,220.376709655,202.328469202
1.35882214111,220.626709655,205.424159463
1.35728415399,220.876709655,208.531075212
1.35574964448,221.126709655,211.649000509
1.3542186008,221.376709655,214.777569137
1.35269101123,221.626709655,217.916479073
1.35116686409,221.876709655,221.065568552
1.34964614776,222.126709655,224.224510636
1.34812885066,222.376709655,227.393077576
1.34661496127,222.626709655,230.570924529
1.34510446813,222.876709655,233.759592905
1.34359735983,223.126709655,236.970485203
1.34209362499,223.376709655,240.17453734
1.34059325231,223.626709655,243.386805381
1.33909623052,223.876709655,246.607001285
1.3376025484,224.126709655,249.818316808
1.3361121948,224.376709655,253.053845049
1.3346251586,224.626709655,256.296513348
1.33314142874,224.876709655,259.550880228
1.3316609942,225.126709655,262.804345408
1.33018384401,225.376709655,266.066872945
1.32870996726,225.626709655,269.335525755
1.32723935309,225.876709655,272.610054324
1.32577199066,226.126709655,275.890175253
1.3243078692,226.376709655,279.174086475
1.32284697799,226.626709655,282.464608732
1.32138930636,226.876709655,285.759995832
1.31993484366,227.126709655,289.060022994
1.31848357932,227.376709655,292.361813951
1.31703550279,227.626709655,295.682790161
1.31559060359,227.876709655,298.994846547
1.31414887127,228.126709655,302.31051588
1.31271029543,228.376709655,305.618600824
1.31127486571,228.626709655,308.947935992
1.3098425718,228.876709655,312.2631745
1.30841340344,229.126709655,315.614961738
1.30698735042,229.376709655,318.944526589
1.30556440255,229.626709655,322.276121916
1.3041445497,229.876709655,325.609420781
1.30272778179,230.126709655,328.919674108
1.30131408878,230.376709655,332.255702561
1.29990346065,230.626709655,335.592701297
1.29849588747,230.876709655,338.930449458
1.29709135931,231.126709655,342.268712035
1.29568986631,231.376709655,345.60668594
1.29429139863,231.626709655,348.945273384
1.2928959465,231.876709655,352.283605889
1.29150350016,232.126709655,355.621325525
1.29011404992,232.376709655,358.961317931
1.28872758612,232.626709655,362.297499129
1.28734409913,232.876709655,365.632532317
1.28596357939,233.126709655,368.966044965
1.28458601736,233.376709655,372.297909415
1.28321140354,233.626709655,375.625132302
1.28183972847,233.876709655,378.952841225
1.28047098275,234.126709655,382.278145854
1.279105157,234.376709655,385.600775845
1.27774224188,234.626709655,388.920894563
1.27638222811,234.876709655,392.23753117
1.27502510642,235.126709655,395.550751167
1.2736708676,235.376709655,398.860372195
1.27231950248,235.626709655,402.166121808
1.27097100192,235.876709655,405.467511241
1.26962535682,236.126709655,408.764818602
1.26828255811,236.376709655,412.057633065
1.26694259679,236.626709655,415.345628154
1.26560546386,236.876709655,418.632610151
1.26427115037,237.126709655,421.910801066
1.26293964743,237.376709655,425.182596112
1.26161094616,237.626709655,428.448678192
1.26028503772,237.876709655,431.708970254
1.25896191332,238.126709655,434.959350941
1.2576415642,238.376709655,438.20715493
1.25632398164,238.626709655,441.448313202
1.25500915695,238.876709655,444.682559621
1.25369708149,239.126709655,447.927786096
1.25238774663,239.376709655,451.147644369
1.2510811438,239.626709655,454.359926138
1.24977726446,239.876709655,457.564408158
1.2484761001,240.126709655,460.76078857
1.24717764225,240.376709655,463.960266597
1.24588188248,240.626709655,467.119464458
1.24458881238,240.876709655,470.290277609
1.24329842359,241.126709655,473.461139303
1.24201070778,241.376709655,476.613776901
1.24072565665,241.626709655,479.756985017
1.23944326193,241.876709655,482.874417925
1.2381635154,242.126709655,485.995219823
1.23688640887,242.376709655,489.108757744
1.23561193418,242.626709655,492.211963582
1.23434008319,242.876709655,495.304618116
1.23307084781,243.126709655,498.400664666
1.23180421999,243.376709655,501.471720572
1.23054019169,243.626709655,504.537300457
1.22927875492,243.876709655,507.579939606
1.22801990173,244.126709655,510.601695639
1.22676362417,244.376709655,513.626247226
1.22550991436,244.626709655,516.63872424
1.22425876443,244.876709655,519.638858168
1.22301016654,245.126709655,522.626405983
1.22176411291,245.376709655,525.601501734
1.22052059575,245.626709655,528.563227076
1.21927960733,245.876709655,531.534080324
1.21804113995,246.126709655,534.446874398
1.21680518593,246.376709655,537.37042196
1.21557173762,246.626709655,540.277969001
1.21434078743,246.876709655,543.171362207
1.21311232775,247.126709655,546.050495178
1.21188635106,247.376709655,548.949453719
1.21066284981,247.626709655,551.799349818
1.20944181653,247.876709655,554.634266026
1.20822324375,248.126709655,557.453994478
1.20700712404,248.376709655,560.237528273
1.20579345001,248.626709655,563.026193005
1.20458221428,248.876709655,565.817427899
1.20337340952,249.126709655,568.555563715
1.20216702841,249.376709655,571.30105966
1.20096306366,249.626709655,574.049361527
1.19976150804,249.876709655,576.732108714
1.19856235431,250.126709655,579.422320054
1.19736559528,250.376709655,582.095343457
1.19617122378,250.626709655,584.74782624
1.19497923268,250.876709655,587.385941574
1.19378961486,251.126709655,590.006247732
1.19260236325,251.376709655,592.628914156
1.19141747079,251.626709655,595.21023499
1.19023493046,251.876709655,597.739515666
1.18905473526,252.126709655,600.323044891
1.18787687822,252.376709655,602.851264138
1.1867013524,252.626709655,605.32398147
1.18552815089,252.876709655,607.813658464
1.18435726679,253.126709655,610.283832006
1.18318869326,253.376709655,612.734242881
1.18202242346,253.626709655,615.168083513
1.18085845057,253.876709655,617.578413482
1.17969676783,254.126709655,619.968391293
1.17853736848,254.376709655,622.337805941
1.1773802458,254.626709655,624.682760049
1.17622539308,254.876709655,627.010447184
1.17507280365,255.126709655,629.316906051
1.17392247087,255.376709655,631.601942471
1.17277438811,255.626709655,633.865413395
1.17162854878,255.876709655,636.122607847
1.17048494631,256.126709655,638.342223509
1.16934357416,256.376709655,640.539547368
1.16820442581,256.626709655,642.71437272
1.16706749476,256.876709655,644.850640395
1.16593277455,257.126709655,646.97993531
1.16480025874,257.376709655,649.086111882
1.16366994091,257.626709655,651.168951625
1.16254181466,257.876709655,653.278251445
1.16141587362,258.126709655,655.314114711
1.16029211147,258.376709655,657.32605508
1.15917052187,258.626709655,659.313838547
1.15805109853,258.876709655,661.253960966
1.15693383518,259.126709655,663.187724207
1.15581872558,259.376709655,665.106608018
1.15470576351,259.626709655,666.990835475
1.15359494276,259.876709655,668.851995878
1.15248625717,260.126709655,670.690456818
1.15137970058,260.376709655,672.485194088
1.15027526686,260.626709655,674.29040701
1.14917294992,260.876709655,676.058695106
1.14807274367,261.126709655,677.793523038
1.14697464206,261.376709655,679.501859873
1.14587863905,261.626709655,681.183573899
1.14478472864,261.876709655,682.838383132
1.14369290483,262.126709655,684.457359764
1.14260316167,262.376709655,686.057864959
1.1415154932,262.626709655,687.630874904
1.14042989352,262.876709655,689.180749732
1.13934635671,263.126709655,690.686835258
1.13826487692,263.376709655,692.176208248
1.13718544829,263.626709655,693.637336061
1.13610806498,263.876709655,695.059549441
1.13503272119,264.126709655,696.463481375
1.13395941114,264.376709655,697.838482853
1.13288812906,264.626709655,699.184346462
1.1318188692,264.876709655,700.527254128
1.13075162585,265.126709655,701.814348726
1.12968639331,265.376709655,703.071665515
1.12862316591,265.626709655,704.299103467
1.12756193797,265.876709655,705.467673508
1.12650270388,266.126709655,706.634599217
1.12544545801,266.376709655,707.770952023
1.12439019477,266.626709655,708.876577543
1.12333690859,266.876709655,709.95436971
1.12228559393,267.126709655,710.99786299
1.12123624525,267.376709655,712.009991141
1.12018885703,267.626709655,712.990521877
1.1191434238,267.876709655,713.940793876
1.11809994008,268.126709655,714.857585784
1.11705840043,268.376709655,715.74190798
1.11601879941,268.626709655,716.594105936
1.11498113163,268.876709655,717.412584139
1.11394539169,269.126709655,718.199456721
1.11291157422,269.376709655,718.955288371
1.11187967388,269.626709655,719.673865484
1.11084968534,269.876709655,720.36180652
1.10982160329,270.126709655,721.01517279
1.10879542244,270.376709655,721.634951738
1.10777113753,270.626709655,722.220694555
1.1067487433,270.876709655,722.771983819
1.10572823453,271.126709655,723.288962143
1.104709606,271.376709655,723.77128797
1.10369285252,271.626709655,724.218758454
1.10267796892,271.876709655,724.633767651
1.10166495005,272.126709655,725.010987663
1.10065379077,272.376709655,725.352699867
1.09964448597,272.626709655,725.658741649
1.09863703054,272.876709655,725.927685047
1.09763141942,273.126709655,726.161857095
1.09662764753,273.376709655,726.359685149
1.09562570985,273.626709655,726.521025456
1.09462560134,273.876709655,726.645697824
1.09362731701,274.126709655,726.765571804
1.09263085186,274.376709655,726.816314788
1.09163620092,274.626709655,726.829843173
1.09064335926,274.876709655,726.805789283
1.08965232193,275.126709655,726.710619922
1.08866308402,275.376709655,726.610913985
1.08767564063,275.626709655,726.473096909
1.0866899869,275.876709655,726.296991935
1.08570611794,276.126709655,726.119566301
1.08472402893,276.376709655,725.866226314
1.08374371504,276.626709655,725.573990711
1.08276517145,276.876709655,725.242561237
1.08178839338,277.126709655,724.833731045
1.08081337605,277.376709655,724.423313426
1.07984011471,277.626709655,723.97323467
1.07886860461,277.876709655,723.48317978
1.07789884104,278.126709655,722.96014211
1.07693081929,278.376709655,722.389643563
1.07596453467,278.626709655,721.825830997
1.0749999825,278.876709655,721.131533577
1.07403715814,279.126709655,720.438741983
1.07307605695,279.376709655,719.704755944
1.0721166743,279.626709655,718.917346851
1.07115900558,279.876709655,718.099347736
1.07020304622,280.126709655,717.240429845
1.06924879163,280.376709655,716.340471975
1.06829623726,280.626709655,715.41616727
1.06734537858,280.876709655,714.430568603
1.06639621105,281.126709655,713.382461604
1.06544873016,281.376709655,712.346200338
1.06450293144,281.626709655,711.197710896
1.0635588104,281.876709655,710.040290243
1.06261636258,282.126709655,708.839518494
1.06167558354,282.376709655,707.595265716
1.06073646884,282.626709655,706.306889662
1.05979901408,282.876709655,704.974452558
1.05886321487,283.126709655,703.597781504
1.05792906681,283.376709655,702.175820799
1.05699656554,283.626709655,700.709933525
1.05606570671,283.876709655,699.199174889
1.05513648598,284.126709655,697.643286459
1.05420889905,284.376709655,696.057890084
1.05328294159,284.626709655,694.411240558
1.05235860932,284.876709655,692.718913635
1.05143589796,285.126709655,690.989314968
1.05051480327,285.376709655,689.204830833
1.04959532098,285.626709655,687.373713182
1.04867744687,285.876709655,685.496771088
1.04776117672,286.126709655,683.573224361
1.04684650634,286.376709655,681.602165132
1.04593343154,286.626709655,679.583924256
1.04502194814,286.876709655,677.518338259
1.04411205199,287.126709655,675.390429981
1.04320373895,287.376709655,673.229386845
1.04229700489,287.626709655,671.057065848
1.04139184569,287.876709655,668.752514252
1.04048825726,288.126709655,666.459652778
1.03958623551,288.376709655,664.105222481
1.03868577637,288.626709655,661.702016836
1.03778687578,288.876709655,659.240311167
1.03688952971,289.126709655,656.738872405
1.03599373411,289.376709655,654.187964613
1.03509948498,289.626709655,651.587354466
1.03420677831,289.876709655,648.941494929
1.03331561012,290.126709655,646.244055253
1.03242597644,290.376709655,643.488672691
1.03153787329,290.626709655,640.691465481
1.03065129675,290.876709655,637.83928763
1.02976624287,291.126709655,634.966239543
1.02888270773,291.376709655,632.011998656
1.02800068744,291.626709655,629.006377283
1.02712017809,291.876709655,625.949165707
1.0262411758,292.126709655,622.803550961
1.02536367672,292.376709655,619.642474017
1.02448767699,292.626709655,616.429228226
1.02361317277,292.876709655,613.162951554
1.02274016023,293.126709655,609.841427482
1.02186863556,293.376709655,606.470214328
1.02099859496,293.626709655,603.046022149
1.02013003464,293.876709655,599.569741167
1.01926295083,294.126709655,596.049886437
1.01839733976,294.376709655,592.465247533
1.01753319769,294.626709655,588.826687556
1.01667052088,294.876709655,585.134056316
1.0158093056,295.126709655,581.406901587
1.01494954815,295.376709655,577.605346705
1.01409124483,295.626709655,573.749037237
1.01323439195,295.876709655,569.836488691
1.01237898584,296.126709655,565.849590211
1.01152502283,296.376709655,561.854083909
1.01067249928,296.626709655,557.776726677
1.00982141155,296.876709655,553.643554657
1.00897175602,297.126709655,549.429532205
1.00812352907,297.376709655,545.186563892
1.0072767271,297.626709655,540.89974843
1.00643134654,297.876709655,536.540921969
1.00558738379,298.126709655,532.125191685
1.0047448353,298.376709655,527.652315726
1.00390369752,298.626709655,523.106123757
1.0030639669,298.876709655,518.517283786
1.00222563992,299.126709655,513.871654851
1.00138871305,299.376709655,509.168029026
1.00055318281,299.626709655,504.391230438
0.999719045686,299.876709655,499.570967798
0.998886298205,300.126709655,494.69204577
0.998054936896,300.376709655,489.754240673
0.997224958301,300.626709655,484.757280818
0.996396358972,300.876709655,479.700936237
0.995569135476,301.126709655,474.585102644
0.994743284387,301.376709655,469.433672516
0.993918802293,301.626709655,464.198122706
0.993095685793,301.876709655,458.913471115
0.992273931498,302.126709655,453.546355913
0.991453536028,302.376709655,448.106940017
0.990634496015,302.626709655,442.629246026
0.989816808104,302.876709655,437.09053714
0.989000468949,303.126709655,431.490558333
0.988185475216,303.376709655,425.834132961
0.987371823581,303.626709655,420.110855913
0.986559510731,303.876709655,414.32566728
0.985748533366,304.126709655,408.478241162
0.984938888194,304.376709655,402.570242741
0.984130571936,304.626709655,396.5978456
0.983323581323,304.876709655,390.562545394
0.982517913095,305.126709655,384.464178257
0.981713564006,305.376709655,378.293220068
0.980910530818,305.626709655,372.068034408
0.980108810305,305.876709655,365.779123975
0.97930839925,306.126709655,359.43555295
0.978509294448,306.376709655,353.01933182
0.977711492704,306.626709655,346.538754539
0.976914990834,306.876709655,339.993649037
0.976119785663,307.126709655,333.373812064
0.975325874027,307.376709655,326.697692211
0.974533252773,307.626709655,319.961743679
0.973741918756,307.876709655,313.149096599
0.972951868845,308.126709655,306.287195878
0.972163099915,308.376709655,299.348413126
0.971375608854,308.626709655,292.343408627
0.97058939256,308.876709655,285.257809086
0.969804447938,309.126709655,278.119538119
0.969020771907,309.376709655,270.914434477
0.968238361393,309.626709655,263.642131597
0.967457213334,309.876709655,256.302097621
0.966677324676,310.126709655,248.89475343
0.965898692376,310.376709655,241.419663762
0.965121313402,310.626709655,233.900892727
0.964345184728,310.876709655,226.297991248
0.963570303341,311.126709655,218.616201008
0.962796666237,311.376709655,210.8672736
0.962024270422,311.626709655,203.049380483
0.961253112909,311.876709655,195.133689452
0.960483190725,312.126709655,187.17703898
0.959714500902,312.376709655,179.150666207
0.958947040484,312.626709655,171.054436288
0.958180806524,312.876709655,162.885355398
0.957415796085,313.126709655,154.64854899
0.956652006238,313.376709655,146.341094313
0.955889434065,313.626709655,137.962738883
0.955128076656,313.876709655,129.540012531
0.95436793111,314.126709655,121.019242045
0.953608994537,314.376709655,112.426869161
0.952851264054,314.626709655,103.737813624
0.952094736789,314.876709655,95.0018470379
0.951339409878,315.126709655,86.1930302706
0.950585280466,315.376709655,77.3120246131
0.949832345708,315.626709655,68.356108682
0.949080602768,315.876709655,59.3292976317
0.948330048818,316.126709655,50.2291946728
0.947580681039,316.376709655,41.0619983095
0.946832496622,316.626709655,31.8147564309
0.946085492765,316.876709655,22.4933029395
0.945339666678,317.126709655,13.0980058092
0.944595015575,317.376709655,3.62218889561
0.943851536684,317.626709655,-5.92243346575
0.943109227238,317.876709655,-15.5418708783
0.942368084481,318.126709655,-25.2364999355
0.941628105664,318.376709655,-35.0065592835
0.940889288046,318.626709655,-44.8523181378
0.940151628898,318.876709655,-54.773905426
0.939415125497,319.126709655,-64.7661688148
0.938679775127,319.376709655,-74.8402812168
0.937945575085,319.626709655,-84.9855986701
0.937212522672,319.876709655,-95.2188127167
0.936480615201,320.126709655,-105.515610798
0.93574984999,320.376709655,-115.897762406
0.935020224368,320.626709655,-126.357451077
0.934291735671,320.876709655,-136.907468122
0.933564381244,321.126709655,-147.523232102
0.932838158439,321.376709655,-158.218072528
0.932113064619,321.626709655,-168.990792511
0.931389097152,321.876709655,-179.842377518
0.930666253416,322.126709655,-190.770553226
0.929944530797,322.376709655,-201.773539598
0.929223926687,322.626709655,-212.87038723
0.92850443849,322.876709655,-224.04008886
0.927786063615,323.126709655,-235.292026912
0.927068799481,323.376709655,-246.622087582
0.926352643512,323.626709655,-258.032763677
0.925637593142,323.876709655,-269.524318062
0.924923645814,324.126709655,-281.094314314
0.924210798977,324.376709655,-292.748489456
0.923499050089,324.626709655,-304.484292138
0.922788396614,324.876709655,-316.304932106
0.922078836027,325.126709655,-328.204594271
0.921370365807,325.376709655,-340.186505565
0.920662983443,325.626709655,-352.251204784
0.919956686433,325.876709655,-364.365044819
0.919251472279,326.126709655,-376.596489354
0.918547338493,326.376709655,-388.903884242
0.917844282596,326.626709655,-401.307607757
0.917142302112,326.876709655,-413.792589537
0.916441394578,327.126709655,-426.358469499
0.915741557534,327.376709655,-439.008808531
0.915042788531,327.626709655,-451.74047494
0.914345085125,327.876709655,-464.560536431
0.91364844488,328.126709655,-477.47926585
0.912952865369,328.376709655,-490.470009784
0.91225834417,328.626709655,-503.546497238
0.91156487887,328.876709655,-516.728341765
0.910872467063,329.126709655,-529.977151732
0.91018110635,329.376709655,-543.312421802
0.90949079434,329.626709655,-556.705208525
0.908801528648,329.876709655,-570.214409761
0.908113306897,330.126709655,-583.810914389
0.907426126717,330.376709655,-597.495051292
0.906739985746,330.626709655,-611.292953255
0.906054881628,330.876709655,-625.15281998
0.905370812014,331.126709655,-639.101046062
0.904687774563,331.376709655,-653.141311336
0.904005766941,331.626709655,-667.269538028
0.903324786821,331.876709655,-681.484965163
0.902644831881,332.126709655,-695.789921139
0.90196589981,332.376709655,-710.18450738
0.9012879883,332.626709655,-724.669139349
0.900611095052,332.876709655,-739.244021874
0.899935217774,333.126709655,-753.863813609
0.89926035418,333.376709655,-768.663916738
0.898586501992,333.626709655,-783.511495508
0.897913658936,333.876709655,-798.423306663
0.897241822749,334.126709655,-813.45392144
0.896570991171,334.376709655,-828.576410481
0.895901161951,334.626709655,-843.791188116
0.895232332845,334.876709655,-859.125591963
0.894564501613,335.126709655,-874.525956329
0.893897666025,335.376709655,-890.018568685
0.893231823856,335.626709655,-905.594903974
0.892566972887,335.876709655,-921.274926689
0.891903110906,336.126709655,-937.048680795
0.891240235709,336.376709655,-952.916722472
0.890578345097,336.626709655,-968.891406248
0.889917436877,336.876709655,-984.862029195
0.889257508866,337.126709655,-1001.10065421
0.888598558882,337.376709655,-1017.34803805
0.887940584755,337.626709655,-1033.69106518
0.887283584317,337.876709655,-1050.12988161
0.886627555409,338.126709655,-1066.66488378
0.885972495877,338.376709655,-1083.29538388
0.885318403576,338.626709655,-1100.02344323
0.884665276363,338.876709655,-1116.84854315
0.884013112104,339.126709655,-1133.77081181
0.883361908673,339.376709655,-1150.7737608
0.882711663945,339.626709655,-1167.89149522
0.882062375807,339.876709655,-1185.10732114
0.881414042149,340.126709655,-1202.42165883
0.880766660867,340.376709655,-1219.83643722
0.880120229865,340.626709655,-1237.34919371
0.879474747052,340.876709655,-1254.96052914
0.878830210343,341.126709655,-1272.67146056
0.87818661766,341.376709655,-1290.44736545
0.87754396693,341.626709655,-1308.35617901
0.876902256086,341.876709655,-1326.36980736
0.876261483069,342.126709655,-1344.50166229
0.875621645824,342.376709655,-1362.71496505
0.874982742302,342.626709655,-1381.02962262
0.874344770461,342.876709655,-1399.44592382
0.873707728265,343.126709655,-1417.99350429
0.873071613684,343.376709655,-1436.61312391
0.872436424691,343.626709655,-1455.33381128
0.87180215927,343.876709655,-1474.16427282
0.871168815406,344.126709655,-1493.09338034
0.870536391094,344.376709655,-1512.12585988
0.869904884331,344.626709655,-1531.26220374
0.869274293123,344.876709655,-1550.5013936
0.868644615479,345.126709655,-1569.84606051
0.868015849416,345.376709655,-1589.29562345
0.867387992957,345.626709655,-1608.85071488
0.866761044127,345.876709655,-1628.51047544
0.866135000962,346.126709655,-1648.27586177
0.865509861499,346.376709655,-1668.14636443
0.864885623783,346.626709655,-1688.10235988
0.864262285865,346.876709655,-1708.18625432
0.863639845801,347.126709655,-1728.37705541
0.863018301652,347.376709655,-1748.67493525
0.862397651485,347.626709655,-1769.10291363
0.861777893373,347.876709655,-1789.61601193
0.861159025393,348.126709655,-1810.23708707
0.860541045631,348.376709655,-1830.88114774
0.859923952174,348.626709655,-1851.719099
0.859307743117,348.876709655,-1872.66603026
0.858692416562,349.126709655,-1893.72227898
0.858077970612,349.376709655,-1914.97435488
0.857464403379,349.626709655,-1936.25045126
0.85685171298,349.876709655,-1957.63675083
0.856239897537,350.126709655,-1979.12501599
0.855628955175,350.376709655,-2000.73285789
0.855018884029,350.626709655,-2022.451869
0.854409682235,350.876709655,-2044.28249777
0.853801347936,351.126709655,-2066.19171454
0.853193879282,351.376709655,-2088.24641295
0.852587274425,351.626709655,-2110.41361132
0.851981531525,351.876709655,-2132.6936199
0.851376648745,352.126709655,-2155.06291079
0.850772624256,352.376709655,-2177.56936139
0.850169456231,352.626709655,-2200.18971855
0.84956714285,352.876709655,-2222.99039889
0.848965682298,353.126709655,-2245.83940981
0.848365072765,353.376709655,-2268.8035072
0.847765312446,353.626709655,-2291.88261949
0.847166399541,353.876709655,-2315.0349592
0.846568332256,354.126709655,-2338.34533864
0.8459711088,354.376709655,-2361.77179973
0.84537472739,354.626709655,-2385.31474041
0.844779186246,354.876709655,-2409.01630525
0.844184483592,355.126709655,-2432.79321125
0.84359061766,355.376709655,-2456.68757857
0.842997586685,355.626709655,-2480.63085691
0.842405388907,355.876709655,-2504.83054224
0.841814022572,356.126709655,-2529.00983109
0.841223485929,356.376709655,-2553.44527366
0.840633777234,356.626709655,-2577.92755421
0.840044894747,356.876709655,-2602.53838704
0.839456836733,357.126709655,-2627.25653909
0.838869601461,357.376709655,-2652.1028578
0.838283187207,357.626709655,-2677.06990868
0.837697592249,357.876709655,-2702.16791402
0.837112814871,358.126709655,-2727.37286199
0.836528853363,358.376709655,-2752.7035884
0.835945706019,358.626709655,-2778.15618088
0.835363371137,358.876709655,-2803.73112849
0.83478184702,359.126709655,-2829.43289212
0.834201131976,359.376709655,-2855.2533564
0.833621224317,359.626709655,-2881.19718021
0.833042122363,359.876709655,-2907.26480502
0.832463824433,360.126709655,-2933.41577808
0.831886328855,360.376709655,-2959.73159133
0.831309633961,360.626709655,-2986.17216086
0.830733738086,360.876709655,-3012.77842393
0.830158639571,361.126709655,-3039.46963811
0.829584336761,361.376709655,-3066.286742
0.829010828005,361.626709655,-3093.22992441
0.828438111659,361.876709655,-3120.30004534
0.82786618608,362.126709655,-3147.49673537
0.827295049633,362.376709655,-3174.82065472
0.826724700685,362.626709655,-3202.19566398
0.826155137608,362.876709655,-3229.7753067
0.82558635878,363.126709655,-3257.48325359
0.825018362582,363.376709655,-3285.31992719
0.824451147399,363.626709655,-3313.2700788
0.823884711621,363.876709655,-3341.37054448
0.823319053645,364.126709655,-3369.59539007
0.822754171867,364.376709655,-3398.02560928
0.822190064693,364.626709655,-3426.51115686
0.821626730529,364.876709655,-3455.1281267
0.821064167787,365.126709655,-3483.87629287
0.820502374886,365.376709655,-3512.74777258
0.819941350244,365.626709655,-3541.77713479
0.819381092288,365.876709655,-3570.90389371
0.818821599447,366.126709655,-3600.15785565
0.818262870154,366.376709655,-3629.59293693
0.817704902848,366.626709655,-3659.13686497
0.817147695971,366.876709655,-3688.81470382
0.816591247969,367.126709655,-3718.64399585
0.816035557293,367.376709655,-3748.59098552
0.815480622399,367.626709655,-3778.67307724
0.814926441745,367.876709655,-3808.89258629
0.814373013794,368.126709655,-3839.24609906
0.813820337015,368.376709655,-3869.73582933
0.813268409878,368.626709655,-3900.3530901
0.812717230861,368.876709655,-3931.11775618
0.812166798442,369.126709655,-3962.02262712
0.811617111106,369.376709655,-3993.05386278
0.81106816734,369.626709655,-4024.21730602
0.810519965638,369.876709655,-4055.53302428
0.809972504495,370.126709655,-4087.00830794
0.809425782412,370.376709655,-4118.60250826
0.808879797894,370.626709655,-4150.33659844
0.808334549448,370.876709655,-4182.10148898
0.807790035588,371.126709655,-4214.22991574
0.807246254829,371.376709655,-4246.38583685
0.806703205693,371.626709655,-4278.68371492
0.806160886704,371.876709655,-4311.12297236
0.805619296389,372.126709655,-4343.70445413
0.805078433282,372.376709655,-4376.42847792
0.804538295919,372.626709655,-4409.29539228
0.80399888284,372.876709655,-4442.30595704
0.80346019259,373.126709655,-4475.46043622
0.802922223716,373.376709655,-4508.75922308
0.80238497477,373.626709655,-4542.18362837
0.801848444308,373.876709655,-4575.77222364
0.80131263089,374.126709655,-4609.50621998
0.800777533079,374.376709655,-4643.40526094
0.800243149443,374.626709655,-4677.43166478
0.799709478553,374.876709655,-4711.60482088
0.799176518984,375.126709655,-4745.92517895
0.798644269314,375.376709655,-4780.35497172
0.798112728126,375.626709655,-4814.97094955
0.797581894007,375.876709655,-4849.73543759
0.797051765547,376.126709655,-4884.68794113
0.796522341339,376.376709655,-4919.75057165
0.795993619981,376.626709655,-4954.96301794
0.795465600075,376.876709655,-4990.325699
0.794938280224,377.126709655,-5025.81428132
0.794411659039,377.376709655,-5061.4784231
0.793885735132,377.626709655,-5097.29091613
0.793360507118,377.876709655,-5133.27123956
0.792835973617,378.126709655,-5169.39652536
0.792312133253,378.376709655,-5205.67007068
0.791788984652,378.626709655,-5242.09745284
0.791266526446,378.876709655,-5278.6774649
0.790744757268,379.126709655,-5315.34260652
0.790223675756,379.376709655,-5352.23084412
0.789703280552,379.626709655,-5389.27409993
0.789183570301,379.876709655,-5426.47268323
0.788664543652,380.126709655,-5463.8715457
0.788146199255,380.376709655,-5501.38240316
0.787628535768,380.626709655,-5539.04995367
0.787111551849,380.876709655,-5576.90422709
0.786595246162,381.126709655,-5614.8867027
0.786079617371,381.376709655,-5653.02724133
0.785564664148,381.626709655,-5691.3264452
0.785050385164,381.876709655,-5729.75768401
0.784536779098,382.126709655,-5768.37517442
0.784023844628,382.376709655,-5807.15316691
0.783511580439,382.626709655,-5846.11770279
0.782999985217,382.876709655,-5885.21670184
0.782489057654,383.126709655,-5924.47762672
0.781978796442,383.376709655,-5963.89878755
0.781469200279,383.626709655,-6003.48467661
0.780960267866,383.876709655,-6043.23035214
0.780451997907,384.126709655,-6083.14068598
0.77994438911,384.376709655,-6123.20166192
0.779437440184,384.626709655,-6163.43977165
0.778931149844,384.876709655,-6203.84081772
0.778425516808,385.126709655,-6244.42169621
0.777920539797,385.376709655,-6285.15716179
0.777416217533,385.626709655,-6326.05676256
0.776912548746,385.876709655,-6367.12284622
0.776409532166,386.126709655,-6408.32173915
0.775907166526,386.376709655,-6449.72227428
0.775405450563,386.626709655,-6491.29086395
0.77490438302,386.876709655,-6533.02802263
0.774403962638,387.126709655,-6574.93394077
0.773904188166,387.376709655,-6617.00940985
0.773405058353,387.626709655,-6659.2761007
0.772906571953,387.876709655,-6701.69199081
0.772408727723,388.126709655,-6744.27867208
0.771911524422,388.376709655,-6787.05090063
0.771414960814,388.626709655,-6829.98088839
0.770919035665,388.876709655,-6873.08331278
0.770423747744,389.126709655,-6916.3587107
0.769929095825,389.376709655,-6959.80522967
0.769435078682,389.626709655,-7003.4279888
0.768941695094,389.876709655,-7047.22507674
0.768448943844,390.126709655,-7091.2006206
0.767956823717,390.376709655,-7135.34819395
0.767465333501,390.626709655,-7179.67161164
0.766974471988,390.876709655,-7224.17151933
0.766484237971,391.126709655,-7268.84232157
0.765994630248,391.376709655,-7313.696683
0.76550564762,391.626709655,-7358.72888816
0.765017288891,391.876709655,-7403.9437101
0.764529552867,392.126709655,-7449.33577835
0.764042438358,392.376709655,-7494.90531389
0.763555944177,392.626709655,-7540.65487543
0.76307006914,392.876709655,-7586.55486569
0.762584812065,393.126709655,-7632.6661724
0.762100171774,393.376709655,-7678.95929182
0.761616147092,393.626709655,-7725.43068236
0.761132736847,393.876709655,-7772.08851216
0.760649939869,394.126709655,-7818.92986516
0.760167754993,394.376709655,-7865.98863488
0.759686181054,394.626709655,-7913.19801642
0.759205216894,394.876709655,-7960.59230021
0.758724861353,395.126709655,-8008.17200977
0.758245113278,395.376709655,-8055.87649602
0.757765971517,395.626709655,-8103.83139792
0.757287434922,395.876709655,-8151.99647596
0.756809502346,396.126709655,-8200.34904621
0.756332172647,396.376709655,-8248.8640641
0.755855444684,396.626709655,-8297.56790225
0.755379317321,396.876709655,-8346.43298198
0.754903789424,397.126709655,-8395.51588608
0.75442885986,397.376709655,-8444.78926079
0.753954527501,397.626709655,-8494.25352428
0.753480791223,397.876709655,-8543.93986826
0.7530076499,398.126709655,-8593.78761811
0.752535102415,398.376709655,-8643.83213175
0.752063147648,398.626709655,-8694.03587047
0.751591784487,398.876709655,-8744.46340129
0.751121011819,399.126709655,-8795.08518481
0.750650828534,399.376709655,-8845.92947028
0.750181233528,399.626709655,-8896.93588594
0.749712225697,399.876709655,-8948.14991127
0.74924380394,400.126709655,-8999.52942928
0.748775967159,400.376709655,-9051.06080634
0.748308714259,400.626709655,-9102.86047513
0.747842044148,400.876709655,-9154.85864124
0.747375955737,401.126709655,-9207.07274145
0.746910447937,401.376709655,-9259.46958237
0.746445519665,401.626709655,-9312.06640331
0.74598116984,401.876709655,-9364.94379369
0.745517397382,402.126709655,-9417.94322081
0.745054201216,402.376709655,-9471.14435236
0.744591580267,402.626709655,-9524.54894316
0.744129533466,402.876709655,-9578.07632253
0.743668059743,403.126709655,-9631.88706219
0.743207158034,403.376709655,-9685.902031
0.742746827275,403.626709655,-9740.2032934
0.742287066407,403.876709655,-9794.62897093
0.741827874371,404.126709655,-9849.2607918
0.741369250112,404.376709655,-9904.08947454
0.740911192579,404.626709655,-9959.13564827
0.740453700721,404.876709655,-10014.3896798
0.739996773491,405.126709655,-10069.8626795
0.739540409845,405.376709655,-10125.5347825
0.73908460874,405.626709655,-10181.4167563
0.738629369137,405.876709655,-10237.5082903
0.738174689999,406.126709655,-10293.8120236
0.737720570292,406.376709655,-10350.3276355
0.737267008983,406.626709655,-10407.0557386
0.736814005044,406.876709655,-10463.9944053
0.736361557447,407.126709655,-10521.1532509
0.735909665169,407.376709655,-10578.5181478
0.735458327188,407.626709655,-10636.0950271
0.735007542484,407.876709655,-10693.8725005
0.734557310041,408.126709655,-10751.9110414
0.734107628844,408.376709655,-10810.1139921
0.733658497882,408.626709655,-10868.5646386
0.733209916145,408.876709655,-10927.2336935
0.732761882627,409.126709655,-10986.1546708
0.732314396323,409.376709655,-11045.2622255
0.731867456232,409.626709655,-11104.5900667
0.731421061353,409.876709655,-11164.1387754
0.730975210691,410.126709655,-11223.9181464
0.730529903249,410.376709655,-11283.9106592
0.730085138036,410.626709655,-11344.126294
0.729640914063,410.876709655,-11404.5595748
0.729197230342,411.126709655,-11465.2231481
0.728754085887,411.376709655,-11526.1115601
0.728311479717,411.626709655,-11587.2326548
0.727869410851,411.876709655,-11648.5732079
0.727427878312,412.126709655,-11710.1407262
0.726986881123,412.376709655,-11771.9355513
0.726546418312,412.626709655,-11833.7976317
0.726106488909,412.876709655,-11896.1489245
0.725667091944,413.126709655,-11958.5322546
0.725228226453,413.376709655,-12021.4043947
0.72478989147,413.626709655,-12084.3485303
0.724352086035,413.876709655,-12147.5245088
0.723914809189,414.126709655,-12210.9326104
0.723478059975,414.376709655,-12274.5744077
0.723041837438,414.626709655,-12338.4503122
0.722606140627,414.876709655,-12402.4240119
0.722170968592,415.126709655,-12466.7701119
0.721736320385,415.376709655,-12531.3523325
0.72130219506,415.626709655,-12596.2406715
0.720868591676,415.876709655,-12661.2974711
0.720435509291,416.126709655,-12726.5926912
0.720002946967,416.376709655,-12792.1268909
0.719570903767,416.626709655,-12857.9717301
0.719139378758,416.876709655,-12923.9864603
0.718708371008,417.126709655,-12990.2423755
0.718277879587,417.376709655,-13056.7364318
0.717847903568,417.626709655,-13123.4773764
0.717418442027,417.876709655,-13190.4656915
0.716989494039,418.126709655,-13257.6939696
0.716561058686,418.376709655,-13325.1680264
0.716133135048,418.626709655,-13392.8874493
0.715705722208,418.876709655,-13460.8222896
0.715278819254,419.126709655,-13529.0455921
0.714852425273,419.376709655,-13597.4970814
0.714426539355,419.626709655,-13666.2186509
0.714001160594,419.876709655,-13735.1961702
0.713576288082,420.126709655,-13804.4066877
0.713151920919,420.376709655,-13873.8702868
0.712728058201,420.626709655,-13943.583732
0.712304699031,420.876709655,-14013.549973
0.711881842511,421.126709655,-14083.740048
0.711459487748,421.376709655,-14154.2142218
0.711037633847,421.626709655,-14224.9433493
0.71061627992,421.876709655,-14295.9574955
0.710195425077,422.126709655,-14367.1993865
'''

DEFAULT_DINT_PASQUAZI_FIG25_SIN_DW_CSV = '''Wavelength [um], Frequency [THz], Dint/2pi [GHz]
3.933427839560,76.216590268886,-833.173875725777
3.921798789726,76.442590268886,-825.731737591759
3.910238299003,76.668590268886,-818.337255466055
3.898745762890,76.894590268886,-810.990232831400
3.887320583968,77.120590268886,-803.690473563944
3.875962171801,77.346590268886,-796.437781933259
3.864669942835,77.572590268886,-789.231962602332
3.853443320295,77.798590268886,-782.072820627570
3.842281734090,78.024590268886,-774.960161458796
3.831184620715,78.250590268886,-767.893790939255
3.820151423155,78.476590268886,-760.873515305606
3.809181590793,78.702590268886,-753.899141187928
3.798274579322,78.928590268886,-746.970475609718
3.787429850646,79.154590268886,-740.087325987891
3.776646872800,79.380590268886,-733.249500132781
3.765925119860,79.606590268886,-726.456806248137
3.755264071857,79.832590268886,-719.709052931131
3.744663214692,80.058590268886,-713.006049172349
3.734122040057,80.284590268886,-706.347604355796
3.723640045350,80.510590268886,-699.733528258897
3.713216733597,80.736590268886,-693.163631052494
3.702851613373,80.962590268886,-686.637723300846
3.692544198725,81.188590268886,-680.155615961631
3.682294009094,81.414590268886,-673.717120385945
3.672100569247,81.640590268886,-667.322048318303
3.661963409192,81.866590268886,-660.970211896638
3.651882064119,82.092590268886,-654.661423652299
3.641856074318,82.318590268886,-648.395496510055
3.631884985115,82.544590268886,-642.172243788094
3.621968346802,82.770590268886,-635.991479198019
3.612105714569,82.996590268886,-629.853016844854
3.602296648439,83.222590268886,-623.756671227040
3.592540713199,83.448590268886,-617.702257236436
3.582837478339,83.674590268886,-611.689590158320
3.573186517988,83.900590268886,-605.718485671386
3.563587410851,84.126590268886,-599.788759847749
3.554039740147,84.352590268886,-593.900229152940
3.544543093553,84.578590268886,-588.052710445908
3.535097063136,84.804590268886,-582.246020979023
3.525701245305,85.030590268886,-576.479978398069
3.516355240744,85.256590268886,-570.754400742250
3.507058654365,85.482590268886,-565.069106444190
3.497811095241,85.708590268886,-559.423914329929
3.488612176563,85.934590268886,-553.818643618924
3.479461515577,86.160590268886,-548.253113924052
3.470358733536,86.386590268886,-542.727145251609
3.461303455644,86.612590268886,-537.240558001306
3.452295311010,86.838590268886,-531.793172966276
3.443333932591,87.064590268886,-526.384811333066
3.434418957147,87.290590268886,-521.015294681644
3.425550025189,87.516590268886,-515.684444985396
3.416726780931,87.742590268886,-510.392084611124
3.407948872247,87.968590268886,-505.138036319050
3.399215950616,88.194590268886,-499.922123262815
3.390527671081,88.420590268886,-494.744168989474
3.381883692206,88.646590268886,-489.603997439505
3.373283676024,88.872590268886,-484.501432946801
3.364727287999,89.098590268886,-479.436300238675
3.356214196982,89.324590268886,-474.408424435856
3.347744075163,89.550590268886,-469.417631052493
3.339316598036,89.776590268886,-464.463745996153
3.330931444355,90.002590268886,-459.546595567818
3.322588296089,90.228590268886,-454.666006461894
3.314286838389,90.454590268886,-449.821805766199
3.306026759542,90.680590268886,-445.013820961974
3.297807750937,90.906590268886,-440.241879923875
3.289629507023,91.132590268886,-435.505810919977
3.281491725273,91.358590268886,-430.805442611773
3.273394106146,91.584590268886,-426.140604054175
3.265336353050,91.810590268886,-421.511124695512
3.257318172307,92.036590268886,-416.916834377532
3.249339273115,92.262590268886,-412.357563335400
3.241399367516,92.488590268886,-407.833142197700
3.233498170359,92.714590268886,-403.343401986435
3.225635399266,92.940590268886,-398.888174117023
3.217810774600,93.166590268886,-394.467290398304
3.210024019431,93.392590268886,-390.080583032534
3.202274859501,93.618590268886,-385.727884615386
3.194563023196,93.844590268886,-381.409028135953
3.186888241512,94.070590268886,-377.123846976747
3.179250248022,94.296590268886,-372.872174913695
3.171648778849,94.522590268886,-368.653846116145
3.164083572634,94.748590268886,-364.468695146861
3.156554370503,94.974590268886,-360.316556962026
3.149060916043,95.200590268886,-356.197266911243
3.141602955269,95.426590268886,-352.110660737529
3.134180236596,95.652590268886,-348.056574577323
3.126792510812,95.878590268886,-344.034844960479
3.119439531049,96.104590268886,-340.045308810272
3.112121052754,96.330590268886,-336.087803443393
3.104836833666,96.556590268886,-332.162166569952
3.097586633785,96.782590268886,-328.268236293478
3.090370215349,97.008590268886,-324.405851110915
3.083187342806,97.234590268886,-320.574849912629
3.076037782789,97.460590268886,-316.775071982402
3.068921304089,97.686590268886,-313.006356997434
3.061837677634,97.912590268886,-309.268545028343
3.054786676460,98.138590268886,-305.561476539167
3.047768075692,98.364590268886,-301.884992387361
3.040781652512,98.590590268886,-298.238933823796
3.033827186146,98.816590268886,-294.623142492764
3.026904457831,99.042590268886,-291.037460431974
3.020013250797,99.268590268886,-287.481730072554
3.013153350246,99.494590268886,-283.955794239048
3.006324543323,99.720590268886,-280.459496149421
2.999526619102,99.946590268886,-276.992679415054
2.992759368559,100.172590268886,-273.555188040746
2.986022584551,100.398590268886,-270.146866424715
2.979316061799,100.624590268886,-266.767559358598
2.972639596860,100.850590268886,-263.417112027448
2.965992988114,101.076590268886,-260.095370009738
2.959376035739,101.302590268886,-256.802179277357
2.952788541691,101.528590268886,-253.537386195615
2.946230309687,101.754590268886,-250.300837523237
2.939701145184,101.980590268886,-247.092380412369
2.933200855359,102.206590268886,-243.911862408572
2.926729249090,102.432590268886,-240.759131450828
2.920286136940,102.658590268886,-237.634035871537
2.913871331134,102.884590268886,-234.536424396514
2.907484645546,103.110590268886,-231.466146144995
2.901125895677,103.336590268886,-228.423050629634
2.894794898637,103.562590268886,-225.406987756501
2.888491473131,103.788590268886,-222.417807825087
2.882215439440,104.014590268886,-219.455361528299
2.875966619401,104.240590268886,-216.519499952463
2.869744836396,104.466590268886,-213.610074577322
2.863549915329,104.692590268886,-210.726937276039
2.857381682614,104.918590268886,-207.869940315194
2.851239966158,105.144590268886,-205.038936354785
2.845124595345,105.370590268886,-202.233778448229
2.839035401016,105.596590268886,-199.454320042359
2.832972215462,105.822590268886,-196.700414977428
2.826934872400,106.048590268886,-193.971917487108
2.820923206963,106.274590268886,-191.268682198487
2.814937055683,106.500590268886,-188.590564132071
2.808976256477,106.726590268886,-185.937418701786
2.803040648630,106.952590268886,-183.309101714976
2.797130072787,107.178590268886,-180.705469372400
2.791244370929,107.404590268886,-178.126378268240
2.785383386369,107.630590268886,-175.571685390091
2.779546963729,107.856590268886,-173.041248118970
2.773734948933,108.082590268886,-170.534924229311
2.767947189191,108.308590268886,-168.052571888965
2.762183532985,108.534590268886,-165.594049659202
2.756443830057,108.760590268886,-163.159216494711
2.750727931394,108.986590268886,-160.747931743597
2.745035689218,109.212590268886,-158.360055147385
2.739366956970,109.438590268886,-155.995446841017
2.733721589302,109.664590268886,-153.653967352854
2.728099442058,109.890590268886,-151.335477604674
2.722500372269,110.116590268886,-149.039838911675
2.716924238134,110.342590268886,-146.766912982471
2.711370899013,110.568590268886,-144.516561919094
2.705840215415,110.794590268886,-142.288648216997
2.700332048982,111.020590268886,-140.083034765048
2.694846262482,111.246590268886,-137.899584845535
2.689382719796,111.472590268886,-135.738162134163
2.683941285905,111.698590268886,-133.598630700055
2.678521826882,111.924590268886,-131.480855005754
2.673124209879,112.150590268886,-129.384699907219
2.667748303118,112.376590268886,-127.310030653829
2.662393975877,112.602590268886,-125.256712888378
2.657061098482,112.828590268886,-123.224612647082
2.651749542296,113.054590268886,-121.213596359572
2.646459179709,113.280590268886,-119.223530848898
2.641189884128,113.506590268886,-117.254283331531
2.635941529963,113.732590268886,-115.305721417355
2.630713992623,113.958590268886,-113.377713109676
2.625507148504,114.184590268886,-111.470126805216
2.620320874977,114.410590268886,-109.582831294117
2.615155050380,114.636590268886,-107.715695759937
2.610009554009,114.862590268886,-105.868589779654
2.604884266108,115.088590268886,-104.041383323662
2.599779067861,115.314590268886,-102.233946755776
2.594693841379,115.540590268886,-100.446150833227
2.589628469697,115.766590268886,-98.677866706665
2.584582836757,115.992590268886,-96.928965920156
2.579556827409,116.218590268886,-95.199320411188
2.574550327394,116.444590268886,-93.488802510664
2.569563223337,116.670590268886,-91.797284942906
2.564595402744,116.896590268886,-90.124640825654
2.559646753984,117.122590268886,-88.470743670067
2.554717166291,117.348590268886,-86.835467380721
2.549806529748,117.574590268886,-85.218686255611
2.544914735280,117.800590268886,-83.620274986149
2.540041674652,118.026590268886,-82.040108657166
2.535187240451,118.252590268886,-80.478062746911
2.530351326089,118.478590268886,-78.934013127051
2.525533825785,118.704590268886,-77.407836062672
2.520734634565,118.930590268886,-75.899408212276
2.515953648250,119.156590268886,-74.408606627784
2.511190763450,119.382590268886,-72.935308754537
2.506445877558,119.608590268886,-71.479392431292
2.501718888739,119.834590268886,-70.040735890224
2.497009695926,120.060590268886,-68.619217756927
2.492318198810,120.286590268886,-67.214717050414
2.487644297837,120.512590268886,-65.827113183114
2.482987894197,120.738590268886,-64.456285960875
2.478348889816,120.964590268886,-63.102115582964
2.473727187357,121.190590268886,-61.764482642065
2.469122690203,121.416590268886,-60.443268124281
2.464535302457,121.642590268886,-59.138353409132
2.459964928933,121.868590268886,-57.849620269557
2.455411475150,122.094590268886,-56.576950871913
2.450874847325,122.320590268886,-55.320227775975
2.446354952367,122.546590268886,-54.079333934935
2.441851697870,122.772590268886,-52.854152695406
2.437364992108,122.998590268886,-51.644567797416
2.432894744027,123.224590268886,-50.450463374413
2.428440863240,123.450590268886,-49.271723953262
2.424003260021,123.676590268886,-48.108234454247
2.419581845298,123.902590268886,-46.959880191070
2.415176530649,124.128590268886,-45.826546870851
2.410787228294,124.354590268886,-44.708120594127
2.406413851090,124.580590268886,-43.604487854855
2.402056312524,124.806590268886,-42.515535540409
2.397714526711,125.032590268886,-41.441150931582
2.393388408383,125.258590268886,-40.381221702583
2.389077872889,125.484590268886,-39.335635921041
2.384782836186,125.710590268886,-38.304282048004
2.380503214831,125.936590268886,-37.287048937935
2.376238925985,126.162590268886,-36.283825838718
2.371989887396,126.388590268886,-35.294502391655
2.367756017402,126.614590268886,-34.318968631463
2.363537234922,126.840590268886,-33.357114986280
2.359333459453,127.066590268886,-32.408832277662
2.355144611063,127.292590268886,-31.474011720582
2.350970610386,127.518590268886,-30.552544923431
2.346811378619,127.744590268886,-29.644323888020
2.342666837514,127.970590268886,-28.749241009577
2.338536909376,128.196590268886,-27.867189076746
2.334421517058,128.422590268886,-26.998061271593
2.330320583952,128.648590268886,-26.141751169599
2.326234033990,128.874590268886,-25.298152739666
2.322161791636,129.100590268886,-24.467160344110
2.318103781880,129.326590268886,-23.648668738669
2.314059930240,129.552590268886,-22.842573072497
2.310030162748,129.778590268886,-22.048768888168
2.306014405952,130.004590268886,-21.267152121671
2.302012586912,130.230590268886,-20.497619102417
2.298024633191,130.456590268886,-19.740066553231
2.294050472853,130.682590268886,-18.994391590360
2.290090034460,130.908590268886,-18.260491723467
2.286143247066,131.134590268886,-17.538264855632
2.282210040213,131.360590268886,-16.827609283356
2.278290343928,131.586590268886,-16.128423696557
2.274384088716,131.812590268886,-15.440607178569
2.270491205560,132.038590268886,-14.764059206148
2.266611625912,132.264590268886,-14.098679649465
2.262745281696,132.490590268886,-13.444368772109
2.258892105295,132.716590268886,-12.801027231091
2.255052029554,132.942590268886,-12.168556076835
2.251224987774,133.168590268886,-11.546856753186
2.247410913709,133.394590268886,-10.935831097407
2.243609741558,133.620590268886,-10.335381340179
2.239821405967,133.846590268886,-9.745410105600
2.236045842023,134.072590268886,-9.165820411188
2.232282985248,134.298590268886,-8.596515667877
2.228532771598,134.524590268886,-8.037399680020
2.224795137459,134.750590268886,-7.488376645389
2.221070019644,134.976590268886,-6.949351155174
2.217357355386,135.202590268886,-6.420228193981
2.213657082340,135.428590268886,-5.900913139837
2.209969138573,135.654590268886,-5.391311764185
2.206293462567,135.880590268886,-4.891330231887
2.202629993212,136.106590268886,-4.400875101223
2.198978669801,136.332590268886,-3.919853323891
2.195339432032,136.558590268886,-3.448172245008
2.191712220000,136.784590268886,-2.985739603107
2.188096974195,137.010590268886,-2.532463530141
2.184493635499,137.236590268886,-2.088252551481
2.180902145184,137.462590268886,-1.653015585914
2.177322444907,137.688590268886,-1.226661945649
2.173754476706,137.914590268886,-0.809101336310
2.170198182999,138.140590268886,-0.400243856940
2.166653506583,138.366590268886,-0.000000000000
2.163120390624,138.592590268886,0.391719348631
2.159598778660,138.818590268886,0.775002909655
2.156088614597,139.044590268886,1.149939010357
2.152589842703,139.270590268886,1.516615584603
2.149102407608,139.496590268886,1.875120172842
2.145626254302,139.722590268886,2.225539922103
2.142161328128,139.948590268886,2.567961585999
2.138707574782,140.174590268886,2.902471524725
2.135264940310,140.400590268886,3.229155705054
2.131833371106,140.626590268886,3.548099700346
2.128412813905,140.852590268886,3.859388690540
2.125003215786,141.078590268886,4.163107462158
2.121604524167,141.304590268886,4.459340408302
2.118216686799,141.530590268886,4.748171528659
2.114839651768,141.756590268886,5.029684429494
2.111473367490,141.982590268886,5.303962323658
2.108117782710,142.208590268886,5.571088030582
2.104772846498,142.434590268886,5.831143976277
2.101438508245,142.660590268886,6.084212193339
2.098114717664,142.886590268886,6.330374320944
2.094801424785,143.112590268886,6.569711604851
2.091498579954,143.338590268886,6.802304897400
2.088206133828,143.564590268886,7.028234657513
2.084924037375,143.790590268886,7.247580950695
2.081652241872,144.016590268886,7.460423449031
2.078390698899,144.242590268886,7.666841431189
2.075139360341,144.468590268886,7.866913782420
2.071898178383,144.694590268886,8.060718994554
2.068667105508,144.920590268886,8.248335166006
2.065446094494,145.146590268886,8.429840001770
2.062235098415,145.372590268886,8.605310813425
2.059034070635,145.598590268886,8.774824519128
2.055842964806,145.824590268886,8.938457643622
2.052661734869,146.050590268886,9.096286318229
2.049490335049,146.276590268886,9.248386280855
2.046328719852,146.502590268886,9.394832875985
2.043176844067,146.728590268886,9.535701054688
2.040034662758,146.954590268886,9.671065374616
2.036902131268,147.180590268886,9.801000000000
2.033779205212,147.406590268886,9.925578701655
2.030665840476,147.632590268886,10.044874856976
2.027561993218,147.858590268886,10.158961449942
2.024467619863,148.084590268886,10.267911071114
2.021382677100,148.310590268886,10.371795917631
2.018307121884,148.536590268886,10.470687793219
2.015240911429,148.762590268886,10.564658108183
2.012184003211,148.988590268886,10.653777879411
2.009136354962,149.214590268886,10.738117730371
2.006097924671,149.440590268886,10.817747891115
2.003068670579,149.666590268886,10.892738198276
2.000048551181,149.892590268886,10.963158095069
1.997037525219,150.118590268886,11.029076631292
1.994035551687,150.344590268886,11.090562463322
1.991042589822,150.570590268886,11.147683854121
1.988058599107,150.796590268886,11.200508673230
1.985083539265,151.022590268886,11.249104396775
1.982117370265,151.248590268886,11.293538107462
1.979160052309,151.474590268886,11.333876494579
1.976211545839,151.700590268886,11.370185853996
1.973271811534,151.926590268886,11.402532088165
1.970340810302,152.152590268886,11.430980706120
1.967418503288,152.378590268886,11.455596823477
1.964504851864,152.604590268886,11.476445162432
1.961599817632,152.830590268886,11.493590051767
1.958703362419,153.056590268886,11.507095426842
1.955815448278,153.282590268886,11.517024829601
1.952936037487,153.508590268886,11.523441408568
1.950065092545,153.734590268886,11.526407918851
1.947202576169,153.960590268886,11.525986722139
1.944348451297,154.186590268886,11.522239786702
1.941502681083,154.412590268886,11.515228687393
1.938665228897,154.638590268886,11.505014605647
1.935836058323,154.864590268886,11.491658329481
1.933015133157,155.090590268886,11.475220253492
1.930202417404,155.316590268886,11.455760378862
1.927397875281,155.542590268886,11.433338313351
1.924601471211,155.768590268886,11.408013271304
1.921813169824,155.994590268886,11.379844073648
1.919032935953,156.220590268886,11.348889147889
1.916260734636,156.446590268886,11.315206528118
1.913496531113,156.672590268886,11.278853855006
1.910740290822,156.898590268886,11.239888375806
1.907991979403,157.124590268886,11.198366944354
1.905251562690,157.350590268886,11.154346021068
1.902519006716,157.576590268886,11.107881672945
1.899794277706,157.802590268886,11.059029573568
1.897077342080,158.028590268886,11.007845003098
1.894368166450,158.254590268886,10.954382848282
1.891666717617,158.480590268886,10.898697602444
1.888972962573,158.706590268886,10.840843365495
1.886286868494,158.932590268886,10.780873843924
1.883608402748,159.158590268886,10.718842350804
1.880937532883,159.384590268886,10.654801805789
1.878274226635,159.610590268886,10.588804735115
1.875618451918,159.836590268886,10.520903271599
1.872970176831,160.062590268886,10.451149154643
1.870329369652,160.288590268886,10.379593730227
1.867695998836,160.514590268886,10.306287950915
1.865070033017,160.740590268886,10.231282375852
1.862451441005,160.966590268886,10.154627170766
1.859840191785,161.192590268886,10.076372107967
1.857236254515,161.418590268886,9.996566566345
1.854639598525,161.644590268886,9.915259531373
1.852050193318,161.870590268886,9.832499595107
1.849468008566,162.096590268886,9.748334956183
1.846893014111,162.322590268886,9.662813419820
1.844325179961,162.548590268886,9.575982397818
1.841764476291,162.774590268886,9.487888908560
1.839210873442,163.000590268886,9.398579577010
1.836664341920,163.226590268886,9.308100634715
1.834124852392,163.452590268886,9.216497919802
1.831592375689,163.678590268886,9.123816876981
1.829066882802,163.904590268886,9.030102557546
1.826548344881,164.130590268886,8.935399619368
1.824036733237,164.356590268886,8.839752326904
1.821532019336,164.582590268886,8.743204551192
1.819034174802,164.808590268886,8.645799769850
1.816543171414,165.034590268886,8.547581067081
1.814058981105,165.260590268886,8.448591133667
1.811581575963,165.486590268886,8.348872266974
1.809110928226,165.712590268886,8.248466370947
1.806647010284,165.938590268886,8.147414956117
1.804189794678,166.164590268886,8.045759139595
1.801739254098,166.390590268886,7.943539645071
1.799295361381,166.616590268886,7.840796802822
1.796858089513,166.842590268886,7.737570549703
1.794427411625,167.068590268886,7.633900429154
1.792003300992,167.294590268886,7.529825591193
1.789585731036,167.520590268886,7.425384792423
1.787174675321,167.746590268886,7.320616396028
1.784770107552,167.972590268886,7.215558371774
1.782372001577,168.198590268886,7.110248296008
1.779980331384,168.424590268886,7.004723351660
1.777595071100,168.650590268886,6.899020328242
1.775216194990,168.876590268886,6.793175621847
1.772843677458,169.102590268886,6.687225235149
1.770477493044,169.328590268886,6.581204777407
1.768117616424,169.554590268886,6.475149464460
1.765764022408,169.780590268886,6.369094118727
1.763416685940,170.006590268886,6.263073169212
1.761075582099,170.232590268886,6.157120651500
1.758740686093,170.458590268886,6.051270207758
1.756411973264,170.684590268886,5.945555086732
1.754089419084,170.910590268886,5.840008143755
1.751772999152,171.136590268886,5.734661840738
1.749462689200,171.362590268886,5.629548246175
1.747158465083,171.588590268886,5.524699035142
1.744860302788,171.814590268886,5.420145489297
1.742568178425,172.040590268886,5.315918496881
1.740282068230,172.266590268886,5.212048552713
1.738001948563,172.492590268886,5.108565758199
1.735727795909,172.718590268886,5.005499821323
1.733459586876,172.944590268886,4.902880056652
1.731197298193,173.170590268886,4.800735385337
1.728940906711,173.396590268886,4.699094335107
1.726690389400,173.622590268886,4.597985040276
1.724445723352,173.848590268886,4.497435241739
1.722206885778,174.074590268886,4.397472286972
1.719973854004,174.300590268886,4.298123130035
1.717746605478,174.526590268886,4.199414331566
1.715525117761,174.752590268886,4.101372058790
1.713309368531,174.978590268886,4.004022085509
1.711099335582,175.204590268886,3.907389792111
1.708894996822,175.430590268886,3.811500165563
1.706696330272,175.656590268886,3.716377799416
1.704503314067,175.882590268886,3.622046893800
1.702315926454,176.108590268886,3.528531255430
1.700134145790,176.334590268886,3.435854297601
1.697957950545,176.560590268886,3.344039040191
1.695787319299,176.786590268886,3.253108109659
1.693622230739,177.012590268886,3.163083739046
1.691462663663,177.238590268886,3.073987767975
1.689308596976,177.464590268886,2.985841642652
1.687160009691,177.690590268886,2.898666415863
1.685016880927,177.916590268886,2.812482746976
1.682879189909,178.142590268886,2.727310901944
1.680746915968,178.368590268886,2.643170753297
1.678620038539,178.594590268886,2.560081780151
1.676498537161,178.820590268886,2.478063068202
1.674382391476,179.046590268886,2.397133309728
1.672271581229,179.272590268886,2.317310803589
1.670166086268,179.498590268886,2.238613455227
1.668065886541,179.724590268886,2.161058776666
1.665970962096,179.950590268886,2.084663886512
1.663881293084,180.176590268886,2.009445509952
1.661796859752,180.402590268886,1.935419978755
1.659717642449,180.628590268886,1.862603231274
1.657643621620,180.854590268886,1.791010812441
1.655574777809,181.080590268886,1.720657873772
1.653511091656,181.306590268886,1.651559173363
1.651452543898,181.532590268886,1.583729075894
1.649399115368,181.758590268886,1.517181552625
1.647350786993,181.984590268886,1.451930181399
1.645307539796,182.210590268886,1.387988146640
1.643269354893,182.436590268886,1.325368239356
1.641236213494,182.662590268886,1.264082857133
1.639208096903,182.888590268886,1.204144004144
1.637184986515,183.114590268886,1.145563291139
1.635166863815,183.340590268886,1.088351935453
1.633153710383,183.566590268886,1.032520761002
1.631145507887,183.792590268886,0.978080198283
1.629142238086,184.018590268886,0.925040284376
1.627143882827,184.244590268886,0.873410662942
1.625150424049,184.470590268886,0.823200584226
1.623161843776,184.696590268886,0.774418905052
1.621178124123,184.922590268886,0.727074088827
1.619199247289,185.148590268886,0.681174205541
1.617225195563,185.374590268886,0.636726931765
1.615255951318,185.600590268886,0.593739550651
1.613291497015,185.826590268886,0.552218951934
1.611331815197,186.052590268886,0.512171631931
1.609376888494,186.278590268886,0.473603693540
1.607426699621,186.504590268886,0.436520846242
1.605481231374,186.730590268886,0.400928406099
1.603540466634,186.956590268886,0.366831295755
1.601604388364,187.182590268886,0.334234044437
1.599672979610,187.408590268886,0.303140787951
1.597746223500,187.634590268886,0.273555268688
1.595824103240,187.860590268886,0.245480835620
1.593906602121,188.086590268886,0.218920444300
1.591993703511,188.312590268886,0.193876656864
1.590085390861,188.538590268886,0.170351642029
1.588181647697,188.764590268886,0.148347175094
1.586282457626,188.990590268886,0.127864637941
1.584387804336,189.216590268886,0.108905019032
1.582497671587,189.442590268886,0.091468913412
1.580612043222,189.668590268886,0.075556522708
1.578730903158,189.894590268886,0.061167655130
1.576854235388,190.120590268886,0.048301725466
1.574982023983,190.346590268886,0.036957755091
1.573114253089,190.572590268886,0.027134371957
1.571250906925,190.798590268886,0.018829810602
1.569391969788,191.024590268886,0.012041912143
1.567537426047,191.250590268886,0.006768124281
1.565687260145,191.476590268886,0.003005501297
1.563841456600,191.702590268886,0.000750704055
1.562000000000,191.928590268886,0.000000000000
1.560162875009,192.154590268886,0.000749263161
1.558330066360,192.380590268886,0.002993974146
1.556501558859,192.606590268886,0.006729220147
1.554677337384,192.832590268886,0.011949694937
1.552857386882,193.058590268886,0.018649698871
1.551041692372,193.284590268886,0.026823138886
1.549230238942,193.510590268886,0.036463528502
1.547423011750,193.736590268886,0.047563987817
1.545619996023,193.962590268886,0.060117243516
1.543821177057,194.188590268886,0.074115628862
1.542026540217,194.414590268886,0.089551083703
1.540236070934,194.640590268886,0.106415154466
1.538449754708,194.866590268886,0.124698994161
1.536667577107,195.092590268886,0.144393362381
1.534889523764,195.318590268886,0.165488625299
1.533115580379,195.544590268886,0.187974755671
1.531345732718,195.770590268886,0.211841332835
1.529579966614,195.996590268886,0.237077542710
1.527818267964,196.222590268886,0.263672177799
1.526060622729,196.448590268886,0.291613637183
1.524307016937,196.674590268886,0.320889926529
1.522557436677,196.900590268886,0.351488658084
1.520811868105,197.126590268886,0.383397050676
1.519070297438,197.352590268886,0.416601929716
1.517332710958,197.578590268886,0.451089727197
1.515599095008,197.804590268886,0.486846481695
1.513869435994,198.030590268886,0.523857838364
1.512143720385,198.256590268886,0.562109048944
1.510421934709,198.482590268886,0.601584971756
1.508704065558,198.708590268886,0.642270071700
1.506990099584,198.934590268886,0.684148420262
1.505280023499,199.160590268886,0.727203695507
1.503573824076,199.386590268886,0.771419182084
1.501871488147,199.612590268886,0.816777771221
1.500173002605,199.838590268886,0.863261960730
1.498478354401,200.064590268886,0.910853855006
1.496787530545,200.290590268886,0.959535165023
1.495100518107,200.516590268886,1.009287208338
1.493417304213,200.742590268886,1.060090909091
1.491737876048,200.968590268886,1.111926798003
1.490062220855,201.194590268886,1.164775012376
1.488390325933,201.420590268886,1.218615296096
1.486722178641,201.646590268886,1.273426999630
1.485057766390,201.872590268886,1.329189080025
1.483397076650,202.098590268886,1.385880100912
1.481740096948,202.324590268886,1.443478232504
1.480086814865,202.550590268886,1.501961251594
1.478437218036,202.776590268886,1.561306541560
1.476791294155,203.002590268886,1.621491092358
1.475149030967,203.228590268886,1.682491500529
1.473510416274,203.454590268886,1.744283969195
1.471875437931,203.680590268886,1.806844308059
1.470244083846,203.906590268886,1.870147933407
1.468616341982,204.132590268886,1.934169868107
1.466992200355,204.358590268886,1.998884741606
1.465371647034,204.584590268886,2.064266789938
1.463754670139,204.810590268886,2.130289855714
1.462141257845,205.036590268886,2.196927388130
1.460531398377,205.262590268886,2.264152442963
1.458925080014,205.488590268886,2.331937682571
1.457322291084,205.714590268886,2.400255375895
1.455723019967,205.940590268886,2.469077398457
1.454127255095,206.166590268886,2.538375232363
1.452534984950,206.392590268886,2.608119966297
1.450946198064,206.618590268886,2.678282295529
1.449360883020,206.844590268886,2.748832521908
1.447779028450,207.070590268886,2.819740553867
1.446200623036,207.296590268886,2.890975906419
1.444625655508,207.522590268886,2.962507701160
1.443054114649,207.748590268886,3.034304666267
1.441485989286,207.974590268886,3.106335136500
1.439921268296,208.200590268886,3.178567053200
1.438359940607,208.426590268886,3.250967964291
1.436801995190,208.652590268886,3.323505024277
1.435247421069,208.878590268886,3.396144994246
1.433696207312,209.104590268886,3.468854241867
1.432148343034,209.330590268886,3.541598741390
1.430603817400,209.556590268886,3.614344073648
1.429062619618,209.782590268886,3.687055426055
1.427524738946,210.008590268886,3.759697592609
1.425990164685,210.234590268886,3.832234973887
1.424458886184,210.460590268886,3.904631577049
1.422930892837,210.686590268886,3.976851015838
1.421406174083,210.912590268886,4.048856510578
1.419884719407,211.138590268886,4.120610888174
1.418366518340,211.364590268886,4.192076582115
1.416851560455,211.590590268886,4.263215632469
1.415339835371,211.816590268886,4.333989685888
1.413831332752,212.042590268886,4.404359995607
1.412326042304,212.268590268886,4.474287421439
1.410823953780,212.494590268886,4.543732429783
1.409325056973,212.720590268886,4.612655093617
1.407829341721,212.946590268886,4.681015092502
1.406336797906,213.172590268886,4.748771712582
1.404847415450,213.398590268886,4.815883846580
1.403361184322,213.624590268886,4.882309993804
1.401878094529,213.850590268886,4.948008260141
1.400398136122,214.076590268886,5.012936358063
1.398921299196,214.302590268886,5.077051606621
1.397447573884,214.528590268886,5.140310931450
1.395976950363,214.754590268886,5.202670864766
1.394509418850,214.980590268886,5.264087545366
1.393044969605,215.206590268886,5.324516718631
1.391583592927,215.432590268886,5.383913736521
1.390125279156,215.658590268886,5.442233557582
1.388670018673,215.884590268886,5.499430746937
1.387217801899,216.110590268886,5.555459476295
1.385768619295,216.336590268886,5.610273523944
1.384322461362,216.562590268886,5.663826274757
1.382879318640,216.788590268886,5.716070720185
1.381439181709,217.014590268886,5.766959458263
1.380002041188,217.240590268886,5.816444693609
1.378567887735,217.466590268886,5.864478237421
1.377136712048,217.692590268886,5.911011507480
1.375708504860,217.918590268886,5.955995528147
1.374283256947,218.144590268886,5.999380930368
1.372860959119,218.370590268886,6.041117951669
1.371441602228,218.596590268886,6.081156436156
1.370025177161,218.822590268886,6.119445834522
1.368611674843,219.048590268886,6.155935204036
1.367201086238,219.274590268886,6.190573208554
1.365793402345,219.500590268886,6.223308118511
1.364388614201,219.726590268886,6.254087810923
1.362986712880,219.952590268886,6.282859769391
1.361587689493,220.178590268886,6.309571084096
1.360191535187,220.404590268886,6.334168451801
1.358798241144,220.630590268886,6.356598175851
1.357407798586,220.856590268886,6.376806166173
1.356020198765,221.082590268886,6.394737939276
1.354635432975,221.308590268886,6.410338618250
1.353253492541,221.534590268886,6.423552932768
1.351874368825,221.760590268886,6.434325219085
1.350498053224,221.986590268886,6.442599420036
1.349124537171,222.212590268886,6.448319085041
1.347753812131,222.438590268886,6.451427370098
1.346385869608,222.664590268886,6.451867037791
1.345020701136,222.890590268886,6.449580457283
1.343658298286,223.116590268886,6.444509604320
1.342298652662,223.342590268886,6.436596061229
1.340941755903,223.568590268886,6.425781016920
1.339587599681,223.794590268886,6.412005266885
1.338236175702,224.020590268886,6.395209213197
1.336887475705,224.246590268886,6.375332864510
1.335541491462,224.472590268886,6.352315836063
1.334198214779,224.698590268886,6.326097349673
1.332857637494,224.924590268886,6.296616233743
1.331519751478,225.150590268886,6.263810923254
1.330184548636,225.376590268886,6.227619459771
1.328852020904,225.602590268886,6.187979491442
1.327522160250,225.828590268886,6.144828272993
1.326194958675,226.054590268886,6.098102665736
1.324870408212,226.280590268886,6.047739137562
1.323548500925,226.506590268886,5.993673762946
1.322229228910,226.732590268886,5.935842222943
1.320912584295,226.958590268886,5.874179805192
1.319598559238,227.184590268886,5.808621403913
1.318287145931,227.410590268886,5.739101519905
1.316978336593,227.636590268886,5.665554260554
1.315672123477,227.862590268886,5.587913339825
1.314368498865,228.088590268886,5.506112078264
1.313067455071,228.314590268886,5.420083403001
1.311768984439,228.540590268886,5.329759847747
1.310473079341,228.766590268886,5.235073552795
1.309179732183,228.992590268886,5.135956265020
1.307888935397,229.218590268886,5.032339337877
1.306600681449,229.444590268886,4.924153731407
1.305314962830,229.670590268886,4.811330012229
1.304031772065,229.896590268886,4.693798353545
1.302751101705,230.122590268886,4.571488535140
1.301472944332,230.348590268886,4.444329943381
1.300197292557,230.574590268886,4.312251571214
1.298924139019,230.800590268886,4.175182018169
1.297653476386,231.026590268886,4.033049490360
1.296385297356,231.252590268886,3.885781800478
1.295119594653,231.478590268886,3.733306367800
1.293856361033,231.704590268886,3.575550218183
1.292595589277,231.930590268886,3.412439984067
1.291337272195,232.156590268886,3.243901904472
1.290081402626,232.382590268886,3.069861825001
1.288827973436,232.608590268886,2.890245197840
1.287576977518,232.834590268886,2.704977081756
1.286328407794,233.060590268886,2.513982142096
1.285082257212,233.286590268886,2.317184650792
1.283838518749,233.512590268886,2.114508486357
1.282597185408,233.738590268886,1.905877133883
1.281358250218,233.964590268886,1.691213685049
1.280121706238,234.190590268886,1.470440838112
1.278887546552,234.416590268886,1.243480897911
1.277655764269,234.642590268886,1.010255775870
1.276426352527,234.868590268886,0.770686989991
1.275199304489,235.094590268886,0.524695664860
1.273974613345,235.320590268886,0.272202531646
1.272752272312,235.546590268886,0.013127928096
1.271532274630,235.772590268886,-0.252608201456
1.270314613568,235.998590268886,-0.525086306099
1.269099282419,236.224590268886,-0.804387228337
1.267886274503,236.450590268886,-1.090592204092
1.266675583164,236.676590268886,-1.383782862707
1.265467201772,236.902590268886,-1.684041226940
1.264261123722,237.128590268886,-1.991449712969
1.263057342436,237.354590268886,-2.306091130389
1.261855851358,237.580590268886,-2.628048682213
1.260656643960,237.806590268886,-2.957405964874
1.259459713736,238.032590268886,-3.294246968222
1.258265054207,238.258590268886,-3.638656075523
1.257072658917,238.484590268886,-3.990718063465
1.255882521434,238.710590268886,-4.350518102151
1.254694635353,238.936590268886,-4.718141755104
1.253508994291,239.162590268886,-5.093674979264
1.252325591889,239.388590268886,-5.477204124989
1.251144421813,239.614590268886,-5.868815936056
1.249965477753,239.840590268886,-6.268597549661
1.248788753421,240.066590268886,-6.676636496415
1.247614242555,240.292590268886,-7.093020700350
1.246441938916,240.518590268886,-7.517838478914
1.245271836286,240.744590268886,-7.951178542976
1.244103928473,240.970590268886,-8.393129996820
1.242938209308,241.196590268886,-8.843782338149
1.241774672644,241.422590268886,-9.303225458086
1.240613312357,241.648590268886,-9.771549641170
1.239454122348,241.874590268886,-10.248845565358
1.238297096538,242.100590268886,-10.735204302027
1.237142228871,242.326590268886,-11.230717315971
1.235989513316,242.552590268886,-11.735476465401
1.234838943862,242.778590268886,-12.249574001947
1.233690514522,243.004590268886,-12.773102570660
1.232544219329,243.230590268886,-13.306155210003
1.231400052342,243.456590268886,-13.848825351863
1.230258007637,243.682590268886,-14.401206821542
1.229118079316,243.908590268886,-14.963393837761
1.227980261502,244.134590268886,-15.535481012658
1.226844548338,244.360590268886,-16.117563351791
1.225710933990,244.586590268886,-16.709736254135
1.224579412647,244.812590268886,-17.312095512083
1.223449978516,245.038590268886,-17.924737311446
1.222322625828,245.264590268886,-18.547758231455
1.221197348834,245.490590268886,-19.181255244755
1.220074141807,245.716590268886,-19.825325717414
1.218952999040,245.942590268886,-20.480067408916
1.217833914849,246.168590268886,-21.145578472161
1.216716883568,246.394590268886,-21.821957453470
1.215601899554,246.620590268886,-22.509303292582
1.214488957184,246.846590268886,-23.207715322652
1.213378050854,247.072590268886,-23.917293270255
1.212269174984,247.298590268886,-24.638137255384
1.211162324011,247.524590268886,-25.370347791449
1.210057492394,247.750590268886,-26.114025785279
1.208954674612,247.976590268886,-26.869272537121
1.207853865164,248.202590268886,-27.636189740639
1.206755058568,248.428590268886,-28.414879482918
1.205658249364,248.654590268886,-29.205444244457
1.204563432111,248.880590268886,-30.007986899177
1.203470601386,249.106590268886,-30.822610714415
1.202379751787,249.332590268886,-31.649419350926
1.201290877934,249.558590268886,-32.488516862884
1.200203974462,249.784590268886,-33.340007697881
1.199119036028,250.010590268886,-34.203996696927
1.198036057308,250.236590268886,-35.080589094450
1.196955032998,250.462590268886,-35.969890518296
1.195875957811,250.688590268886,-36.872006989729
1.194798826480,250.914590268886,-37.787044923431
1.193723633758,251.140590268886,-38.715111127504
1.192650374417,251.366590268886,-39.656312803465
1.191579043244,251.592590268886,-40.610757546251
1.190509635051,251.818590268886,-41.578553344218
1.189442144662,252.044590268886,-42.559808579138
1.188376566925,252.270590268886,-43.554632026202
1.187312896704,252.496590268886,-44.563132854019
1.186251128880,252.722590268886,-45.585420624617
1.185191258355,252.948590268886,-46.621605293441
1.184133280048,253.174590268886,-47.671797209354
1.183077188896,253.400590268886,-48.736107114639
1.182022979855,253.626590268886,-49.814646144994
1.180970647896,253.852590268886,-50.907525829539
1.179920188012,254.078590268886,-52.014858090808
1.178871595212,254.304590268886,-53.136755244755
1.177824864521,254.530590268886,-54.273330000754
1.176779990985,254.756590268886,-55.424695461594
1.175736969665,254.982590268886,-56.590965123484
1.174695795640,255.208590268886,-57.772252876051
1.173656464007,255.434590268886,-58.968673002338
1.172618969880,255.660590268886,-60.180340178809
1.171583308391,255.886590268886,-61.407369475344
1.170549474687,256.112590268886,-62.649876355244
1.169517463935,256.338590268886,-63.907976675224
1.168487271318,256.564590268886,-65.181786685420
1.167458892034,256.790590268886,-66.471423029385
1.166432321300,257.016590268886,-67.777002744092
1.165407554350,257.242590268886,-69.098643259928
1.164384586434,257.468590268886,-70.436462400703
1.163363412818,257.694590268886,-71.790578383642
1.162344028786,257.920590268886,-73.161109819389
1.161326429637,258.146590268886,-74.548175712005
1.160310610688,258.372590268886,-75.951895458972
1.159296567272,258.598590268886,-77.372388851187
1.158284294736,258.824590268886,-78.809776072966
1.157273788447,259.050590268886,-80.264177702045
1.156265043786,259.276590268886,-81.735714709576
1.155258056150,259.502590268886,-83.224508460129
1.154252820953,259.728590268886,-84.730680711694
1.153249333624,259.954590268886,-86.254353615677
1.152247589608,260.180590268886,-87.795649716903
1.151247584366,260.406590268886,-89.354691953616
1.150249313375,260.632590268886,-90.931603657477
1.149252772128,260.858590268886,-92.526508553566
1.148257956133,261.084590268886,-94.139530760379
1.147264860913,261.310590268886,-95.770794789833
1.146273482008,261.536590268886,-97.420425547261
1.145283814972,261.762590268886,-99.088548331416
1.144295855374,261.988590268886,-100.775288834466
1.143309598801,262.214590268886,-102.480773142001
1.142325040852,262.440590268886,-104.205127733027
1.141342177142,262.666590268886,-105.948479479967
1.140361003303,262.892590268886,-107.710955648665
1.139381514980,263.118590268886,-109.492683898380
1.138403707834,263.344590268886,-111.293792281793
1.137427577539,263.570590268886,-113.114409244998
1.136453119786,263.796590268886,-114.954663627512
1.135480330280,264.022590268886,-116.814684662267
1.134509204741,264.248590268886,-118.694601975615
1.133539738903,264.474590268886,-120.594545587324
1.132571928515,264.700590268886,-122.514645910583
1.131605769341,264.926590268886,-124.455033751996
1.130641257157,265.152590268886,-126.415840311587
1.129678387756,265.378590268886,-128.397197182799
1.128717156946,265.604590268886,-130.399236352490
1.127757560545,265.830590268886,-132.422090200939
1.126799594391,266.056590268886,-134.465891501841
1.125843254331,266.282590268886,-136.530773422312
1.124888536229,266.508590268886,-138.616869522883
1.123935435962,266.734590268886,-140.724313757504
1.122983949421,266.960590268886,-142.853240473545
1.122034072512,267.186590268886,-145.003784411791
1.121085801153,267.412590268886,-147.176080706448
1.120139131277,267.638590268886,-149.370264885139
1.119194058830,267.864590268886,-151.586472868904
1.118250579773,268.090590268886,-153.824840972202
1.117308690080,268.316590268886,-156.085505902911
1.116368385737,268.542590268886,-158.368604762327
1.115429662745,268.768590268886,-160.674275045162
1.114492517118,268.994590268886,-163.002654639547
1.113556944885,269.220590268886,-165.353881827034
1.112622942086,269.446590268886,-167.728095282589
1.111690504775,269.672590268886,-170.125434074599
1.110759629020,269.898590268886,-172.546037664867
1.109830310901,270.124590268886,-174.990045908616
1.108902546511,270.350590268886,-177.457599054486
1.107976331959,270.576590268886,-179.948837744534
1.107051663362,270.802590268886,-182.463903014239
1.106128536855,271.028590268886,-185.002936292494
1.105206948582,271.254590268886,-187.566079401611
1.104286894702,271.480590268886,-190.153474557323
1.103368371387,271.706590268886,-192.765264368777
1.102451374819,271.932590268886,-195.401591838542
1.101535901196,272.158590268886,-198.062600362601
1.100621946726,272.384590268886,-200.748433730358
1.099709507632,272.610590268886,-203.459236124635
1.098798580148,272.836590268886,-206.195152121672
1.097889160521,273.062590268886,-208.956326691125
1.096981245009,273.288590268886,-211.742905196070
1.096074829885,273.514590268886,-214.555033393002
1.095169911432,273.740590268886,-217.392857431833
1.094266485945,273.966590268886,-220.256523855891
1.093364549735,274.192590268886,-223.146179601927
1.092464099120,274.418590268886,-226.061972000105
1.091565130435,274.644590268886,-229.004048774011
1.090667640022,274.870590268886,-231.972558040647
1.089771624239,275.096590268886,-234.967648310434
1.088877079455,275.322590268886,-237.989468487210
1.087984002050,275.548590268886,-241.038167868232
1.087092388417,275.774590268886,-244.113896144175
1.086202234959,276.000590268886,-247.216803399133
1.085313538093,276.226590268886,-250.347040110617
1.084426294246,276.452590268886,-253.504757149555
1.083540499858,276.678590268886,-256.690105780296
1.082656151380,276.904590268886,-259.903237660605
1.081773245274,277.130590268886,-263.144304841666
1.080891778015,277.356590268886,-266.413459768081
1.080011746088,277.582590268886,-269.710855277869
1.079133145990,277.808590268886,-273.036644602468
1.078255974230,278.034590268886,-276.390981366735
1.077380227327,278.260590268886,-279.774019588944
1.076505901812,278.486590268886,-283.185913680788
1.075632994228,278.712590268886,-286.626818447376
1.074761501128,278.938590268886,-290.096889087238
1.073891419077,279.164590268886,-293.596281192320
1.073022744651,279.390590268886,-297.125150747987
1.072155474436,279.616590268886,-300.683654133022
1.071289605031,279.842590268886,-304.271948119626
1.070425133044,280.068590268886,-307.890189873418
1.069562055095,280.294590268886,-311.538536953436
1.068700367815,280.520590268886,-315.217147312135
1.067840067845,280.746590268886,-318.926179295389
1.066981151838,280.972590268886,-322.665791642489
1.066123616457,281.198590268886,-326.436143486144
1.065267458375,281.424590268886,-330.237394352484
1.064412674278,281.650590268886,-334.069704161053
1.063559260860,281.876590268886,-337.933233224816
1.062707214827,282.102590268886,-341.828142250156
1.061856532895,282.328590268886,-345.754592336872
1.061007211791,282.554590268886,-349.712744978182
1.060159248253,282.780590268886,-353.702762060725
1.059312639028,283.006590268886,-357.724805864554
1.058467380874,283.232590268886,-361.779039063141
1.057623470559,283.458590268886,-365.865624723379
1.056780904863,283.684590268886,-369.984726305575
1.055939680574,283.910590268886,-374.136507663458
1.055099794491,284.136590268886,-378.321133044172
1.054261243423,284.362590268886,-382.538767088281
1.053424024191,284.588590268886,-386.789574829766
1.052588133624,284.814590268886,-391.073721696026
1.051753568561,285.040590268886,-395.391373507881
1.050920325852,285.266590268886,-399.742696479564
1.050088402356,285.492590268886,-404.127857218731
1.049257794944,285.718590268886,-408.547022726454
1.048428500494,285.944590268886,-413.000360397222
1.047600515896,286.170590268886,-417.488038018944
1.046773838049,286.396590268886,-422.010223772946
1.045948463862,286.622590268886,-426.567086233973
1.045124390254,286.848590268886,-431.158794370188
1.044301614153,287.074590268886,-435.785517543170
1.043480132496,287.300590268886,-440.447425507920
1.042659942232,287.526590268886,-445.144688412854
1.041841040318,287.752590268886,-449.877476799807
1.041023423721,287.978590268886,-454.645961604032
1.040207089416,288.204590268886,-459.450314154201
1.039392034390,288.430590268886,-464.290706172403
1.038578255638,288.656590268886,-469.167309774146
1.037765750165,288.882590268886,-474.080297468355
1.036954514984,289.108590268886,-479.029842157375
1.036144547119,289.334590268886,-484.016117136966
1.035335843602,289.560590268886,-489.039296096310
1.034528401476,289.786590268886,-494.099553118003
1.033722217791,290.012590268886,-499.197062678064
1.032917289607,290.238590268886,-504.331999645925
1.032113613995,290.464590268886,-509.504539284439
1.031311188032,290.690590268886,-514.714857249877
1.030510008807,290.916590268886,-519.963129591928
1.029710073415,291.142590268886,-525.249532753698
1.028911378963,291.368590268886,-530.574243571713
1.028113922565,291.594590268886,-535.937439275915
1.027317701344,291.820590268886,-541.339297489666
1.026522712434,292.046590268886,-546.779996229744
1.025728952976,292.272590268886,-552.259713906348
1.024936420119,292.498590268886,-557.778629323093
1.024145111023,292.724590268886,-563.336921677012
1.023355022855,292.950590268886,-568.934770558557
1.022566152792,293.176590268886,-574.572355951598
1.021778498020,293.402590268886,-580.249858233423
1.020992055731,293.628590268886,-585.967458174738
1.020206823129,293.854590268886,-591.725336939667
1.019422797424,294.080590268886,-597.523676085753
1.018639975836,294.306590268886,-603.362657563956
1.017858355594,294.532590268886,-609.242463718655
1.017077933934,294.758590268886,-615.163277287646
1.016298708101,294.984590268886,-621.125281402144
1.015520675349,295.210590268886,-627.128659586781
1.014743832939,295.436590268886,-633.173595759610
1.013968178143,295.662590268886,-639.260274232098
1.013193708239,295.888590268886,-645.388879709134
1.012420420513,296.114590268886,-651.559597289023
1.011648312261,296.340590268886,-657.772612463487
1.010877380787,296.566590268886,-664.028111117669
1.010107623403,296.792590268886,-670.326279530129
1.009339037427,297.018590268886,-676.667304372844
1.008571620189,297.244590268886,-683.051372711210
1.007805369025,297.470590268886,-689.478672004041
1.007040281278,297.696590268886,-695.949390103569
1.006276354302,297.922590268886,-702.463715255445
1.005513585456,298.148590268886,-709.021836098736
1.004751972109,298.374590268886,-715.623941665930
1.003991511638,298.600590268886,-722.270221382932
1.003232201426,298.826590268886,-728.960865069063
1.002474038865,299.052590268886,-735.696062937064
1.001717021357,299.278590268886,-742.476005593096
1.000961146308,299.504590268886,-749.300884036734
1.000206411134,299.730590268886,-756.170889660974
0.999452813260,299.956590268886,-763.086214252228
0.998700350115,300.182590268886,-770.047049990330
0.997949019140,300.408590268886,-777.053589448528
0.997198817780,300.634590268886,-784.106025593489
0.996449743491,300.860590268886,-791.204551785300
0.995701793734,301.086590268886,-798.349361777465
0.994954965979,301.312590268886,-805.540649716904
0.994209257703,301.538590268886,-812.778610143960
0.993464666391,301.764590268886,-820.063437992389
0.992721189535,301.990590268886,-827.395328589368
0.991978824635,302.216590268886,-834.774477655492
0.991237569198,302.442590268886,-842.201081304773
0.990497420739,302.668590268886,-849.675336044642
0.989758376780,302.894590268886,-857.197438775947
0.989020434851,303.120590268886,-864.767586792956
0.988283592488,303.346590268886,-872.385977783353
0.987547847236,303.572590268886,-880.052809828242
0.986813196647,303.798590268886,-887.768281402144
0.986079638278,304.024590268886,-895.532591372998
0.985347169697,304.250590268886,-903.345939002162
0.984615788476,304.476590268886,-911.208523944412
0.983885492196,304.702590268886,-919.120546247941
0.983156278444,304.928590268886,-927.082206354360
0.982428144816,305.154590268886,-935.093705098701
0.981701088913,305.380590268886,-943.155243709410
0.980975108345,305.606590268886,-951.267023808355
0.980250200727,305.832590268886,-959.429247410819
0.979526363683,306.058590268886,-967.642116925505
0.978803594842,306.284590268886,-975.905835154533
0.978081891843,306.510590268886,-984.220605293443
0.977361252328,306.736590268886,-992.586630931190
0.976641673949,306.962590268886,-1001.004116050150
0.975923154365,307.188590268886,-1009.473265026115
0.975205691239,307.414590268886,-1017.994282628297
'''

EMBEDDED_DINT_PRESET_NAMES = [
    "default_dint_zero_flat",
    "default_dint_pasquazi2018_fig25_SiN_DW",
    "default_dint_pasquazi2018_MgF2",
    "default_dint_wilson2019_GaP",
    "default_dint_wilson2019_GaP_waveguide",  # backwards-compatible alias
    "default_dint_coen2012_MgF2",
]

EMBEDDED_DINT_PRESETS = {
    "default_dint_zero_flat": None,
    "default_dint_pasquazi2018_fig25_SiN_DW": DEFAULT_DINT_PASQUAZI_FIG25_SIN_DW_CSV,
    "default_dint_pasquazi2018_MgF2": DEFAULT_DINT_HERR_MGF2_CSV,
    "default_dint_wilson2019_GaP": EMBEDDED_DINT_WILSON2019_CSV,
    "default_dint_wilson2019_GaP_waveguide": EMBEDDED_DINT_WILSON2019_CSV,
    "default_dint_coen2012_MgF2": EMBEDDED_DINT_COEN2012_CSV,
}

# Pasquazi Fig. 25 reproduction guidance:
#   Stable cavity-soliton target: set detuning to Delta = 6.
#   Unstable modulation-instability target: set detuning to Delta = 3.
# The embedded dispersion is an approximate reconstruction of the published
# curve, not the authors' original numerical dispersion dataset.

# ============================================================
# PROGRAM START
# ============================================================


if __name__ == "__main__":
    run_gui()

'''
References

[1] S. A. Diddams, “The evolving optical frequency comb,” Science, 306, 1318–1324 (2004). https://doi.org/10.1126/science.1102529

[2] "Dispersion engineering and measurement of whispering gallery mode microresonator for Kerr frequency comb generation" Fujii, Shun and Tanabe, Takasumi.  Nanophotonics, vol. 9, no. 5, 2020, pp. 1087-1104. https://doi.org/10.1515/nanoph-2019-0497 

[3] D. J. Wilson, K. Schneider, S. Hönl, M. Anderson, Y. Baumgartner, L. Czornomaz, T. J. Kippenberg, P. Seidler, “Integrated gallium phosphide nonlinear photonics,” Nature Photonics, 14, 57–62 (2020). https://doi.org/10.1038/s41566-019-0535-9 

[4] T. Herr, V. Brasch, J. D. Jost, C. Y. Wang, N. M. Kondratiev, M. L. Gorodetsky, T. J. Kippenberg, “Temporal solitons in optical microresonators,” Nature Photonics, 8, 145–152 (2014). https://doi.org/10.1038/nphoton.2013.343 

[5] A. Pasquazi, M. Peccianti, L. Razzari, D. J. Moss, S. Coen, M. Erkintalo, Y. K. Chembo, T. Hansson, S. Wabnitz, P. Del’Haye, X. Xue, A. M. Weiner, R. Morandotti, “Micro-combs: A novel generation of optical sources,” Physics Reports, 729, 1–81 (2018). https://doi.org/10.1016/j.physrep.2017.08.004 

[6] G. Moille, Q. Li, D. A. Westly, A. A. Savchenkov, K. Srinivasan, “pyLLE: a Fast and User Friendly Lugiato–Lefever Equation Solver,” Journal of Research of NIST, 124, 124012 (2019). https://doi.org/10.6028/jres.124.012 

[7] O. Melchert, “pyGLLE: A Python-based solver for the generalized Lugiato–Lefever equation,” SoftwareX, 13, 100639 (2021). https://doi.org/10.1016/j.softx.2020.100639 

[8] L. A. Lugiato, R. Lefever, “Spatial dissipative structures in passive optical systems,” Physical Review Letters, 58, 2209–2211 (1987). https://doi.org/10.1103/PhysRevLett.58.2209 

[9] J. A. C. Weideman, B. M. Herbst, “Split-step methods for the solution of the nonlinear Schrödinger equation,” SIAM Journal on Numerical Analysis, 23, 485–507 (1986). https://doi.org/10.1137/0723033 

[10] G. P. Agrawal, “Nonlinear Fiber Optics,” Academic Press, 5th Edition (2013). https://doi.org/10.1016/C2011-0-00045-5 

'''
 


    
