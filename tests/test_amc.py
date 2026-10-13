"""Run: MPLBACKEND=Agg python test_v16e.py. No GUI is required."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import expm
root=Path(__file__).resolve().parents[1]/'Examples/Yi2017'
s=importlib.util.spec_from_file_location('pycombs',root/'pycombs_v16r_Yi2017_Fig2c_i.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_linear_exact():
    st=m.LLEState(); st.set_amc_config(raman_tau_fs=0)
    rng=np.random.default_rng(10); x=rng.normal(size=(2048,2))+1j*rng.normal(size=(2048,2))
    f=rng.normal(size=2048)+1j*rng.normal(size=2048); h=.001
    y=st._amc_linear(x,f,h); da,db,_,_=st.amc_dispersion()
    for i in [0,1024,1096,2047]:
        M=np.array([[-1-1j*(st.DV+2*da[i]/st.kappa_avg),1j*4*np.pi*st.amc['G_hz']/st.kappa_avg],
                    [1j*4*np.pi*st.amc['G_hz']/st.kappa_avg,-2*np.pi*st.amc['kappa_B_hz']/st.kappa_avg-1j*(st.DV+2*db[i]/st.kappa_avg)]])
        aug=np.zeros((3,3),complex); aug[:2,:2]=M;aug[0,2]=f[i]
        exact=(expm(aug*h)@np.r_[x[i],1])[:2]
        np.testing.assert_allclose(y[i],exact,atol=2e-12,rtol=2e-12)

def test_legacy_limit():
    a=m.LLEState();b=m.LLEState()
    for st in (a,b):
        m.apply_named_startup_preset(st,'default_startup_pasquazi2018_fig19_exact_quadratic')
        st.reinitialize(new_n_modes=512)
        st.noise_switch=False
        st.E_t_fast_norm=.3+.1*np.cos(2*np.pi*np.arange(512)/512)+0j
        st.spectrum_E_t_fast_norm=m.fftshift(m.fft(st.E_t_fast_norm))
    b.set_amc_config(enabled=True,G_hz=0,raman_tau_fs=0)
    a.step(50);b.step(50)
    np.testing.assert_allclose(a.spectrum_E_t_fast_norm,b.spectrum_E_t_fast_norm,rtol=2e-12,atol=2e-12)
    assert np.max(abs(b.spectrum_B_t_fast_norm))==0
    assert not a.amc['enabled']

def test_coupling_energy():
    st=m.LLEState();st.amc.update(kappa_B_hz=0,raman_tau_fs=0)
    st.kappa_all[:]=0; st._amc_cache=None
    rng=np.random.default_rng(1);x=rng.normal(size=(2048,2))+1j*rng.normal(size=(2048,2))
    y=st._amc_linear(x,np.zeros(2048),.005)
    np.testing.assert_allclose(np.sum(abs(y)**2,axis=1),np.sum(abs(x)**2,axis=1),rtol=2e-12,atol=2e-12)

def test_reset_rescale():
    st=m.LLEState();st.step(10);B=st.spectrum_B_t_fast_norm.copy()*st.E_amp_2_norm_factor
    st.Q*=1.1;st.recalculate_derived_preserve_state()
    np.testing.assert_allclose(st.spectrum_B_t_fast_norm*st.E_amp_2_norm_factor,B,rtol=1e-13)
    st.reinitialize(new_n_modes=2048);assert np.max(abs(st.spectrum_B_t_fast_norm))==0
    try:st.set_amc_config(kappa_B_hz=-1)
    except ValueError:pass
    else:raise AssertionError('Invalid linewidth accepted')
    for preset in ('default_startup_yi2017_amc_methods','default_startup_yi2017_amc_crystal_seed'):
        m.apply_named_startup_preset(st,preset);st.reinitialize(new_n_modes=2048)
        st.step(5);assert np.isfinite(st.E_t_fast_norm).all()


def test_exceptional_point():
    st=m.LLEState();st.dint[:]=0
    st.set_amc_config(kappa_B_hz=2.12e6*3,G_hz=2.12e6/2,
        offset_B_hz=0,delta_fsr_hz=0,D2_B_hz=0)
    x=np.ones((2048,2),complex);F=np.ones(2048);h=.1
    y=st._amc_linear(x,F,h)
    matrix=np.array([[-1-1j*st.DV,1j],[1j,-3-1j*st.DV]])
    augmented=np.zeros((3,3),complex);augmented[:2,:2]=matrix;augmented[0,2]=1
    np.testing.assert_allclose(y[0],(expm(augmented*h)@np.ones(3))[:2],rtol=1e-12,atol=1e-12)

if __name__=='__main__':
    for test in (test_linear_exact,test_legacy_limit,test_coupling_energy,test_reset_rescale,test_exceptional_point):
        test();print('PASS',test.__name__)
