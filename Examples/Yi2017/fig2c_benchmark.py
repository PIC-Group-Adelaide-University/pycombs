"""Yi et al. 2017 Fig. 2c qualitative benchmark using the unchanged v16r solver.
python fig2c_benchmark.py          # plot supplied numerical snapshots
python fig2c_benchmark.py --run    # regenerate four cases (several minutes)
"""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent
CASES=[('i',22.),('ii',29.),('iii',31.),('iv',46.)]

def solver_module():
    spec=importlib.util.spec_from_file_location('v16',next(ROOT.glob('pycombs_v16r*.py')))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def state(m,alpha,N=2048):
    st=m.LLEState()
    cfg=dict(m.STARTUP_PRESETS['default_startup_yi2017_amc_methods'],number_modes=N,detuning_start_DV=alpha)
    c=dict(cfg['amc'])
    # Analytical fitted hybrid offset; NOT an explicitly reported numerical offset.
    lower=-62.2e6; da=.5*17e3*72**2
    c['offset_B_hz']=lower+(42.4e6)**2/(da-lower)-50.9e6*72-da
    cfg['amc']=c
    m.apply_user_startup_config(st,cfg);st.reinitialize(new_n_modes=N)
    return st

def seed(m,st):
    # Generalized soliton width (paper Eq. 33) and Raman shift (Eq. 26).
    # This is a convenient initial condition, not an author-provided initial field.
    alpha=st.DV;d2=2*2*np.pi*17e3/st.kappa_avg
    coefficient=4*st.amc['raman_tau_fs']*1e-15*2*np.pi*st.fsr*d2/15
    width=brentq(lambda w:d2/2*(1/w**2+(coefficient/w**4)**2)-alpha,.003,.05)
    center=-coefficient/width**4
    theta=2*np.pi*np.arange(st.number_modes)/st.number_modes-np.pi
    roots=np.roots([1,-2*alpha,1+alpha**2,-st.P_norm])
    intensity=min(r.real for r in roots if abs(r.imag)<1e-7 and r.real>=0)
    background=np.sqrt(st.P_norm)/(1+1j*(alpha-intensity))
    amplitude=np.sqrt(d2)/width
    cosphase=2*amplitude/(np.pi*np.sqrt(st.P_norm))*np.cosh(np.pi*center*width/2)
    if cosphase>1:raise ValueError('Seed cannot be sustained at this pump/operating point')
    phase=np.arccos(cosphase)
    st.E_t_fast_norm=background+amplitude*np.exp(1j*phase+1j*center*theta)/np.cosh(np.clip(theta/width,-700,700))
    st.spectrum_E_t_fast_norm=m.fftshift(m.fft(st.E_t_fast_norm))
    # Auxiliary family starts empty: its peak must emerge through the coupling.
    st.B_t_fast_norm=np.zeros(st.number_modes,complex)
    st.spectrum_B_t_fast_norm=np.zeros(st.number_modes,complex)
    return dict(width_phi=float(width),spectral_center_mu=float(center),phase_rad=float(phase))

def metrics(mu,A,B):
    pa=abs(A)**2;pb=abs(B)**2
    # Excluding the pump mode is a diagnostic, not the paper's exact sech-fit centre.
    comb=pa.copy();comb[mu==0]=0
    return dict(centroid_excluding_pump_THz=float(sum(mu*comb)/sum(comb)*.022),
        B_peak_mode=int(mu[np.argmax(pb)]),
        B_peak_over_A_comb_power=float(max(pb)/sum(comb)),
        B_total_over_A_comb_power=float(sum(pb)/sum(comb)),
        A_comb_power_norm=float(sum(comb)),finite=bool(np.isfinite(A).all() and np.isfinite(B).all()))

def run(args,out):
    m=solver_module();start=time.time()
    for name,alpha in CASES:
        st=state(m,alpha,args.modes);seed_parameters=seed(m,st)
        st.tal_step*=args.rt_step;st.dt_phys_s*=args.rt_step;st.dt_phys_ns*=args.rt_step
        steps=int(round(args.duration_us*1e-6/st.dt_phys_s));history=[]
        remaining=steps
        while remaining:
            block=min(remaining,max(1,int(round(.5e-6/st.dt_phys_s))))
            st.step(block);remaining-=block
            A=st.spectrum_E_t_fast_norm/st.number_modes;B=st.spectrum_B_t_fast_norm/st.number_modes
            entry=metrics(st.mu,A,B);entry['time_us']=st.j*st.dt_phys_s*1e6;history.append(entry)
            print(name,entry,flush=True)
        np.savez_compressed(out/f'case_{name}.npz',mu=st.mu,A=A,B=B,fieldA=st.E_t_fast_norm,fieldB=st.B_t_fast_norm)
        (out/f'case_{name}_history.json').write_text(json.dumps(dict(alpha=alpha,P_norm=st.P_norm,
            amc=st.amc,seed=seed_parameters,history=history,elapsed_seconds=time.time()-start),indent=2))

def plot(out):
    report={'status':'Qualitative benchmark; exact reproduction unverified',
        'paper':'Yi et al., Nature Communications 8, 14869 (2017), doi:10.1038/ncomms14869',
        'assumptions':{'P_norm':100,'lower_hybrid_at_mu72_Hz':-62.2e6,
            'delta_FSR_Hz':50.9e6,'initial_field':'Raman-shifted sech + CW in A; B=0'},'cases':{}}
    fig=plt.figure(figsize=(14,8.6),facecolor='white')
    fig.text(.5,.965,'Yi2017 Fig. 2c: first spectral benchmark',ha='center',fontsize=18,weight='bold')
    ref=ROOT/'paper_fig2c_reference.png'
    if ref.exists():
        ax=fig.add_axes([.045,.59,.91,.29]);ax.imshow(plt.imread(ref));ax.axis('off')
        fig.text(.5,.90,'Published target - Yi et al. 2017, Fig. 2c (CC BY 4.0)',ha='center',fontsize=11)
    fig.text(.5,.555,'v16r simulations - qualitative comparison with reconstructed inputs',ha='center',fontsize=12,weight='bold')
    for i,(name,alpha) in enumerate(CASES):
        d=np.load(out/f'case_{name}.npz');mu=d['mu'];A=d['A'];B=d['B']
        pa=abs(A)**2;pb=abs(B)**2
        reference=max(pa[mu!=0]);floor=1e-10*reference
        ax=fig.add_axes([.075+i*.225,.24,.207,.27])
        sel=(mu>=-150)&(mu<=150)
        for p,color,label in [(pa,'#1725bd','Family A'),(pb,'#e12732','Family B')]:
            y=10*np.log10(np.maximum(p,floor)/reference)
            ax.vlines(mu[sel],-80,y[sel],color=color,lw=.7,label=label)
        ax.set(xlim=(-150,150),ylim=(-80,20),xticks=[-100,0,100],yticks=[-80,-60,-40,-20,0,20],xlabel='Mode number')
        ax.set_title(f'{name}: alpha = {alpha:g}'+ (' (transient)' if name=='iv' else ''),fontsize=11)
        if i==0:ax.set_ylabel('Relative modal power (dB)');ax.legend(loc='lower left',fontsize=8)
        else:ax.set_yticklabels([])
        ax.grid(axis='y',alpha=.2)
        info=metrics(mu,A,B);info['alpha']=alpha;info['snapshot_time_us']=2.
        history_file=out/f'case_{name}_history.json'
        if history_file.exists(): info['snapshot_time_us']=json.loads(history_file.read_text())['history'][-1]['time_us']
        info['settling_status']='Changing at last observed blocks' if name=='iv' else 'Nearly settled in bundled run; inspect history after reruns'
        report['cases'][name]=info
        fig.text(.178+i*.225,.15,f'B peak: mode {info["B_peak_mode"]}\nPeak / A comb: {100*info["B_peak_over_A_comb_power"]:.2f}%',ha='center',fontsize=9)
    fig.text(.5,.045,'Both families use one common power reference per panel; spacing is 20 dB.\n'
        'Case iii uses alpha=31, while the paper places its strong branch near alpha=29.\n'
        'Case iv is a transient snapshot. Pump power, bare B offset and seeds are reconstruction choices.\n'
        'The paper\'s hysteresis and exact operating points have not been reproduced.',
        ha='center',fontsize=10,color='#444444')
    fig.savefig(out/'fig2c_comparison.png',dpi=160)
    fig.savefig(out/'fig2c_simulated_spectra.svg')
    plt.close(fig)
    (out/'benchmark_summary.json').write_text(json.dumps(report,indent=2))
    for name,_ in CASES:
        d=np.load(out/f'case_{name}.npz');mu=d['mu'];a=abs(d['A'])**2;b=abs(d['B'])**2
        reference=max(a[mu!=0])
        np.savetxt(out/f'case_{name}_spectra.csv',np.column_stack((mu,a,b,10*np.log10(np.maximum(a,1e-30)/reference),10*np.log10(np.maximum(b,1e-30)/reference))),
            delimiter=',',header='mu,A_power_norm,B_power_norm,A_dB_relative,B_dB_relative',comments='')
    print(json.dumps(report['cases'],indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true');ap.add_argument('--duration-us',type=float,default=2.)
    ap.add_argument('--rt-step',type=float,default=.5);ap.add_argument('--modes',type=int,default=2048)
    ap.add_argument('--output',type=Path,default=ROOT/'results')
    args=ap.parse_args()
    if args.duration_us<=0 or args.rt_step<=0 or args.modes<2048 or args.modes%2:ap.error('Positive time and step; even mode count >=2048 required')
    args.output.mkdir(exist_ok=True,parents=True)
    if args.run:run(args,args.output)
    plot(args.output)
if __name__=='__main__':main()
