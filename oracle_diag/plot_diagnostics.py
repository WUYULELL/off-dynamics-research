"""Scatter figures for observed fixed-V diagnostics; no fabricated future panels."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,r'C:\Users\TXin\.codex\skills\scipilot-figure-skill\scripts')
from setup_style import setup_style
from profile_data import profile_data
from visual_qa import audit_layout,render_preview
from export_figure import export_figure
HERE=Path(__file__).resolve().parent
FOLDER=HERE/'results/verified_20261003/diagnostics'
FIGS=HERE/'results/verified_20261003/figures'


def main():
    FIGS.mkdir(exist_ok=True);setup_style(journal='general',lang='en',use_sciplots=False)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'svg.fonttype':'none'})
    qa={}
    for path in sorted(FOLDER.glob('*_bellman.npz')):
        data=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
        frame=pd.DataFrame({key:data[key] for key in ['oracle','knn','diff']})
        profile=profile_data(frame)
        (FIGS/(path.stem+'_profile.json')).write_text(json.dumps(profile,indent=2,default=str),encoding='utf-8')
        assert np.isfinite(frame.to_numpy()).all()
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        for ax,key,title,metric,color in zip(axes,['knn','diff'],['kNN reference','Displacement correction'],
                                           ['knn','diff_mean_state'],['#0072B2','#D55E00']):
            ax.scatter(data['oracle'],data[key],s=4,alpha=.22,color=color,edgecolors='none',rasterized=True)
            ax.set_xlabel('Oracle Bellman mismatch');ax.set_ylabel('Estimated mismatch')
            # Entire range is retained; rank panels below expose crowded ordering.
            ax.set_xlim(left=0);ax.set_ylim(bottom=0)
            rho=meta['metrics'][metric]['batch_spearman_mean']
            ax.set_title(title+'\nbatch Spearman = '+('undefined' if rho is None else '{:.3f}'.format(rho)))
        fig.suptitle(meta['env'].split('-')[0]+' | '+meta['V']+' | 4,999 transitions, 1 fixed V',fontsize=10)
        preview=FIGS/(path.stem+'.png');render_preview(fig,str(preview),dpi=180)
        issues=audit_layout(fig);qa[path.stem]=issues
        if any(level=='FAIL' for level,_ in issues):raise RuntimeError('Figure QA failure '+path.stem)
        export_figure(fig,str(FIGS/path.stem),formats=['png','svg','pdf'],dpi=300,size_inches=(7.2,3.2),tight=False)
        plt.close(fig)
    (FIGS/'layout_checks.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print('Figures checked:',len(qa))


if __name__=='__main__':main()
