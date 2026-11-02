"""Read stage0 CSV; render diagnostic curves only, without fitting or smoothing."""
import os
from pathlib import Path
import sys
import hashlib
import json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/v9_hopper_label_headroom_20261030'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path.home()/'.codex/skills/scipilot-figure-skill/scripts'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from profile_data import profile_data
from visual_qa import audit_layout
from setup_style import setup_style

def main():
    source=OUT/'hopper_T_curves.csv'
    df=pd.read_csv(source)
    assert len(df)==500 and not df.isna().any().any()
    if not (OUT/'CURVE_DATA_PROFILE.json').exists():
        profile=profile_data(str(source),group_cols=['seed'])
        with open(OUT/'CURVE_DATA_PROFILE.json','x',encoding='utf-8') as f:json.dump(profile,f,ensure_ascii=False,indent=2,default=str)
    setup_style(journal='general',lang='en',use_sciplots=False)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','figure.constrained_layout.use':False,'figure.autolayout':False})
    fig,axes=plt.subplots(2,5,figsize=(12.2,5.4),sharex=True,sharey=True)
    fig.set_layout_engine('none')
    fig.subplots_adjust(left=.07,right=.985,bottom=.16,top=.84,wspace=.12,hspace=.34)
    upper=float(np.ceil(df.score.max()/10)*10+5)
    for ax,s in zip(axes.flat,sorted(df.seed.unique())):
        d=df[df.seed==s].sort_values('step')
        color='#D55E00' if s==205 else '#009E73' if s==206 else '#0072B2'
        ls='--' if s==205 else '-.' if s==206 else '-'
        ax.axvspan(460,500,color='#888888',alpha=.13,zorder=0)
        ax.plot(d.step/1000,d.score,color=color,linestyle=ls,linewidth=1.2,marker='o',markersize=2)
        v=d[d.step>=460000].score.mean()
        ax.set_title('Seed %d | Last5 %.2f'%(s,v),color=color)
        ax.set_xlim(0,510);ax.set_ylim(0,upper)
        ax.set_xticks([0,250,500]);ax.grid(axis='y',alpha=.16)
    fig.suptitle('Hopper-morph target-only IQL: all R14 seeds',y=.97,fontsize=13,fontweight='bold')
    fig.text(.5,.89,'Unsmoothed evaluation means; 10 episodes per point. Gray band: fixed Last5 window.',ha='center',fontsize=9)
    fig.supxlabel('Training updates (thousands)',y=.06)
    fig.supylabel('OTDF normalized score',x=.012)
    fig.canvas.draw()
    issues=audit_layout(fig)
    config=dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        library_versions=dict(matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__),
        points=500,seeds=10,episodes_per_point=10,smoothing=False,error_bands=False,axes_shared=True,figure_inches=[12.2,5.4],
        output_dpi=300,layout_issues=issues,selection_reason='time-series small multiples expose each seed without mean aggregation masking failures',
        alternatives=['overlay all seeds, but overlap obscures205/206','focus only205/206, but omit context'],publication_target='internal stage0 diagnostic')
    if '--vector-after-qa' in sys.argv:
        fig.savefig(OUT/'figures/hopper_T_all_seeds_v3.svg',facecolor='white')
        print('Vector exported after preview visual QA')
    else:
        with open(OUT/'PLOT_CONFIG_v3.json','x',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2,default=str)
        preview=OUT/'figures/hopper_T_all_seeds_v3.png'
        fig.savefig(preview,dpi=300,facecolor='white')
        Image.open(preview).convert('L').save(OUT/'figures/hopper_T_all_seeds_v3_grayscale.png',dpi=(300,300))
        print(json.dumps(dict(rendered=str(preview),layout_issues=issues),default=str))

if __name__=='__main__':main()
