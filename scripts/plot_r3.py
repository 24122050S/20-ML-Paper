"""Hình R3: thời gian học và RSS đo được,3 timing repeats trên1 seed."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    value=json.loads((ROOT/'reports/r3_cifar100_k5_summary.json').read_bytes())
    assert value['status']=='complete' and value['n_cases']==9 and value['n_seeds']==1
    rows=value['rows'];x=list(range(3));labels=[str(row['batch_size']) for row in rows]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.4),layout='constrained')
    colors=['#2274a5','#3487b8','#65a7c7']
    for axis,field,scale,title,ylabel in ((axes[0],'learning_seconds',1,'Analytic learning time','Seconds (base +5 incremental phases)'),
        (axes[1],'peak_learning_rss_bytes',1024**2,'Observed learning peak RSS','MiB (whole CPU worker)')):
        stats=[row['metrics'][field] for row in rows]
        axis.bar(x,[v['mean']/scale for v in stats],yerr=[v['sample_sd']/scale for v in stats],color=colors,capsize=5,width=0.6)
        for i,v in enumerate(stats):
            axis.scatter([i-0.09,i,i+0.09],[v['values_by_repeat'][str(r)]/scale for r in (1,2,3)],s=20,color='#172d3d',zorder=4)
        axis.set(title=title,ylabel=ylabel,xlabel='Analytic batch size',xticks=x,xticklabels=labels)
        axis.grid(axis='y',color='#e0e5e8',linewidth=0.6);axis.set_axisbelow(True);axis.margins(y=0.18)
    fig.suptitle('R3 · CIFAR-100 / K=5 · seed42 · d1024\n3 timing repeats; error bars are sample SD; dots are individual runs',fontsize=12)
    path=ROOT/'reports/figures/r3_cifar100_k5_batch_analysis.png';path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180,facecolor='white');plt.close(fig);print('Saved',path)


if __name__=='__main__':main()
