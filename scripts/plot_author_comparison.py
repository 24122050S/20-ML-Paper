"""Vẽ kết quả source tác giả; một seed, không tạo error bar như nhiều seed."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    """Đọc JSON gốc và diagnostic để xuất PNG hai panel, không sửa số liệu."""
    report=ROOT/'reports';suite=json.loads((report/'author_comparison_results.json').read_bytes())
    diagnostic=json.loads((report/'author_comparison_base_diagnostic.json').read_bytes())
    fig,(left,right)=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for case,color,label in zip(suite['cases'],['#1768ac','#c34b21'],['Author: controlled N(0,1) FE','Author main: native FE']):
        rows=case['results'];classes=[r['n_classes'] for r in rows]
        left.plot(classes,[r['accuracy_pct'] for r in rows],marker='o',color=color,label=label,linewidth=2)
        right.semilogy(classes,[r['group_comparison']['relative_w_error'] for r in rows],marker='o',color=color,label=label,linewidth=2)
    left.set(title='All-seen accuracy: one seed, d=1024',xlabel='Classes seen',ylabel='Accuracy (%)',xticks=range(50,101,10))
    right.semilogy([r['n_classes'] for r in diagnostic['results']],[r['comparison']['relative_w_error'] for r in diagnostic['results']],
        marker='s',linestyle='--',color='#348a43',label='Posthoc: matched base state',linewidth=2)
    right.set(title='Weight error vs group reference',xlabel='Classes seen',ylabel='Relative Frobenius W error',xticks=range(50,101,10))
    for axis in (left,right):
        axis.grid(alpha=.22);axis.spines[['top','right']].set_visible(False)
        axis.legend(fontsize=8.5,loc='best')
    right.legend(fontsize=8.5,loc='center left',bbox_to_anchor=(.02,.32))
    fig.suptitle('Pinned author ACIL source vs reimplementation — CIFAR-100/K5',fontsize=13,weight='bold')
    fig.supxlabel('Same R1 seed42 checkpoint; no fresh BP. Native FE differs. Original C1 score allclose fails at 2 phases; predictions agree.',fontsize=9)
    destination=report/'figures/author_acil_comparison.png';fig.savefig(destination,dpi=170);plt.close(fig)
    print(destination)


if __name__=='__main__':main()
