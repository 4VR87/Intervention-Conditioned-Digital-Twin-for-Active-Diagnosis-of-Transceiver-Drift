"""All quantitative artwork comes from retained executed results."""
from pathlib import Path
from io import BytesIO
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,Circle,Polygon
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.labelsize':11,
    'axes.titlesize':12,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':10,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
    'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'custom',
    'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
    'mathtext.bf':'Times New Roman:bold'})
C=[plt.cm.viridis(x) for x in [.12,.36,.60,.82]]
R=json.loads((ROOT/'data/results.json').read_text())

def save(fig,name):
    fig.set_facecolor('white')
    for ext in ['png','tiff','pdf','svg','eps']:
        b=BytesIO();kw={'dpi':600,'bbox_inches':'tight','facecolor':'white'}
        if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(b,format=ext,**kw);p=ROOT/f'figures/{name}.{ext}'
        temp=p.with_name(p.name+'.tmp');temp.write_bytes(b.getvalue());temp.replace(p)
    plt.close(fig)

def grid(ax):ax.grid(axis='y',color='.88',linewidth=.6);ax.set_axisbelow(True)

def main():
    # Physical optical path and causal intervention loop.
    f,a=plt.subplots(figsize=(7,4.6));a.set(xlim=(0,10),ylim=(0,6));a.axis('off')
    def box(x,y,w,h,text,color):
        a.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='black',linewidth=.8));a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(x,y,xx,yy,color='black',style='-'):
        a.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=12,color=color,linestyle=style,linewidth=1.1))
    box(.1,4.1,2.4,1.25,'Laser and IQ\ntransmitter','#cce7eb')
    # Fiber coil is a pictorial path.
    x=np.linspace(3,6.6,500);y=4.7+.26*np.sin((x-3)*np.pi*6)
    a.plot(x,y,color=C[1],linewidth=2);a.text(4.8,5.4,'Fiber link and optical filter',ha='center')
    box(7.4,4.1,2.4,1.25,'Coherent receiver\nand telemetry','#cce7eb');arrow(2.5,4.7,3,4.7);arrow(6.6,4.7,7.4,4.7)
    a.text(1.7,3.25,'Bias and\nphase drift',ha='left',va='center',fontsize=10);a.text(4.8,3.6,'Loss and filter offset',ha='center',fontsize=10)
    box(.5,1,2.5,1.25,'One selected probe\nbias / detuning\nor power','#f6edb1')
    box(3.7,1,2.6,1.25,'Commissioned twin\nphysical residuals','#dcebcf')
    box(7,1,2.8,1.25,'Learned diagnosis\nand prediction set','#dcebcf')
    arrow(8.6,4.1,5,2.25);arrow(6.3,1.6,7,1.6);arrow(1.3,2.25,1.3,4.1)
    a.plot([8.4,8.4,1.75],[1,.6,.6],color='black',linewidth=1);arrow(1.75,.6,1.75,1)
    a.text(5,.68,'Learned choice from passive telemetry',ha='center',fontsize=10)
    save(f,'Figure1_architecture')

    # Mechanistic interventions on held fixed illustrative parameters, no optimization.
    f,axs=plt.subplots(2,1,figsize=(6.8,6.3),layout='constrained')
    b=np.linspace(-.1,.1,101);det=np.linspace(-1,1,101)
    for label,db,df,col in [('Transceiver drift',.13,0,C[0]),('Link filter offset',0,2.,C[2])]:
        eps=.012+.001+.45*np.sin(db+b)**2+.001+.012*(df/5)**2
        axs[0].plot(b,-10*np.log10(eps),label=label,color=col,linewidth=2)
        eps2=.012+.001+.45*np.sin(db)**2+.001+.012*((df-det)/5)**2
        axs[1].plot(det,-10*np.log10(eps2),label=label,color=col,linewidth=2)
    axs[0].set(xlabel='Applied IQ bias offset (rad)',ylabel='Effective SNR (dB)',title='A  Bias intervention in the assumed model')
    axs[1].set(xlabel='Applied detuning offset (GHz)',ylabel='Effective SNR (dB)',title='B  Detuning intervention in the assumed model')
    for ax in axs:grid(ax)
    f.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    save(f,'Figure2_interventions')

    d=np.load(ROOT/'data/test.npz');f,axs=plt.subplots(2,1,figsize=(6.8,7.0),layout='constrained')
    y=d['y'];z=d['residual']
    for k,label in enumerate(R['classes']):
        idx=np.flatnonzero(y==k)[:200]
        axs[0].scatter(z[idx,0,0],z[idx,0,1],s=8,color=C[k],label=label)
        axs[1].scatter(z[idx,1,0]-z[idx,0,0],z[idx,0,0],s=8,color=C[k],label=label)
    axs[0].set(xlabel='Passive SNR residual (dB)',ylabel='Passive power residual (dB)',title='A  First 200 test episodes per class')
    axs[1].set(xlabel='Bias-probe SNR residual change (dB)',ylabel='Passive SNR residual (dB)',title='B  Same episodes after the fixed positive bias probe')
    f.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    for ax in axs:grid(ax)
    save(f,'Figure3_telemetry')

    policies=list(R['policies']);short=['Logistic\npassive','ML\npassive','Random\nprobe','Fixed\nprobe','Active\nprobe','All six\nprobes']
    vals=np.array([R['policies'][p]['nominal']['accuracy']*100 for p in policies]);ci=np.array([R['policies'][p]['nominal']['accuracy_ci'] for p in policies])*100
    f,axs=plt.subplots(2,1,figsize=(6.8,6.2),layout='constrained')
    x=np.arange(6);axs[0].bar(x,vals,color=[C[1]]*4+[C[2],C[0]],width=.6)
    axs[0].errorbar(x,vals,yerr=[vals-ci[:,0],ci[:,1]-vals],fmt='none',color='black',capsize=3)
    for i,v in enumerate(vals):axs[0].text(i,ci[i,1]+.25,f'{v:.2f}',ha='center',fontsize=9)
    axs[0].set(ylim=(85,100),ylabel='Test accuracy (%)',title='A  Nominal synthetic test population of 2000 episodes',xticks=x,xticklabels=short)
    axs[1].bar(x,[0,0,1,1,1,6],color=C[1],width=.6);axs[1].set(ylabel='Additional measurement views',xticks=x,xticklabels=short,ylim=(0,7),title='B  Diagnostic acquisition budget')
    for ax in axs:grid(ax)
    save(f,'Figure4_accuracy')

    f,axs=plt.subplots(1,2,figsize=(7,3.6),layout='constrained')
    for ax,p,title in zip(axs,['Passive ML','Active ML'],['A  Passive ML','B  Active ML']):
        cm=np.array(R['policies'][p]['nominal']['confusion']);rate=cm/cm.sum(axis=1)[:,None]
        im=ax.imshow(rate,vmin=0,vmax=1,cmap='viridis')
        for i in range(4):
            for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,color='black' if rate[i,j]>.6 else 'white')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=['H','T','F','TF'],yticklabels=['H','T','F','TF'],xlabel='Predicted class',ylabel='True class',title=title)
    f.colorbar(im,ax=axs,shrink=.8,label='Fraction within each true class')
    save(f,'Figure5_confusion')

    df=pd.read_csv(ROOT/'data/coverage_sensitivity.csv');f,axs=plt.subplots(2,1,figsize=(6.8,6.1),layout='constrained')
    for j,p in enumerate(['Passive ML','Fixed probe','Active ML','All probes']):
        q=df[df.policy==p];axs[0].plot(100*(1-q.alpha),100*q.coverage,'o-',color=C[j],label=p)
        axs[1].plot(100*q.singleton_rate,100*q.singleton_error,'o-',color=C[j],label=p)
    axs[0].plot([80,99],[80,99],'--',color='.4',label='Target coverage')
    axs[0].set(xlabel='Target marginal coverage (%)',ylabel='Observed set coverage (%)',title='A  Frozen-policy split conformal sets');axs[0].legend(frameon=False,ncol=2,fontsize=8,loc='lower right')
    axs[1].set(xlabel='Singleton diagnoses (% of all episodes)',ylabel='Error among singleton diagnoses (%)',title='B  Selective diagnosis tradeoff')
    for ax in axs:grid(ax)
    save(f,'Figure6_uncertainty')

    f,axs=plt.subplots(2,1,figsize=(6.8,6.1),layout='constrained')
    for fn,col,xlabel,ax,title in [('training_sensitivity.csv','n_training','Classifier training episodes',axs[0],'A  Nested training-size sensitivity'),('noise_sensitivity.csv','noise_multiplier','Monitoring noise multiplier',axs[1],'B  Same paired latent episodes and noise directions')]:
        df=pd.read_csv(ROOT/'data'/fn)
        for j,p in enumerate(['Passive ML','Fixed probe','Active ML','All probes']):
            q=df[df.policy==p];ax.plot(q[col],100*q.accuracy,'o-',color=C[j],label=p)
        ax.set(xlabel=xlabel,ylabel='Nominal accuracy (%)',title=title);grid(ax)
    f.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    save(f,'Figure7_sensitivity')

    f,axs=plt.subplots(2,1,figsize=(6.8,6.1),layout='constrained');names=['Passive ML','Fixed probe','Active ML','All probes'];x=np.arange(4)
    for ax,key,title in [(axs[0],'accuracy','A  Forced diagnosis accuracy'),(axs[1],'coverage','B  Set coverage using nominal thresholds')]:
        ax.bar(x-.17,[100*R['policies'][p]['nominal'][key] for p in names],width=.33,color=C[1],label='Nominal')
        ax.bar(x+.17,[100*R['policies'][p]['stress'][key] for p in names],width=.33,color=C[3],label='Shifted nuisance / model mismatch')
        ax.set(xticks=x,xticklabels=names,ylabel='Episodes (%)',ylim=(0,105),title=title);grid(ax)
    f.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=9);axs[1].axhline(90,linestyle='--',color='.35',linewidth=1)
    save(f,'Figure8_shift')

    f,axs=plt.subplots(2,1,figsize=(6.8,6.3),layout='constrained')
    pp=np.linspace(-1,1,101);loss=2.;received=10**((pp-loss)/10)
    axs[0].plot(pp,received,label='2 dB transmitter attenuation',color=C[0],linewidth=3)
    axs[0].plot(pp,received,'--',label='2 dB link attenuation',color=C[3],linewidth=2)
    axs[0].set(xlabel='Applied launch-power change (dB)',ylabel='Normalized received power',title='A  Restricted gain-only nonidentifiable control');axs[0].legend(frameon=False,loc='upper left')
    q=pd.read_csv(ROOT/'data/symbol_validation.csv');axs[1].semilogy(q.snr_db,q.awgn_approximation,'-',color=C[0],label='Gray 16-QAM AWGN approximation')
    axs[1].semilogy(q.snr_db,q.measured_ber,'o',color=C[2],label='Counted errors in additive-noise symbols')
    axs[1].set(xlabel='Symbol SNR (dB)',ylabel='Bit-error rate',title='B  Equivalent-noise symbol check with 2 million bits per point');axs[1].legend(frameon=False,loc='lower left',fontsize=8)
    for ax in axs:grid(ax)
    save(f,'Figure9_limits_and_symbols')
    print('Generated 9 figure sets: Times New Roman, 600 dpi PNG/TIFF and vector EPS/PDF/SVG.')

if __name__=='__main__':main()
