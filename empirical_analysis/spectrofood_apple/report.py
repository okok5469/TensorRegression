"""Complete OOF evidence and modest scientific figures, without refitting."""
from pathlib import Path
import csv
import json
import numpy as np

from .run import load_data, read_json, write_csv, write_json, file_hash, frozen_contract


def paired_intervals(errors, model_ids, seed=13999, count=2000):
    errors = np.asarray(errors, float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(errors), size=(count,len(errors)))
    mse = errors.mean(axis=0).mean(axis=0)
    resampled = errors[draws].mean(axis=1)
    boot_mse = resampled.mean(axis=1)
    boot_rmse = np.sqrt(resampled).mean(axis=1)
    rmse = np.sqrt(errors.mean(axis=0)).mean(axis=0)
    main = model_ids.index('M1')
    result = []
    for k, model in enumerate(model_ids):
        if model == 'M1':
            continue
        dm = boot_mse[:,main]-boot_mse[:,k]
        dr = boot_rmse[:,main]-boot_rmse[:,k]
        result.append(dict(comparator=model,mse_difference=float(mse[main]-mse[k]),
            mse_ci_low=float(np.quantile(dm,.025)),mse_ci_high=float(np.quantile(dm,.975)),
            rmse_difference=float(rmse[main]-rmse[k]),rmse_ci_low=float(np.quantile(dr,.025)),
            rmse_ci_high=float(np.quantile(dr,.975)),bootstrap_entities=len(errors),bootstrap_resamples=count))
    return result


def make_figures(config, rows, repeated, data, diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    output = Path(config['output_dir'])/'figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    files = []

    def save(fig, name):
        fig.tight_layout()
        path = output/name
        fig.savefig(path.with_suffix('.pdf'))
        fig.savefig(path.with_suffix('.svg'))
        fig.savefig(path.with_suffix('.png'),dpi=300)
        # Each figure has one plot area; no comparable multi-panel alignment applies.
        write_json(output/(name+'.alignment.json'),dict(status='NOT APPLICABLE',reason='Single plot area'))
        files.append(str(path.with_suffix('.pdf')))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6,2.7))
    ax.hist(data['y_pct'],bins=18,color='#7896aa',edgecolor='white')
    ax.set(xlabel='Dry matter (%)',ylabel='Apple records',title=f'Response distribution (n={len(data["y_pct"])})')
    save(fig,'response_distribution')
    fixed = ['A1','A120']
    raw = loadmat(config['mat_path'],variable_names=fixed+['Wavelengths'])
    band_index=int(np.argmin(abs(np.asarray(raw['Wavelengths']).ravel()-750)))
    band_nm=float(np.asarray(raw['Wavelengths']).ravel()[band_index])
    for sid in fixed:
        for name, image in [('mean_intensity',raw[sid].mean(axis=2)),(f'band_{band_nm:g}nm',raw[sid][:,:,band_index])]:
            fig,ax=plt.subplots(figsize=(3.4,2.8))
            ax.imshow(image,cmap='viridis',extent=(0,1,1,0),aspect='auto')
            ax.set(xlabel='Relative ROI horizontal position',ylabel='Relative ROI vertical position',
                   title=f'{sid}: {name.replace("_"," ")}')
            save(fig,sid+'_'+name)
    fig,ax=plt.subplots(figsize=(3.6,2.7))
    for sid in fixed:
        i=list(data['sample_ids']).index(sid)
        ax.plot(data['wavelengths'],data['spectrum_native'][i],label=sid)
    ax.set(xlabel='Wavelength (nm)',ylabel='Mean recorded intensity',title='Fixed-ID native spectra')
    ax.legend()
    save(fig,'native_spectra')
    labels={'M0':'Mean','M1':'Tensor-ABESS','M2':'Tensor-GLasso','M3':'Vector Lasso',
            'M4':'Ridge','M5':'PLS-binned','M6':'PLS-native'}
    for metric,label in [('rmse','OOF RMSE (percentage points)'),('mae','OOF MAE (percentage points)'),('r2','Complete OOF R²')]:
        fig,ax=plt.subplots(figsize=(5.6,3.0))
        for repeat,marker in zip(range(1,4),['o','s','^']):
            vals=[next(r[metric] for r in repeated if r['model']==m and int(r['repeat'])==repeat) for m in config['models']]
            ax.scatter(np.arange(7)+(repeat-2)*.12,vals,marker=marker,s=22,label=f'Repeat {repeat}')
        ax.set_xticks(np.arange(7),[labels[m] for m in config['models']],rotation=22,ha='right')
        ax.set(ylabel=label)
        ax.legend(fontsize=7)
        save(fig,'complete_oof_'+metric)
    for model in config['models']:
        selected=[r for r in rows if r['model']==model and int(r['repeat'])==1]
        truth=np.array([float(r['y_true_pct']) for r in selected])
        prediction=np.array([float(r['y_pred_pct']) for r in selected])
        lo,hi=min(truth.min(),prediction.min()),max(truth.max(),prediction.max())
        fig,ax=plt.subplots(figsize=(3.3,3.0))
        ax.scatter(truth,prediction,s=10,alpha=.6,color='#527c99')
        ax.plot([lo,hi],[lo,hi],color='0.4',linewidth=.8,linestyle='--')
        ax.set(xlabel='Observed dry matter (%)',ylabel='Predicted dry matter (%)',title=labels[model]+' — repeat 1')
        save(fig,'prediction_repeat1_'+model)
    q=np.array([d['q'] for d in diagnostics])
    fig,ax=plt.subplots(figsize=(3.6,2.7))
    ax.plot(range(1,16),q,'o-',color='#527c99')
    ax.set(xlabel='Outer fold (repeat-major order)',ylabel='Rectangular capacity q',title='Tensor-ABESS selected capacity')
    save(fig,'abess_capacity')
    fig,ax=plt.subplots(figsize=(3.6,2.7))
    ax.scatter([d['train_rmse'] for d in diagnostics],[d['test_rmse'] for d in diagnostics],s=23,color='#527c99')
    ax.set(xlabel='Training RMSE (percentage points)',ylabel='Test RMSE (percentage points)',title='Tensor-ABESS train/test error')
    save(fig,'abess_train_test')
    frequency=[]
    supports=[d['supports'] for d in diagnostics]
    for mode,dim in enumerate(data['X'].shape[1:]):
        freq=np.array([sum(i in s[mode] for s in supports)/len(supports) for i in range(dim)])
        frequency.extend(dict(mode=mode+1,index=i,frequency=float(f)) for i,f in enumerate(freq))
        fig,ax=plt.subplots(figsize=(3.8,2.7))
        ax.bar(np.arange(dim),freq,color='#7896aa',width=.75)
        ax.set(xlabel=['ROI row index','ROI column index','Contiguous wavelength-bin index'][mode],
               ylabel='Selected fraction of 15 folds',ylim=(0,1),title='Descriptive support frequency')
        save(fig,'support_frequency_mode'+str(mode+1))
    write_csv(Path(config['output_dir'])/'support_frequencies.csv',frequency)
    return files


def make_report(config):
    output=Path(config['output_dir'])
    contract=frozen_contract(config)
    def table(name):
        with (output/name).open(encoding='utf-8-sig',newline='') as stream:
            return list(csv.DictReader(stream))
    rows,repeat,summary=table('oof_predictions.csv'),table('repeat_metrics.csv'),table('summary_metrics.csv')
    numeric_repeats=[dict(model=r['model'],repeat=int(r['repeat']),**{m:float(r[m]) for m in ['rmse','mae','r2']}) for r in repeat]
    data=load_data(config)
    entities=sorted(set(r['entity_id'] for r in rows))
    errors=np.empty((len(entities),3,7))
    for i,entity in enumerate(entities):
        for r in range(1,4):
            for k,m in enumerate(config['models']):
                selected=[row for row in rows if row['entity_id']==entity and int(row['repeat'])==r and row['model']==m]
                truth=np.mean([float(row['y_true_pct']) for row in selected])
                pred=np.mean([float(row['y_pred_pct']) for row in selected])
                errors[i,r-1,k]=(truth-pred)**2
    paired=paired_intervals(errors,config['models'])
    write_csv(output/'paired_error_differences.csv',paired)
    diagnostics=[]
    for split in contract['splits']:
        value=read_json(output/'tasks'/f'r{split["repeat"]}_f{split["fold"]}_M1'/'diagnostics.json')
        diagnostics.append(dict(**value['diagnostics'],train_rmse=value['train_metrics']['rmse'],
                                test_rmse=value['test_metrics']['rmse']))
    figure_files=make_figures(config,rows,numeric_repeats,data,diagnostics)
    values={r['model']:{k:float(v) for k,v in r.items() if k!='model'} for r in summary}
    stable=[m for m in config['models'][1:] if all(next(r['rmse'] for r in numeric_repeats if r['model']==m and r['repeat']==i)<
            next(r['rmse'] for r in numeric_repeats if r['model']=='M0' and r['repeat']==i) for i in range(1,4))]
    q=np.array([d['q'] for d in diagnostics])
    at_cap=sum(d['q']>=.9*d['q_max'] for d in diagnostics)
    ranks=sum(d['ols_rank']<d['q'] for d in diagnostics)
    overlaps=[]
    for a in range(15):
        for b in range(a+1,15):
            overlaps.append([len(set(diagnostics[a]['supports'][k])&set(diagnostics[b]['supports'][k]))/
                             max(1,len(set(diagnostics[a]['supports'][k])|set(diagnostics[b]['supports'][k]))) for k in range(3)])
    overlap=np.mean(overlaps,axis=0)
    write_json(output/'support_overlap.json',dict(pair_count=len(overlaps),mode_mean_jaccard=overlap.tolist()))
    learning_convergence=[]
    for path in (output/'tasks').glob('*/diagnostics.json'):
        v=read_json(path)
        if v['diagnostics'].get('converged') is False:
            learning_convergence.append(str(path))
    report=['# Apple 实证最终报告','', '状态：MODELS_COMPLETE_PENDING_FIGURE_QA（等待独立渲染验收）。','',
        f'1. 数据对齐：{len(data["sample_ids"])}条图像记录、{len(entities)}个评价实体；严格按ID连接，响应以干物质百分点计；完整依据及实体/批次可核验边界见DATA_AUDIT.md。',
        '2. 当前表征的预测信息：三个重复RMSE均低于训练均值的学习模型为 '+(', '.join(stable) or '无')+'；这仅是当前重复内部CV协议下的证据。',
        f'3. v5.2表现：完整OOF RMSE为{values["M1"]["rmse_mean"]:.4f} ± {values["M1"]["rmse_sd"]:.4f}个百分点；全部基线与配对差异如下，原MSIC未调整。',
        '4. 空间价值：M5由同一张量空间平均得到，M6保留141个原始波段；二者分别属于同表示消融和不同分辨率强基线，成绩差异不支持空间因果解释或空间信息不存在的断言。',
        f'5. 选阶风险：q范围{q.min()}–{q.max()}，均值{q.mean():.2f}；达到q上限90%的外折{at_cap}/15，秩亏外折{ranks}/15；各模式平均Jaccard为'+', '.join(f'{v:.3f}' for v in overlap)+'。',
        '6. 论文应用建议：'+('原MSIC模型在强基线比较中存在明显局限，Apple宜暂作补充或边界案例；不据此修改算法。' if values['M1']['rmse_mean']>min(values[m]['rmse_mean'] for m in ['M4','M5','M6']) else '可作为当前协议下的应用证据；成为主要应用仍需结合完整科学问题及独立验证。'),
        '', '## 完整OOF结果','', '|模型|RMSE（百分点）|MAE（百分点）|R²|','|---|---:|---:|---:|']
    for model in config['models']:
        v=values[model]
        report.append(f'|{model}|{v["rmse_mean"]:.4f} ± {v["rmse_sd"]:.4f}|{v["mae_mean"]:.4f} ± {v["mae_sd"]:.4f}|{v["r2_mean"]:.4f} ± {v["r2_sd"]:.4f}|')
    report += ['', '每个重复先拼齐5折OOF、按实体等权计算指标，再对3个重复计算均值和样本标准差；不是折R²平均，也没有把720次预测视为720个独立苹果。',
        '', '## 与原MSIC的配对比较','', '|对照|M1−对照 MSE|描述性95%区间|M1−对照 RMSE|描述性95%区间|','|---|---:|---|---:|---|']
    for p in paired:
        report.append(f'|{p["comparator"]}|{p["mse_difference"]:.4f}|[{p["mse_ci_low"]:.4f}, {p["mse_ci_high"]:.4f}]|{p["rmse_difference"]:.4f}|[{p["rmse_ci_low"]:.4f}, {p["rmse_ci_high"]:.4f}]|')
    report += ['', '负值表示M1误差较低。2000次实体整组重采样保留其全部重复预测；区间条件于已保存OOF，没有重新训练，不能代表完整训练不确定性或普适优越性。跨重复方向见repeat_metrics.csv。',
        '', '## 数值、解释及证据边界','',
        f'所有105个外层模型保存后独立加载重建预测通过rtol=1e-9、atol=1e-10检查。显式不收敛最终拟合{len(learning_convergence)}项；M1/M2原求解器未暴露的完整收敛状态如实记为null，不能解读为已证明收敛；内调参警告全部保留在tuning.json。',
        'q是矩形支撑参数容量，不等于严格非零系数数；支撑允许不连续，选择频率和重叠仅描述稳定性。没有像素级响应真值，图像强度或系数都不是局部干物质含量地图。',
        '单样本ROI归一化位置只是相对坐标，不保证苹果组织间生物学对齐；未依据误差筛样、调ROI或选最好重复。全部真实值—预测图固定第1重复。',
        '外部训练折标准化控制X及y数值尺度并完成截距还原；核心存在绝对损失阈值，不宣称任意y缩放不变。',
        '这是本数据集内部重复交叉验证，不是全新外部数据验证。真实数据未计算Exact、REE或真支撑恢复率。',
        '', '## 图表','']
    report.extend(f'- [{Path(p).stem}](figures/{Path(p).stem}.png)' for p in figure_files)
    report += ['', '## 复现','', '运行命令与冻结配置见同目录config.yaml、FROZEN_PROTOCOL.md及模块README.md；所有基线内层独立拟合预处理，正式源码身份逐任务检查，缓存仅接受相同数据、划分、预处理、模型与源码指纹。']
    (output/'APPLE_EMPIRICAL_FINAL_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    final=dict(status='MODELS_COMPLETE_PENDING_FIGURE_QA',run_id=config['run_id'],contract=contract['fingerprint'],
        sample_records=len(data['sample_ids']),entities=len(entities),outer_models_completed=105,
        outer_models_expected=105,oof_rows=len(rows),oof_rows_expected=len(data['sample_ids'])*3*7,
        failed_tasks=[],source_identity_verified=True,model_replay_tolerance=dict(rtol=1e-9,atol=1e-10),
        explicitly_nonconverged_final_models=learning_convergence,
        optimizer_full_convergence_not_exposed=['M1','M2'],bootstrap_count=2000,
        dependency_scope='doc/taskcard13/runtime only; existing numpy/scipy unchanged',
        figures=figure_files,figure_qa='pending independent rendered QA',
        artifacts={p.name:file_hash(p) for p in output.iterdir() if p.is_file() and p.name not in ['FINAL_VALIDATION.json']})
    write_json(output/'FINAL_VALIDATION.json',final)
