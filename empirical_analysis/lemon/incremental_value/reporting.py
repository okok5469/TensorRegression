"""Tables and five quantitative figures from persisted real OOF predictions."""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
from .metrics import MODELS, BASELINE

COLORS = ('#666666','#0072B2','#D55E00','#009E73','#CC79A7')


def _csv(path, rows):
    rows=list(rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def _n(value):
    return 'NA' if value is None else f'{value:.4f}'


def _finish(fig, axes, name, directory, layouts):
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    layouts[name]={'figure_inches':fig.get_size_inches().tolist(),
                   'axes':[{'bbox_figure_fraction':list(a.get_position().bounds),'title':a.get_title()}
                           for a in np.asarray(axes,dtype=object).ravel()],
                   'source':'saved_complete_oof_predictions','png_dpi':300}
    root=os.environ.get('NATURE_FIGURE_SKILL_ROOT')
    if root:
        tools_dir=str(Path(root)/'scripts')
        sys.path.insert(0,tools_dir)
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
            require_matplotlib_panel_alignment(fig,json_out=directory/(name+'.alignment.json'),
                                               tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        finally:
            sys.path.remove(tools_dir)
    else:
        layouts[name]['external_alignment_audit']='not run: optional QA tool not configured'
    fig.savefig(directory/(name+'.png'),dpi=300)
    fig.savefig(directory/(name+'.pdf'))
    fig.savefig(directory/(name+'.svg'))
    plt.close(fig)


def generate_reports(output_dir, config, audit, stats, fold_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output_dir); cv=output/'cv'; figs=output/'figures'; reports=output/'reports'
    figs.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True)
    for key,filename in {'metrics_by_repeat':'model_metrics_by_repeat.csv','model_summary':'model_summary.csv',
                         'incremental_metrics':'incremental_metrics.csv','per_subject_improvement':'per_subject_improvement.csv'}.items():
        _csv(cv/filename,stats[key])
    with (cv/'oof_predictions.csv').open(encoding='utf-8',newline='') as f:
        oof=list(csv.DictReader(f))
    summary={r['model']:r for r in stats['model_summary']}
    reps=sorted({int(r['repeat']) for r in stats['metrics_by_repeat']})
    perf={(r['model'],int(r['repeat'])):r for r in stats['metrics_by_repeat']}
    inc={r['model']:r for r in stats['incremental_metrics'] if r['repeat']=='all'}
    inc_rep={(r['model'],int(r['repeat'])):r for r in stats['incremental_metrics'] if r['repeat']!='all'}
    layouts={}
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'font.size':9,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        positions=np.arange(len(MODELS))
        for ax,metric,label in zip(axes,('rmse','r2'),('OOF RMSE (log TMT-B)','OOF R²')):
            ax.bar(positions,[summary[m][metric+'_mean'] for m in MODELS],color=COLORS,alpha=.8)
            for i,m in enumerate(MODELS):
                ax.scatter([i]*len(reps),[perf[(m,r)][metric] for r in reps],color='black',s=18,zorder=3)
            ax.set_xticks(positions,MODELS,rotation=25,ha='right',rotation_mode='anchor',fontsize=8)
            ax.set_ylabel(label);ax.axhline(0,color='black',lw=.6)
        fig.suptitle('Complete OOF performance: points are CV repeats')
        _finish(fig,axes,'model_comparison',figs,layouts)
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        x=np.arange(4)
        ax.bar(x,[inc[m]['delta_rmse'] for m in MODELS[1:]],color=COLORS[1:],alpha=.8)
        low=[inc[m]['delta_rmse_ci_low'] for m in MODELS[1:]]
        high=[inc[m]['delta_rmse_ci_high'] for m in MODELS[1:]]
        ax.vlines(x,low,high,color='black',lw=1.2)
        ax.scatter(x,low,marker='_',color='black');ax.scatter(x,high,marker='_',color='black')
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,MODELS[1:])
        ax.set_ylabel('ΔRMSE: covariates − EEG model')
        ax.set_title('Pooled repeats: descriptive 95% subject-cluster intervals')
        _finish(fig,[ax],'incremental_rmse',figs,layouts)
        subjects=sorted([r for r in stats['per_subject_improvement'] if r['model']=='Tensor-ABESS'],
                        key=lambda r:r['mean_improvement'])
        difference=np.array([r['mean_improvement'] for r in subjects])
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        ax.vlines(np.arange(len(difference)),0,difference,
                  colors=np.where(difference>=0,COLORS[1],COLORS[2]),lw=1)
        ax.axhline(0,color='black',lw=.8)
        ax.set_xlabel('Subjects ranked by cross-repeat mean error improvement')
        ax.set_ylabel('Covariate squared error − ABESS squared error')
        ax.set_title('Paired per-subject error difference (all subjects retained)')
        _finish(fig,[ax],'paired_error_difference',figs,layouts)
        selected_models=(BASELINE,'Tensor-ABESS','Ridge')
        scatter_rows=[r for r in oof if r['model'] in selected_models]
        values=np.array([[float(r['y_true']),float(r['y_pred'])] for r in scatter_rows])
        lo,hi=float(values.min()),float(values.max());padding=.03*(hi-lo)
        limits=(lo-padding,hi+padding)
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,m in zip(axes,selected_models):
            rows=[r for r in oof if r['model']==m]
            ax.scatter([float(r['y_true']) for r in rows],[float(r['y_pred']) for r in rows],
                       color=COLORS[MODELS.index(m)],s=12,alpha=.35)
            ax.plot(limits,limits,color='black',ls='--',lw=.8)
            ax.set_xlim(limits);ax.set_ylim(limits)
            ax.set_xlabel('Observed log(TMT-B)');ax.set_ylabel('OOF predicted log(TMT-B)');ax.set_title(m)
        fig.suptitle('Repeated OOF predictions: each subject once per repeat')
        _finish(fig,axes,'predicted_vs_observed',figs,layouts)
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
        for m,c in zip(MODELS,COLORS):
            axes[0].plot(reps,[perf[(m,r)]['rmse'] for r in reps],marker='o',color=c,label=m)
        for m,c in zip(MODELS[1:],COLORS[1:]):
            axes[1].plot(reps,[inc_rep[(m,r)]['delta_rmse'] for r in reps],marker='o',color=c)
        for ax in axes:
            ax.set_xticks(reps);ax.set_xlabel('Outer CV repeat')
        axes[0].set_ylabel('Complete OOF RMSE');axes[1].set_ylabel('ΔRMSE versus covariates')
        axes[1].axhline(0,color='black',lw=.8)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,fontsize=8)
        _finish(fig,axes,'repeat_stability',figs,layouts)
    (figs/'figure_layout.json').write_text(json.dumps(layouts,indent=2),encoding='utf-8')
    gaps=[]
    for m in MODELS:
        fits=[r for r in fold_rows if r['model']==m]
        train=np.array([r['train_rmse'] for r in fits]);test=np.array([r['test_rmse'] for r in fits])
        gaps.append({'model':m,'n_outer_fits':len(fits),'mean_fold_train_rmse':float(train.mean()),
                     'mean_fold_test_rmse':float(test.mean()),'mean_fold_test_minus_train_rmse':float((test-train).mean()),
                     'total_runtime_sec':float(sum(r['runtime_sec'] for r in fits))})
    _csv(cv/'train_test_gap.csv',gaps)
    positive={m:sum(inc_rep[(m,r)]['delta_rmse']>0 for r in reps) for m in MODELS[1:]}
    lines=['# LEMON EEG 增量预测价值实验','',
           f'有效受试者 {audit["n_subjects"]} 名，张量 {audit["tensor_shape"]}；已完成 {len(reps)} 次完整五折外层验证，所有模型共享划分。',
           '','## 完整 OOF 性能','',
           '每次重复先合并全部受试者OOF预测，再计算指标；表中为重复间均值，SD为样本标准差。',
           '','|模型|RMSE|RMSE SD|MAE|R²|','|---|---:|---:|---:|---:|']
    for m in MODELS:
        r=summary[m];lines.append(f'|{m}|{_n(r["rmse_mean"])}|{_n(r["rmse_sd"])}|{_n(r["mae_mean"])}|{_n(r["r2_mean"])}|')
    lines+=['','## EEG增量收益','',
            '总汇对受试者/重复等权汇总MSE后计算RMSE差；它不必等于各重复RMSE差的平均。正值表示EEG改善。',
            '','|模型|ΔRMSE|95%描述性区间|增量R²|正收益重复数|改善受试者比例|',
            '|---|---:|---|---:|---:|---:|']
    for m in MODELS[1:]:
        r=inc[m];lines.append(f'|{m}|{_n(r["delta_rmse"])}|[{_n(r["delta_rmse_ci_low"])}, {_n(r["delta_rmse_ci_high"])}]|{_n(r["incremental_r2"])}|{positive[m]}/{len(reps)}|{r["improved_subject_fraction"]:.1%}|')
    lines+=['','区间使用2000次受试者聚类bootstrap，同一人的所有重复预测共同重抽样；条件于已保存OOF，未在bootstrap中重新训练模型，不代表严格总体显著性或因果效应。',
            '','## 逐人误差与训练测试差距','',
            '|模型|逐人改善中位数|恶化比例|绝对差异最大10%人群贡献份额|','|---|---:|---:|---:|']
    for m in MODELS[1:]:
        r=inc[m];lines.append(f'|{m}|{_n(r["median_subject_improvement"])}|{r["worsened_subject_fraction"]:.1%}|{r["top_absolute_10pct_share"]:.1%}|')
    lines+=['','贡献份额以逐人跨重复平均差异的绝对值为分母，仅诊断集中性；不据此删除受试者。',
            '','|模型|折均训练RMSE|折均测试RMSE|测试减训练|总拟合秒数|','|---|---:|---:|---:|---:|']
    for r in gaps:
        lines.append(f'|{r["model"]}|{_n(r["mean_fold_train_rmse"])}|{_n(r["mean_fold_test_rmse"])}|{_n(r["mean_fold_test_minus_train_rmse"])}|{r["total_runtime_sec"]:.1f}|')
    eeg_best=min(MODELS[1:],key=lambda m:summary[m]['rmse_mean'])
    positive_mean=[m for m in MODELS[1:] if inc[m]['delta_rmse']>0]
    stable=[m for m in positive_mean if positive[m]==len(reps)]
    lines+=['','## 实际结论','',f'五种模型中平均RMSE最低的是 {min(MODELS,key=lambda m:summary[m]["rmse_mean"])}；EEG模型内部最低的是 {eeg_best}。',
            f'Tensor-ABESS在 {positive["Tensor-ABESS"]}/{len(reps)} 次重复中改善协变量基线。']
    if not positive_mean:
        lines.append('本次属于情形A：所有EEG模型的总汇增量均不为正，当前EEG特征与验证流程尚未显示可靠增量价值；这不能推出EEG完全不含相关信息。')
    else:
        lines.append(f'总汇正增量模型：{", ".join(positive_mean)}；各次重复均为正的模型：{", ".join(stable) or "无"}。需结合区间和改善幅度解释，不能凭内部排名判定成功。')
    abess_fits=[r for r in fold_rows if r['model']=='Tensor-ABESS']
    largest=max(abess_fits,key=lambda r:float(r['test_rmse']))
    key=(int(largest['repeat']),int(largest['outer_fold']))
    baseline_lookup={(r['subject_id'],int(r['repeat'])):r for r in oof if r['model']==BASELINE}
    all_differences=[];fold_differences=[]
    for r in oof:
        if r['model']!='Tensor-ABESS':
            continue
        b=baseline_lookup[(r['subject_id'],int(r['repeat']))]
        d=(float(r['y_true'])-float(b['y_pred']))**2-(float(r['y_true'])-float(r['y_pred']))**2
        all_differences.append(d)
        if (int(r['repeat']),int(r['outer_fold']))==key:
            fold_differences.append(d)
    share=sum(fold_differences)/sum(all_differences) if sum(all_differences)!=0 else None
    lines+=['','## Tensor-ABESS波动的定位','',
            f'ABESS最差外折为重复标识{key[0]}、外折{key[1]}（重复从0编号），选择q={largest["q"]}，'
            f'训练RMSE={_n(float(largest["train_rmse"]))}、测试RMSE={_n(float(largest["test_rmse"]))}。',
            f'该外折贡献全部三次重复ABESS相对协变量的总SSE差异的{share:.1%}。这是误差贡献诊断，所有样本和外折仍保留在主结果中。',
            f'其余外折q范围为{min(int(r["q"]) for r in abess_fits if r is not largest)}至'
            f'{max(int(r["q"]) for r in abess_fits if r is not largest)}；这一容量跳变和训练/测试差距支持进一步检查MSIC及支持搜索稳定性，尚不足以确定哪个算法环节是唯一原因。']
    parameter_counts={}
    for m in ('Vector Lasso','Tensor-GLasso','Ridge'):
        values=[json.loads(r['selected_parameters']) for r in fold_rows if r['model']==m]
        chosen=[p.get('lambda',p.get('lambda_ratio')) for p in values]
        counts={str(v):chosen.count(v) for v in sorted(set(chosen))}
        parameter_counts[m]=counts
    (output/'diagnostics/selected_parameter_counts.json').write_text(json.dumps(parameter_counts,indent=2),encoding='utf-8')
    lines+=['','## 正则化选择诊断','',
            '参数选择频数：'+json.dumps(parameter_counts,ensure_ascii=False)+'。',
            '强惩罚或网格上界频繁被选择，说明内层验证更倾向抑制EEG项；上界选择也提示网格边界需披露。若扩展网格，应当作为新的预设验证实验，不能根据本次外层结果反复追调。']
    lines.append('Vector Lasso的ratio=1对应零EEG系数；Tensor-GLasso保留原非空支持回退及OLS重拟合，因此ratio=1仍可能有非零EEG项，不能把它解释成完全退回协变量基线。Ridge选择网格上界仅说明倾向强收缩，不能证明唯一最佳lambda已定位。')
    lines+=['','## 方法审计与后续','',
            '旧外层预处理隔离正确，但Lasso/GLasso内部CV使用外层预处理数据而未逐内折重新拟合；本次外部内层CV全部重拟合X/y残差化、中心化和缩放。年龄保留原分类哑编码，性别1=female→0、2=male→1。',
            'Vector Lasso和Tensor-GLasso的lambda比例固定网格，由每个当前训练局部设计计算lambda_max；这是新的透明外部调参政策，保留原ISTA、矩形支持及OLS定义。Ridge使用RSS/(2n)+lambda||beta||²/2的对偶解。',
            'Tensor-ABESS保持原BIC初始化及终版自适应MSIC；固定MSIC训练准则不是独立验证误差。未根据外层测试调节MSIC或核心算法。',
            '若ABESS训练误差明显较低而测试误差较高，优先诊断容量、矩形支持约束与MSIC偏好。后续应在独立验证或新预先固定嵌套方案内比较MSIC设置、通道/时窗/频带结构，不能根据本轮外层测试追调参数再声称独立验证。',
            '','## 产物','',
            'cv/：OOF预测、外层划分、内层选参、各重复指标、总汇、增量收益、逐人差异及训练测试差距。',
            'figures/：模型比较、增量收益、逐人差异、真实值散点和重复稳定性五类PNG/PDF及布局审计。',
            'diagnostics/：数据来源、输入/源码指纹、预处理规范、折划分泄漏检查。config_used.json和run_state.json记录正式状态。']
    (reports/'EEG_INCREMENTAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (reports/'EXECUTION_SUMMARY.md').write_text('# 执行结果\n\n'+
        f'已生成{len(reps)}次完整外层重复、五种模型、{audit["n_subjects"]}名受试者的统计与图表。\n\n'+
        '主实验状态以run_state.json为准；单元测试、smoke、真实计算与独立复算证据分别记录，不相互替代。\n'+
        '详情见EEG_INCREMENTAL_REPORT.md，图版最终视觉及PDF审计记录见figures/。\n',encoding='utf-8')
