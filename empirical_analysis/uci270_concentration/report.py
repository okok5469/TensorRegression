"""Build Chinese evidence reports from immutable, audited task outputs."""
import json
import pickle
from pathlib import Path
import numpy as np
from .analysis import collect

NAMES={1:'乙醇',2:'乙烯',3:'氨',4:'乙醛',5:'丙酮',6:'甲苯'}
METHODS={'M0':'训练均值','M1':'Tensor-ABESS96','M2':'Tensor-GLasso96','M3':'Lasso96','M4':'Ridge96','M5':'PLS96','M6':'Ridge128','M7':'Lasso128'}


def fmt(value):
    return 'NA' if value is None else f'{value:.4g}'


def table(rows,columns):
    return '| '+' | '.join(label for _,label in columns)+' |\n|'+ '|'.join(['---']*len(columns))+'|\n'+''.join(
        '| '+' | '.join(fmt(r[key]) if isinstance(r[key],(float,np.floating)) or r[key] is None else str(r[key]) for key,_ in columns)+' |\n' for r in rows)


def build(root):
    root=Path(root); result=collect(root);summary=result['summary'];comp=result['complexity']
    final=[r for r in summary if r['split']=='final']
    if len(summary)!=192 or len(final)!=48:raise ValueError('Formal report requires complete192 tasks')
    index={(r['gas_id'],r['split'],r['method']):r for r in summary}
    ci={(r['gas_id'],r['split'],r['method']):r for r in comp}
    comparisons=[]
    for gas in range(1,7):
        a=index[(gas,'final','M1')];same=[index[(gas,'final',m)] for m in ('M2','M3','M4','M5')]
        best=min(same,key=lambda r:r['rmse']);full=min([index[(gas,'final',m)] for m in ('M6','M7')],key=lambda r:r['rmse'])
        c=ci[(gas,'final','M1')]
        comparisons.append(dict(gas=NAMES[gas],rmse=a['rmse'],r2=a['r2'],mean_rmse=index[(gas,'final','M0')]['rmse'],
            best96=METHODS[best['method']],best96_rmse=best['rmse'],delta96=a['rmse']-best['rmse'],
            best128_rmse=full['rmse'],nnz=c['effective_nnz'],sensors=c['n_sensors'],outside=a['outside_n']))
    finite_positive=sum(r['r2'] is not None and r['r2']>0 for r in comparisons)
    beats_mean=sum(r['rmse']<r['mean_rmse'] for r in comparisons)
    wins=sum(r['delta96']<0 for r in comparisons)
    allwarn=[];convergence=[]
    for marker in sorted((root/'models').glob('*/COMPLETE.json')):
        folder=marker.parent
        with (folder/'model.pkl').open('rb') as f:a=pickle.load(f)
        scopes=[('outer',None,None,a['diagnostics'])]+[('inner',r['parameter'],f['fold'],f['diagnostics']) for r in a['cv_records'] for f in r['folds']]
        for scope,param,fold,d in scopes:
            convergence.append(dict(task=folder.name,scope=scope,parameter=param,fold=fold,converged=d['converged'],
                                    selected=(scope=='outer' or param==a['parameters']),warnings=d['warnings'],attempts=d.get('attempts',[])))
            if d['warnings']:allwarn.append(convergence[-1])
    (root/'convergence_audit.json').write_text(json.dumps(convergence,ensure_ascii=False,indent=2),encoding='utf-8')
    unresolved=[r for r in convergence if r['converged'] is False]
    selected_bad=[r for r in unresolved if r['selected']]
    opening=f'''# UCI270 六类浓度回归：正式实证报告

## 一页结论

192/192 个外层模型已完成；这是计算完成数，数据正确性、数值重放和预测价值分别验收，最终验收状态见 FINAL_VALIDATION.json。

1. 数据可用于已知气体类别下的浓度估计：13,910条记录、6类、10个时间批次，浓度直接来自原始标签；两个独立解析器一致，ZIP CRC/本地副本一致。没有把类别编码当浓度，也没有把13,910条记录当成独立设备。
2. 16×3×2严格对应传感器、平滑尺度、上升/衰减阶段；全部1,335,360个动态坐标检查通过。96/128维完全重复组均为0，因此本次没有训练端重复清除；仍保存并验证统一清除规则。
3. 第10批中，ABESS有{finite_positive}/6类R²为正，{beats_mean}/6类RMSE低于历史训练均值；相对同96维最佳竞争方法，有{wins}/6类误差更低。各类误差不能混成一个ppmv总误差。
4. 稀疏性同时计算全部方法的实际非零数、传感器数和矩形外包容量。ABESS的q不是完整自由度，Lasso的外包矩形不是其实际系数数；少系数也不能直接换算为少检测成本。
5. 共有{len(unresolved)}次候选/外层拟合在既定数值预算后明确未收敛，其中{len(selected_bad)}次属于最终选择的参数对应内折或外层；ABESS/GLasso未暴露的完整收敛状态保持unknown。不能给出无条件的主实证推荐。
6. 这是同一传感器系统的一次未来批次验证。未知试次/设备依赖、只有三个重叠开发窗和一个最终批次，限制了统计推断；不宣称显著优越、总体非劣或支撑恢复。

'''
    columns=[('gas','气体'),('rmse','ABESS RMSE'),('r2','R²'),('mean_rmse','均值RMSE'),('best96','最佳同输入对照'),('best96_rmse','其RMSE'),('delta96','ABESS−对照'),('best128_rmse','最佳128维RMSE'),('nnz','有效非零'),('sensors','传感器'),('outside','范围外n')]
    body=opening+table(comparisons,columns)+'\n误差单位均为ppmv；每类最终测试600条；差值为负表示ABESS误差较低，比较对象是实际列出的四个96维竞争方法。排名仅是当前批次的描述。\n\n'
    body+='## 六类全部方法与四个时间窗口\n\n'
    for gas in range(1,7):
        rows=[dict(r,method=METHODS[r['method']]) for r in summary if r['gas_id']==gas]
        body+=f'### {NAMES[gas]}\n\n'+table(rows,[('split','窗口'),('method','方法'),('n_train','训练n'),('n_test','测试n'),('rmse','RMSE'),('mae','MAE'),('r2','R²'),('bias','偏差'),('train_rmse','训练RMSE'),('outside_n','范围外n')])+'\n'
    body+='''## 固定协议与解释边界

训练1—6/1—7/1—8批分别评价第7/8/9批；开发审计锁定后，训练1—9批只生成一次第10批预测。所有方法使用同一记录ID；内层保留最近至多3个合格滚动边界，至少2个。gas6 dev1使用边界1和5，边界1验证响应恒定仍可计算MSE，不计算R²。每层X/y缩放仅用该层训练样本，未裁剪预测、筛极端浓度或按成绩改协议。

主96维排除了32个稳态量；Ridge96↔Ridge128及Lasso96↔Lasso128才是同模型信息完整性比较，ABESS↔128维的差异不能全部归因于算法。有限浓度档位不是随机连续抽样，范围内不保证无漂移；范围外误差不能单独证明漂移是唯一原因。

最终测试前的完整源码/数据/config合同和两道验收锁均已保留。一次工程修订仅补齐内折预测留痕，发生在任何真实拟合前，旧版本保存在pre_audit_fix；未改科学协议。ABESS容忍带与原生候选选择在真实Pilot上额外复跑抽检，其余ABESS模型的支撑、OLS与MSIC逐一复算，其他方法核对预测重放与选参；不把有限搜索停止解释为全局最优。

## 图表与复现

- [最终RMSE](figures/final_rmse.pdf)
- [开发至最终误差](figures/temporal_error.pdf)
- [预测与实测](figures/final_predictions.pdf)
- [训练范围分层](figures/range_error.pdf)
- [误差与实际非零数](figures/error_complexity.pdf)
- [模式选择频率](figures/abess_mode_frequency.pdf)

来源和引用见[SOURCE_AUDIT.md](SOURCE_AUDIT.md)，数据质量见[DATA_AUDIT.md](DATA_AUDIT.md)，映射见[TENSOR_MAPPING_AUDIT.md](TENSOR_MAPPING_AUDIT.md)，协议见[FROZEN_PROTOCOL.md](FROZEN_PROTOCOL.md)。本地哈希用于复现，并非官方服务器认证。
'''
    (root/'UCI270_EMPIRICAL_FINAL_REPORT.md').write_text(body,encoding='utf-8')
    drift='# 预测与跨批次变化\n\n开发训练窗互相重叠，各批次分别评价；不混合六类原始量程计算总R²。宏平均标准化RMSE先除以各类当前训练y标准差，再六类等权平均。\n\n'
    macro=[dict(split=s,method=METHODS[m],macro_normalized_rmse=float(np.mean([r['normalized_rmse'] for r in summary if r['split']==s and r['method']==m]))) for s in ('dev1','dev2','dev3','final') for m in METHODS]
    drift+=table(macro,[('split','窗口'),('method','方法'),('macro_normalized_rmse','宏平均标准化RMSE')])
    delta=[]
    for gas in range(1,7):
        for s in ('dev1','dev2','dev3','final'):
            for m in ('M2','M3','M4','M5','M6','M7'):
                delta.append(dict(gas=NAMES[gas],split=s,comparison='ABESS − '+METHODS[m],delta=index[(gas,s,'M1')]['rmse']-index[(gas,s,m)]['rmse']))
    drift+='\n## 按类别同窗口配对误差差\n\n'+table(delta,[('gas','气体'),('split','窗口'),('comparison','比较'),('delta','RMSE差ppmv')])
    info=[]
    for gas in range(1,7):
        for m,full in [('M4','M6'),('M3','M7')]:
            info.append(dict(gas=NAMES[gas],method=METHODS[m],delta=index[(gas,'final',m)]['rmse']-index[(gas,'final',full)]['rmse']))
    drift+='\n## 舍弃32个稳态量的代价\n\n正值表示完整128维较好，负值表示当前96维较好；未以此重选输入。\n\n'+table(info,[('gas','气体'),('method','96维模型'),('delta','96−128 RMSE')])
    drift+='\n范围内/低于/高于训练范围的全部计数与误差见range_metrics.csv；缺记录的分层指标为NA。只有一个final批次，不用零SD伪装零不确定性。批次变化同时混合浓度覆盖、响应漂移和模型限制，本设计不能作因果分解。\n'
    (root/'PREDICTION_AND_DRIFT_ANALYSIS.md').write_text(drift,encoding='utf-8')
    sparse='# 同输入复杂度、稳定性与成本\n\n阈值为标准化X/y坐标下abs(beta)>1e-8×max(1,max(abs(beta)))；只描述数值，不改预测。传感器编号1—16，尺度索引0/1/2对应.001/.01/.1，阶段0/1对应i/d。\n\n'
    sparse+=table([dict(r,method=METHODS[r['method']],gas=NAMES[r['gas_id']]) for r in comp if r['split']=='final'],[('gas','气体'),('method','方法'),('q','原生矩形q'),('strict_nnz','严格非零'),('effective_nnz','有效非零'),('n_sensors','传感器'),('n_scales','尺度'),('n_stages','阶段'),('envelope_capacity','外包容量'),('selected_rank','选中设计秩'),('pls_components','PLS成分'),('total_seconds','训练+调参秒')])
    stable=[]
    for gas in range(1,7):
        for m in METHODS:
            for axis in ('sensors','scales','stages'):
                values=[r['jaccard'] for r in result['stability'] if r['gas_id']==gas and r['method']==m and r['axis']==axis]
                stable.append(dict(gas=NAMES[gas],method=METHODS[m],axis=axis,jaccard=float(np.mean(values)) if values else None))
    sparse+='\n## 三个开发窗的两两Jaccard均值\n\n'+table(stable,[('gas','气体'),('method','方法'),('axis','模式'),('jaccard','均值')])
    sparse+='\n两个空集Jaccard定义为1，是“同时未选”的记号，不代表有可用预测信号。三个训练窗重叠，频率分母仅3，不解释为独立重复或支撑恢复。final支撑不计入频率。128维不将稳态量强塞入尺度/阶段；PLS成分少不等于传感器少。GLasso表中q指最终矩形OLS支撑，原生组状态见model_complexity.csv。预测计时受系统影响，是描述性测量，不是硬件公平跑分。\n'
    (root/'SPARSITY_AND_STABILITY_REPORT.md').write_text(sparse,encoding='utf-8')
    return dict(final_comparisons=comparisons,unresolved_fits=len(unresolved),selected_unresolved_fits=len(selected_bad),warning_fits=len(allwarn))


if __name__=='__main__':
    import sys
    sys.path[:0]=[str(Path('doc/taskcard13/runtime').resolve()),str(Path('src').resolve())]
    build('doc/uci270_concentration')
