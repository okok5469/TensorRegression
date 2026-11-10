# Tensor-ABESS OvR-LS 0.1.0

这是对冻结的v5.2-search-final＋原MSIC的多分类应用封装。全部K个0/1指示响应分别减训练频率、调用原回归器，再恢复频率，按最大原尺度分数分类；各类允许不同矩形支撑。分数不是概率，原MSIC不是多项似然准则，既有回归模拟不证明分类一致性。软件可交付不意味着分类优于对照。

要求64位CPython3.12。Windows在独立venv实测；Linux入口提供但当前UNVERIFIED。依赖版本见requirements-lock.txt。包内wheel包含正式32源，不能再用另一份tensor_abess覆盖它。

## 三步开始（Windows PowerShell）

在新目录解压，进入包含本文件的目录：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-index --find-links .\wheelhouse .\dist\tensor_abess_ovr_ls-0.1.0-py3-none-any.whl
python -m tensor_abess_ovr.cli verify-release
python -m tensor_abess_ovr.cli smoke --output .\smoke_first
```

然后按DATA_CONTRACT.md准备自己的数据与开发protocol，先校验再运行：

```powershell
python -m tensor_abess_ovr.cli validate-data --data .\my_dataset --config .\my_protocol.json
python -m tensor_abess_ovr.cli run --data .\my_dataset --config .\my_protocol.json --output .\my_dev_run
python -m tensor_abess_ovr.cli validate-return --run .\my_dev_run
python -m tensor_abess_ovr.cli pack-return --run .\my_dev_run --output .\my_dev_return.zip
```

回传最后一个ZIP。输出目录必须全新；已有运行不会被覆盖。smoke的固定小fixture只检查安装和程序行为，不能作为分类性能证据，也不要求100%准确率。

## Linux入口（未实测）

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps dist/tensor_abess_ovr_ls-0.1.0-py3-none-any.whl
bash run.sh verify-release
bash run.sh smoke --output smoke_first
```

wheelhouse中的二进制依赖是Windows专用；Linux从锁定requirements安装相应平台发行文件，先完成同样的verify/smoke。平台浮点差异如实记录，不宣称未测平台逐位一致。

## 故障定位

verify打印实际import路径、环境和源码身份；路径应在本次venv内。请使用干净venv，清除旧PYTHONPATH影响，不手改核心或参数来解决错误。输入NaN/Inf、缺类、轴语义冲突、重复sample_id、group/time越界会直接报出位置；按原始数据和声明修正，不静默删行。

Logistic候选及最终拟合保存迭代与警告；同目标只允许5000→20000步，未解决则比较未验收。每类核心900s、数据集7200s是默认资源预算，超时保存失败。资源扩充须另存协议，不改既有运行。默认单进程，数值线程由CLI在导入前固定为1。

本包不自动评价保留测试集。evaluation_stage默认dev，只运行active_split_ids列明且用途相符的划分。final需独立冻结配置，测试用过后不能再称未见。原算法初始化500步保持身份；正常返回不等于所有内层问题有最优性证书。

详见TEAMMATE_TASKCARD.md、METHOD.md、DATA_CONTRACT.md及RESULT_RETURN.md。此包内部交付，不自动发布、不发送邮件；原源码版权/许可说明随源保留，不额外授予公共许可。
