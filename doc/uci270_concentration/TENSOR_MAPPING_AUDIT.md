> **BLOCKED / NOT FOR SCIENTIFIC DELIVERY.** Official class-ID definitions conflict with the official gas-by-batch count table and local DAT counts. Chemical names below are provisional and are NOT semantically accepted. Earlier Gate D PASS is withdrawn; numerical fits retain the original gas_id without relabeling. See [SCIENCE_REVIEW.md](SCIENCE_REVIEW.md).

# 张量映射审计

全部 13,910 条 × 96 坐标（1,335,360 值）逐项检查通过：X[j,k,0] = raw[8j+2+k]，X[j,k,1] = raw[8j+5+k]（零基）。先把每传感器六动态列拆为阶段×尺度，再转置为传感器×尺度×阶段；禁止直接 reshape 96 列。

data/metadata.json 保存全部原列号、C-order 坐标与排除的 32 个稳态列。人工单调输入单元测试检查完整方向。用第一批训练均值/标准差检查全部记录标准化前后映射一致，无标签参与。

C-order 展平内积与张量收缩最大绝对差 1.81898940355e-12（浮点加法次序）；rtol=1e-12、atol=1e-8 通过。

轴语义：16 传感器 × 3 平滑参数（0.001/0.01/0.1）× 2 响应阶段（i/d）。尺度不是原始时间，阶段不是气体；DAT 不含原始波形。矩形支撑是假设，第三轴只有 2 个坐标。数学映射与官方字段语义均通过，独立来源证据见 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)。
