# nano-vLLM 混合调度优化

基于 nano-vLLM 实现 **Decode 优先 + Token Budget 驱动的 Chunked Prefill**，缩短长 Prompt 加入时已有请求的生成停顿。在 RTX 5080 / Qwen3-0.6B 的两组长输入负载下，最大停顿分别降低 **72.56%**、**89.81%**。

## 发现问题

先让 4 条请求进入 Decode，再加入 1 条或 4 条 3000-token Prompt。预算为 512 时，原调度下已有请求的最大停顿分别达到 **100.41 ms**、**388.27 ms**。

检查[上游 Scheduler](reference/nanovllm/engine/scheduler.py) 发现：先处理 waiting 中的 Prefill，只要选中 Prefill 就提前返回，本轮不会执行 Decode。上游已有 Chunked Prefill，但连续 chunk 仍可连续占用调度轮次。

## 实现方案

1. **Decode 优先**：先为 running 请求各安排 1 个 token，并预留需要的 KV block。
2. **剩余预算处理 Prefill**：按 FIFO 推进等待请求，允许将长 Prompt 拆成跨步执行的 chunk。例如预算 1024、3 条 Decode，剩余 1021 个 token 用于 Prefill。
3. **分组执行与状态维护**：同一步先执行 Decode，再执行 Prefill；中间 chunk 只推进 KV 状态，最终 chunk 完成后才追加生成 token、转入 running。

| 改动文件 | 职责 |
|---|---|
| [scheduler.py](src/nanovllm/engine/scheduler.py) | 两组请求共享 token / sequence 预算，处理 KV 预留、抢占和 chunk 状态 |
| [schedule_output.py](src/nanovllm/engine/schedule_output.py) | 返回 Decode / Prefill 两组请求与统计信息 |
| [llm_engine.py](src/nanovllm/engine/llm_engine.py) | 依次执行两组请求，分别更新调度状态 |

Decode 保留 CUDA Graph，Prefill 沿用 eager 路径；两组在同一调度步内**顺序执行**，不是 GPU 并发，也没有合并成一次模型 forward。其余上游源码保持一致，两组使用相同算子，无自定义 CUDA 融合扩展。

## 实验结果

**环境与方法**：2026-09-21，RTX 5080 / Qwen3-0.6B / BF16 / PyTorch 2.11.0+cu128。每个进程先预热，按 A/B、B/A 顺序各测 3 次，报告 6 次结果的中位数。完整实验覆盖 3 种预算、5 类负载，共 180 条测量。

预算 **512**，已有 **4 条 Decode**：

| 新增负载 | 原调度最大停顿 | 混合调度最大停顿 | 降幅 |
|---|---:|---:|---:|
| 1 条 × 3000 tokens | 100.41 ms | 27.55 ms | **72.56%** |
| 4 条 × 3000 tokens | 388.27 ms | 39.55 ms | **89.81%** |

最大停顿按完整 `Engine.step()` 返回时间统计，包含新负载注入到下一 token 的首次等待；每轮求最大值，再对六轮取中位数。它不是平均 TPOT，也不是网络客户端延迟。

**收益与代价**：单长 Prompt 场景每轮出现 6 个混合 step，并保留 31 次实际 Graph replay；Decode 能在 Prefill 推进期间持续执行。该场景新请求 TTFT 从 95.85 增至 125.94 ms，输出吞吐从 688.76 降至 654.22 token/s；四长 Prompt 的吞吐从 366.05 降至 314.15 token/s。短输入最大停顿也从 17.30 增至 18.24 ms。优化目标是缓解长输入对已有 Decode 的阻塞，不代表所有指标改善。

**正确性**：15 项测试覆盖预算、FIFO、chunk 边界、EOS、抢占、前缀复用、KV 计数与 GPU 元数据。四组模型对照共检查 64 行 logits，top-1 全部一致，最大相对 L2 为 0.025113、最小余弦相似度为 0.999797，满足预设容差（L2 < 0.03、cosine > 0.999）；不声称逐位一致。

[全部负载对比](results/20260921/comparison.txt) · [原始测量与数值对照](results/20260921/) · [实验环境](results/20260921/environment.json)

## 复现

需要 Linux / WSL2、NVIDIA GPU 和兼容的 CUDA 环境。依赖版本见上方实验环境；安装 FlashAttention 需要匹配本机 PyTorch / CUDA。

```bash
git clone https://github.com/123bawanglong/nano-vllm-mixed-scheduling.git
cd nano-vllm-mixed-scheduling
python -m pip install -e .

export NANOVLLM_MODEL=/path/to/Qwen3-0.6B
export NANOVLLM_RESULTS_DIR=results/local-run
python -m unittest discover -s tests -v
python scripts/run_experiments.py --action all
python scripts/summarize.py "$NANOVLLM_RESULTS_DIR"
```

实验脚本先执行四组模型数值验证，再串行运行性能对照；输入为固定种子的合成 token。模型权重和完整 logits 不随仓库分发，由复现实验生成。

```text
src/nanovllm/        混合调度实现
reference/nanovllm/  固定上游基线，用于公平对照
tests/              调度与执行测试
scripts/            实验运行及统计
results/20260921/   原始 JSON、数值验证及汇总
```

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)，基线版本 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。这是独立实验仓库，沿用 [MIT License](LICENSE) 并保留原作者署名。
