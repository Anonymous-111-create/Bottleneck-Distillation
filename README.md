# Bottleneck Distillation

这是论文 **Clear the Bottleneck: Learning When Language Agents Are Ready to Act** 的代码实现。代码展示 readiness certificate 如何把前置条件的状态转为动作级蒸馏信号，并与任务结果奖励共同训练策略。

## 安装与运行

需要 Python 3.10+。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
bd-train --updates 40 --seed 0
python -m unittest discover -s tests -v
```

默认任务要求依次 `inspect → verify → commit`。训练命令逐轮打印采样轨迹的成功率和 readiness KL 损失。推理时只调用不带证书的 `LinearPolicy.probabilities(state, actions)`。

## 方法

1. `TaskAdapter.certificates` 用可执行检查器产生包含条件、实体绑定和 `INACTIVE/SAT/UNSAT` 状态的证书表。
2. 对每个 `UNSAT` 条件，仅把该证书改为 `SAT`，教师分别计算事实与反事实下的规范动作分布。这里的规范动作是单个动作标签，因此动作分数直接归一化。
3. `method.py` 计算 `log p_factual - log p_completed`、按学生分布中心化的 RTA、自然对数 JS 散度、top-K 加权 readiness advantage，以及 KL 正则化的目标分布。
4. `train.py` 对同一任务采样多条轨迹，用组内标准化结果奖励更新学生；每个结果更新配套一次 `KL(target || student)` 蒸馏更新。教师在每轮开始时同步学生参数，目标分布停止梯度。默认参数 `K=4, τ=0.7, β=0.05, λ=0.3` 与论文附录 F 一致。

示例采用 NumPy 线性策略。教师在同步参数上增加可解释的证书动作偏置，提供事实/完成证书响应；真实语言模型可替换 `LinearPolicy`，保持 `probabilities(state, actions, certificates)` 接口。

## 接入任务与评测

在 [`environment.py`](src/bottleneck_distillation/environment.py) 中实现 `TaskAdapter` 的 `reset`、`actions`、`certificates`、`step`；用 `feature_fn` 把任务观测映射为固定维度特征，并将动作名传给 `LinearPolicy`。`step` 返回终局归一化奖励。外部评测可直接读取无证书的学生动作分布；真实完成/匹配控制分支及 PCG 审计可基于任务状态快照单独接入。

仓库中的任务与教师仅用于运行算法流程；论文报告的跨基准结果需要对应环境、数据、语言模型及真实完成审计。
