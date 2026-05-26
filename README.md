# 改进 DatawiseAgent

当前改进方向。先做 InfiAgentBench 任务

约束文件：

- `AGENTS.md`：项目级工作规则
- `skills/infiagentbench_error_aware_agent/SKILL.md`：针对 InfiAgentBench 改进方向的专用 skill

开发顺序建议：先做 schema_profile 和 final_block，再做 answer_contract、skill_router、verifier。

新增 InfiAgentBench 改进版入口见：

- `agent_extensions/infiagentbench/`
- `scripts/run_infiagent_improve.py`
- `scripts/eval_infiagent_improve.py`
- `docs/infiagentbench_improvement.md`
