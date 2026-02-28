# Code Review: PR #2071

| 属性 | 值 |
|------|------|
| 标题 | attentionupdate sp放大到32 |
| 作者 | skyqwzw |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2071](https://gitcode.com/cann/ops-transformer/merge_requests/2071) |
| 审查时间 | 2026-02-27 12:35:26 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | 2dce22e870d1 |
| 发现 | 严重 1 / 一般 0 / 建议 0 |

---

## 变更概述

本 MR 将 attention_update 算子的 SP（Sequence Parallelism）上限从 16 放大到 32，变更集中在 `aclnn_attention_update.cpp` 的常量定义，仅修改 1 行。

- aclnn_attention_update.cpp: 将 `SP_MAX` 从 16 改为 32

涉及 1 个文件，1 处修改。

## 审查发现

共发现 1 个问题（严重 1 / 一般 0 / 建议 0）

---

### #1 [严重] 跨文件遗漏：tiling 层 ATTR_SP_MAX 未同步修改，SP 17~32 会被 tiling 拦截

- 位置：`attention/attention_update/op_host/attention_update_tiling.cpp:26, 175-176`
- 规则：高价值缺陷模式 #5（跨文件遗漏清理）
- 置信度：确定（已读取 tiling 文件确认 `ATTR_SP_MAX = 16` 且在第 175 行用于校验）

问题代码：
```cpp
// tiling.cpp:26
constexpr uint64_t ATTR_SP_MAX = 16;

// tiling.cpp:175-176
OP_CHECK_IF(!(sp_ >= 1 && sp_ <= ATTR_SP_MAX),
            OP_LOGE(context_->GetNodeName(), "Sp need in [1,16],but got %lu", sp_), return ge::GRAPH_FAILED);
```

分析：API 层的 `SP_MAX` 已改为 32，但 tiling 层的 `ATTR_SP_MAX` 仍为 16。算子执行流程是 API 校验 -> tiling 校验 -> kernel 执行。当用户传入 SP=17~32 时，API 层校验通过，但到 tiling 层会被拦截并报错 `"Sp need in [1,16]"`。这意味着本 MR 的修改实际上不生效，SP 仍然无法超过 16。

修复建议：
```cpp
// tiling.cpp:26
constexpr uint64_t ATTR_SP_MAX = 32;

// tiling.cpp:176 —— 错误信息也需同步更新
OP_LOGE(context_->GetNodeName(), "Sp need in [1,%lu],but got %lu", ATTR_SP_MAX, sp_), return ge::GRAPH_FAILED);
```

---

## 总结

本 MR 意图明确，但存在 1 个严重的跨文件遗漏：tiling 层的 `ATTR_SP_MAX` 未同步放大到 32，导致 SP>16 的场景仍会被拦截，修改实质上不生效。建议补充 `attention_update_tiling.cpp` 的修改后再合入。
