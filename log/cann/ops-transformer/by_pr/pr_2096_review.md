# Code Review: PR #2096

| 属性 | 值 |
|------|------|
| 标题 | FAI路由&PSE导致DDR问题修改 |
| 作者 | yangyibin |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2096](https://gitcode.com/cann/ops-transformer/merge_requests/2096) |
| 审查时间 | 2026-02-27 19:24:55 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | 075576269c77 |
| 发现 | 严重 0 / 一般 0 / 建议 2 |

---

## 变更概述

本 MR 修复 FusedInferAttentionScore (FIA) 算子的两类问题：
- copy_gm_to_ub.h / fia_block_vec_nonquant.h: PSE Shift 拷贝时使用条件化的 `bIdx`（可能为 0）查询 `GetActualSeqLength`，导致 DDR 错误。新增 `actualBIdx` 字段传递真实 batch index。
- fused_infer_attention_score_tiling_v3.cpp: `CheckSpecConditions` 中特化路由未排除 `quantScale2` 场景，导致不支持的路径被错误选中。

涉及 3 个文件，共 7 行新增/修改。

## 审查发现

共发现 2 个问题（严重 0 / 一般 0 / 建议 2）

---

### #1 [建议] 冗余三目运算
- 位置：`attention/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_v3.cpp:1019`
- 规则：2.1.3（冗余代码）
- 置信度：确定

问题代码：
```cpp
bool quantScale2Flag = context->GetOptionalInputTensor(QUANT_SCALE2_INDEX) != nullptr ? true : false;
```

`!= nullptr` 的结果已经是 `bool`，`? true : false` 完全多余。

修复建议：
```cpp
bool quantScale2Flag = context->GetOptionalInputTensor(QUANT_SCALE2_INDEX) != nullptr;
```

---

### #2 [建议] actualBIdx 类型与 bIdx 不一致
- 位置：`attention/common/op_kernel/memcopy/copy_gm_to_ub.h:375`
- 规则：无明确规则，一致性建议
- 置信度：确定

问题代码：
```cpp
uint64_t actualBIdx = 0;
```

同一结构体中 `bIdx` 为 `uint32_t`，而新增的 `actualBIdx` 为 `uint64_t`。两者语义相同（均为 batch index），类型应保持一致。batch index 不会超过 `uint32_t` 范围，`uint64_t` 无额外收益，反而引入不一致性，且增加了 8 字节的结构体大小（原 `bIdx` 只占 4 字节）。

修复建议：
```cpp
uint32_t actualBIdx = 0;
```

---

## 总结

核心逻辑变更正确：`actualBIdx` 修复了 `pseShiftByBatch == false` 时 `GetActualSeqLength` 获取错误 batch 序列长度的 bug；`quantScale2Flag` 条件补充修复了路由缺陷。两个建议级别的改进点不影响功能正确性。
