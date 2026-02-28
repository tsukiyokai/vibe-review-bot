# Code Review: PR #2098

| 属性 | 值 |
|------|------|
| 标题 | fix constraint of ratio between numHeads and numKeyValueHeads |
| 作者 | tangkaidi |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2098](https://gitcode.com/cann/ops-transformer/merge_requests/2098) |
| 审查时间 | 2026-02-27 19:26:33 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | 4f6359069269 |

---

## 变更概述

本 PR 对 3 个 attention 算子族（fused_infer_attention_score、incre_flash_attention、prompt_flash_attention）共 12 个 Markdown 文档进行修改，统一移除 "numHeads 与 numKeyValueHeads 的比值不能大于 64" 这一约束描述。变更涉及 V1-V4 多个版本的 API 文档，其中 fused_infer_attention_score V3 采用按产品线差异化处理——仅删除 Atlas A2 段，保留 Ascend 950 特定约束。

通过产品支持表验证：
- V1/V2/V4 仅支持 Atlas A2 系列，移除通用比值约束合理
- V3 支持 Atlas A2 和 Atlas A3，保留 Ascend 950 约束段是有意为之（V3 doc 中包含的 Ascend 950 约束虽然与其产品支持表不一致，但属于已有问题）
- V5 仅支持 Ascend 950，未纳入此 PR 范围，合理

---

## 审查发现

### 发现 1：代码侧仍保留 ratio <= 64 校验，文档移除后将导致用户困惑

- 位置：`attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp:1748-1750`，`attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_arch38.cpp:677-680`
- 规则：文档与代码一致性
- 置信度：高
- 问题代码：

```cpp
// prompt_flash_attention_tiling.cpp:1748
if (nQ / nKV > 64) {   // G cannot be greater than 64.
    OP_LOGE(contextKeyParams.opName, "numHeads / numKeyValueHeads = %d, cannot be larger than 64", nQ / nKV);
    return false;
}
```

```cpp
// prompt_flash_attention_tiling_arch38.cpp:677
if (nQ / nKV > 64 && (!enablePFAMLA)) {
    OP_LOGE(contextKeyParams.opName, "numHeads / numKeyValueHeads = %d, cannot be larger than 64.", nQ / nKV);
    return false;
}
```

- 分析：文档已移除 prompt_flash_attention 全系列（V1-V4）的比值上限约束，但 C++ tiling 代码中仍存在硬校验。`prompt_flash_attention_tiling.cpp:1748` 是无条件拦截（`nQ / nKV > 64` 直接返回 false），`prompt_flash_attention_tiling_arch38.cpp:677` 是非 MLA 场景拦截。用户依据新文档传入 ratio > 64 的参数时，会在运行时被代码拒绝。

  同时 `prompt_flash_attention_tiling_v2.cpp:866` 也有类似检查（限定在 antiquant/fullquant 场景）。

- 修复建议：请确认代码侧是否已在另一个 PR 中同步修改。如果代码变更尚未合入，建议在 PR 描述中注明依赖关系，或将文档和代码变更合并提交，避免文档与实际行为出现窗口期不一致。

---

### 发现 2：aclnnFusedInferAttentionScoreV4 移除全部比值约束，但 V3 同产品仍保留 Ascend 950 约束段

- 位置：`attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md:633`
- 规则：文档完整性 / 跨版本一致性
- 置信度：中
- 问题代码：

```html
<li>需要满足numHeads整除numKeyValueHeads。</li>
```

- 分析：V4 产品支持表显示仅支持 Atlas A2 和 Atlas A3（不支持 Ascend 950），所以移除全部比值约束在逻辑上正确。但值得注意的是：V3 在同一 PR 中保留了 Ascend 950 的分场景约束（伪量化/全量化 <= 64，MLA decode <= 128 等），而 V3 的产品支持表同样标记 Ascend 950 为不支持。这一矛盾是已有问题，但在本 PR 修改 V3 内容时是一个好的修正机会。

- 修复建议：如果 V3 确实不支持 Ascend 950，考虑一并移除 V3 中残留的 Ascend 950 约束段（lines 840-841），或修正 V3 产品支持表。这不是此 PR 引入的问题，可作为 follow-up 处理。

---

### 发现 3：V3 文档中 "numKeyValueHead" 缺少尾部 's'

- 位置：`attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md:839`
- 规则：命名一致性
- 置信度：高
- 问题代码：

```
需要满足numHeads整除numKeyValueHead。
```

- 分析：所有其他文件均使用 "numKeyValueHeads"（带 's'），V3 第 839 行写作 "numKeyValueHead" 缺少尾部 's'。这是已有 typo，此 PR 修改了相邻行但未触及此处。既然已在编辑此区域，建议一并修正。

- 修复建议：将 "numKeyValueHead" 改为 "numKeyValueHeads"。

---

## 总结

本 PR 的文档变更意图清晰——放开 Atlas A2 产品线上 numHeads/numKeyValueHeads 比值的 64 倍上限约束。12 个文件的修改范围覆盖完整，V5（Ascend 950 专属）合理地排除在外。

核心风险在于发现 1：prompt_flash_attention 的 C++ tiling 代码仍保留 ratio <= 64 的硬拦截。如果代码侧变更未同步合入，用户将面临"文档说可以，代码说不行"的情况。建议作者确认代码侧变更的合入计划。发现 2 和 3 是已有问题，优先级较低，可视情况在此 PR 或后续 PR 中一并处理。
