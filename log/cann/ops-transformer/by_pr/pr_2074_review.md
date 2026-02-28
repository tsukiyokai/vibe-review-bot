# Code Review: PR #2074

| 属性 | 值 |
|------|------|
| 标题 | 增加NTD NTD_TND D不等长拦截 |
| 作者 | wld12 |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2074](https://gitcode.com/cann/ops-transformer/merge_requests/2074) |
| 审查时间 | 2026-02-27 12:39:51 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | 01984dea3aac |
| 发现 | 严重 0 / 一般 0 / 建议 1 |

---

## 变更概述

本 MR 为 PromptFlashAttention 算子在 NTD/NTD_TND layout 下增加了 Q/K 与 V head dimension 不等长（`isQKVDDifferent`）的拦截校验。主要变更：
- prompt_flash_attention_tiling_v2.cpp: 在 `CheckNTDLayoutCrossover` 函数中，新增 GQA 场景下 `isQKVDDifferent` 为 true 时的报错拦截，共 5 行新增代码。

## 审查发现

共发现 1 个问题（严重 0 / 一般 0 / 建议 1）

---

### #1 [建议] 错误信息语法不规范

- 位置：`attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp:2735`
- 规则：1.3.x（注释/文本可读性）
- 置信度：确定

问题代码：
```cpp
"In GQA scenario, not support layout %s when query and key headdim is not equal to value headdim."
```

该错误信息语法不自然（"not support" 缺少主语/助动词），与同函数内已有的报错风格不一致。对比同函数其他信息：
- `"When layout is %s, left padding is not supported!"`
- `"In GQA scenario, when layout is NTD, d size of query must be 64 or 128, but got d = %d."`

注意：这条信息来自 BSH/BSND crossover 函数（约 2769 行）的原样复制，原文同样存在此问题。

修复建议：
```cpp
"In GQA scenario, layout %s is not supported when query and key headdim is not equal to value headdim."
```

---

## 总结

代码逻辑正确：格式字符串 `%s` 与 `layoutStr.c_str()` 匹配无误；`isQKVDDifferent` 是类成员变量，作用域合法；GQA 条件（排除所有 MLA/Rope/Quant 场景）与 BSH/BSND crossover 函数中的同类拦截保持一致。仅有一条错误信息措辞的建议，按需处理即可。
