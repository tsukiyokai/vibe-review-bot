# Code Review: PR #2076

| 属性 | 值 |
|------|------|
| 标题 | cleancode清理 |
| 作者 | ZhouChuping |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2076](https://gitcode.com/cann/ops-transformer/merge_requests/2076) |
| 审查时间 | 2026-02-27 12:34:54 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | 315dfbe5d4be |
| 发现 | 严重 0 / 一般 0 / 建议 1 |

---

## 变更概述

本 MR 为 moe_re_routing 模块的 cleancode 清理，主要变更：
- moe_re_routing_r_tiling.cpp: 移除注释中的 "todo" 前缀
- moe_re_routing_re_tiling.cpp: 同上

涉及 2 个文件，各 1 处注释修改，无逻辑变更。

## 审查发现

共发现 1 个问题（严重 0 / 一般 0 / 建议 1）

---

### #1 [建议] 移除 TODO 后残留注释表意模糊
- 位置：`moe/moe_re_routing/op_host/moe_re_routing_r_tiling.cpp:34`, `moe/moe_re_routing/op_host/moe_re_routing_re_tiling.cpp:35`
- 规则：1.3.3（注释应准确描述代码意图）
- 置信度：确定

问题代码：
```cpp
// tokenSum_ == 0时需要判断
```

原注释 `// todo tokenSum_ == 0时需要判断` 作为 TODO 标记，含义是"此处 tokenSum_ == 0 的处理尚需补充"。移除 "todo" 后，剩余文字 "tokenSum_ == 0时需要判断" 读起来仍像一个待办事项（"需要判断"），但紧接其后的 `if (tokenSum_ == 0)` 已经做了处理。这会让后续读者困惑：到底还需不需要进一步判断？

建议改为准确描述已有逻辑的意图，或在确认逻辑完备后直接删除该注释：

```cpp
// tokenSum_为0时仅使用1个block
if (tokenSum_ == 0) {
    blockNum_ = 1;
}
```

---

## 总结

变更安全，仅涉及注释文字修改，无功能风险。建议顺带澄清残留注释的表述，避免后续维护者误解为未完成事项。
