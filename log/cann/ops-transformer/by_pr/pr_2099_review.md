# Code Review: PR #2099

| 属性 | 值 |
|------|------|
| 标题 | feature: infer FA, Ascend950, deprecate api IFAV4/PFAV3 |
| 作者 | leiqingji |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2099](https://gitcode.com/cann/ops-transformer/merge_requests/2099) |
| 审查时间 | 2026-02-27 19:29:23 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | c2b50916b3b1 |
| 发现 | 严重 0 / 一般 1 / 建议 0 |

---

## 变更概述

本 MR 为 IncreFlashAttention (IFA) 和 PromptFlashAttention (PFA) 算子扩展 Ascend950 (DAV_3510) 平台拦截，主要变更：

- aclnn_incre_flash_attention.cpp / _v2.cpp / _v3.cpp: 将错误信息从 "V1 to V3" 更新为 "V1 to V4"
- aclnn_incre_flash_attention_v4.cpp: 新增 Ascend950 平台检查和所需的 3 个头文件 include
- aclnn_prompt_flash_attention.cpp: 将错误信息从 "V1 to V2" 更新为 "V1 to V3"，并将 execute 函数中的平台检查从 deprecation warning 之后移至之前
- aclnn_prompt_flash_attention_v2.cpp: 将错误信息从 "V1 to V2" 更新为 "V1 to V3"
- aclnn_prompt_flash_attention_v3.cpp: 新增 Ascend950 平台检查（两个函数均添加）

涉及 7 个 C++ 文件，共 34 行新增 / 13 行删除。

## 审查发现

共发现 1 个问题（严重 0 / 一般 1 / 建议 0）

---

### #1 [一般] PFA V3 可能缺少 `opdev/op_log.h` 头文件

- 位置：`attention/prompt_flash_attention/op_host/op_api/aclnn_prompt_flash_attention_v3.cpp:55, 73`
- 规则：编译正确性 / include 完整性
- 置信度：待确认

问题代码：

```cpp
OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnPromptFlashAttention versions V1 to V3 are no longer supported on Ascend950.");
```

分析：

新增代码使用了 `OP_LOGE` 宏，该宏通常定义在 `opdev/op_log.h` 中。同系列的 PFA V1 (`aclnn_prompt_flash_attention.cpp:22`) 和 PFA V2 (`aclnn_prompt_flash_attention_v2.cpp:21`) 均显式 include 了 `opdev/op_log.h`，但 V3 没有。V3 文件有 `opdev/op_errno.h`（V1/V2 中不存在），如果该头文件未间接包含 `opdev/op_log.h`，则会导致编译失败。

同样，IFA V4 在添加平台检查时正确地新增了 `opdev/op_log.h`、`opdev/common_types.h`、`opdev/platform.h` 三个 include，而 PFA V3 没有做类似处理。

修复建议：

在 `aclnn_prompt_flash_attention_v3.cpp` 中显式添加 `opdev/op_log.h`（如果 `opdev/op_errno.h` 确实不包含它的话）：

```cpp
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"    // for OP_LOGE
```

需人工确认 `opdev/op_errno.h` 是否已间接包含 `opdev/op_log.h`。若是，则此问题不成立。

---

## 总结

变更逻辑清晰，所有文件的平台检查模式一致，错误信息更新正确。PFA V1 execute 函数中平台检查位置的调整（移至 deprecation warning 之前）是合理的优化。IFA V4 不添加 deprecation warning 是正确的（V4 本身就是推荐迁移的目标版本，PFA V3 同理）。唯一需要确认的是 PFA V3 文件的 `OP_LOGE` 头文件依赖是否通过间接 include 满足。
