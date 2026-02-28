# Code Review: PR #2072

| 属性 | 值 |
|------|------|
| 标题 | 新增Inner接口 |
| 作者 | zzzyh22 |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2072](https://gitcode.com/cann/ops-transformer/merge_requests/2072) |
| 审查时间 | 2026-02-27 12:51:38 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | ae2bf8aea330 |
| 发现 | 严重 0 / 一般 2 / 建议 1 |

---

## 变更概述

本 MR 为 LightningIndexer 算子新增 aclnn 外层接口（Inner 封装层），主要变更：
- aclnn_lightning_indexer.cpp: 新增文件，实现 `aclnnLightningIndexerGetWorkspaceSize` 和 `aclnnLightningIndexer` 两个公开接口，通过 `TensorHolder` RAII 类处理可选输出参数 `sparseValuesOut` 的空指针占位，然后透传调用 Inner 接口。
- aclnn_lightning_indexer.h: 新增文件，声明上述两个接口，使用 `__attribute__((visibility("default")))` 导出符号。

涉及 2 个 C++ 文件（均为新增），整体是对已有 mla_prolog_v2/v3 中 TensorHolder 模式的复用。

## 审查发现

共发现 3 个问题（严重 0 / 一般 2 / 建议 1）

---

### #1 [一般] 使用 C 标准头文件 `<string.h>` 而非 C++ 对应物

- 位置：`attention/lightning_indexer/op_host/op_api/aclnn_lightning_indexer.cpp:11`
- 规则：2.2.1
- 置信度：确定

问题代码：
```cpp
#include <string.h>
```

C++ 代码应使用 C++ 标准头文件。仓库中同类文件（如 `aclnn_mla_prolog_v2_weight_nz.cpp:10`、`aclnn_mla_prolog_v3_weight_nz.cpp:10`）均使用 `<cstring>`。

修复建议：
```cpp
#include <cstring>
```

---

### #2 [一般] 缺少 `CheckTensorConditionalNotNull` 校验调用

- 位置：`attention/lightning_indexer/op_host/op_api/aclnn_lightning_indexer.cpp:110`
- 规则：参数校验完整性（参照仓库既有模式）
- 置信度：较确定（已确认 `aclnn_mla_prolog_v2_weight_nz.cpp:121` 和 `aclnn_mla_prolog_v3_weight_nz.cpp:159,162` 中 TensorHolder 创建后均调用了 `CheckTensorConditionalNotNull`）

问题代码：
```cpp
    if (sparseValuesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor sparseValuesOut!");
        return ge::GRAPH_FAILED;
    }
    // 此处缺少 CheckTensorConditionalNotNull 调用，直接进入 Inner 调用
```

`TensorHolder` 提供了 `CheckTensorConditionalNotNull` 方法，用于校验调用者传入的可选参数是否与业务条件一致（例如：当 `returnValues` 为 true 时，`sparseValuesOut` 理应由调用者提供而非由 Holder 占位；反之亦然）。仓库中所有使用 TensorHolder 的位置都在 null check 之后、Inner 调用之前插入了此校验。缺少此调用会导致调用者传参不一致时无法获得任何诊断信息。

修复建议：
```cpp
    if (sparseValuesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor sparseValuesOut!");
        return ge::GRAPH_FAILED;
    }
    sparseValuesOutHolder.CheckTensorConditionalNotNull(returnValues);
```

---

### #3 [建议] 文件末尾缺少换行符

- 位置：`attention/lightning_indexer/op_host/op_api/aclnn_lightning_indexer.cpp:128`
- 规则：POSIX 文本文件规范
- 置信度：确定

问题代码：
```cpp
#endif
\ No newline at end of file
```

POSIX 标准要求文本文件以换行符结尾。缺少尾部换行可能导致部分工具产生警告。

修复建议：在 `#endif` 后添加一个空行（换行符）。

---

## 总结

代码整体结构清晰，正确复用了仓库中 TensorHolder RAII 模式。建议优先处理 #2（补充 `CheckTensorConditionalNotNull` 校验），保持与仓库既有实现的一致性；#1 和 #3 为规范性问题，可一并修复。
