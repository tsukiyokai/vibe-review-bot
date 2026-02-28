# Code Review: PR #2070

| 属性 | 值 |
|------|------|
| 标题 | context结构体切换 |
| 作者 | zzg_code |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2070](https://gitcode.com/cann/ops-transformer/merge_requests/2070) |
| 审查时间 | 2026-02-27 13:01:48 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | 9190e56bef92 |
| 发现 | 严重7 / 一般2 / 建议2 |

---

## 变更概述

本MR为MC2 MoE Distribute Dispatch算子引入context结构体切换机制，主要变更：
- 新增 `mc2_moe_context.h`：定义 `Mc2MoeContext` 结构体，封装rank信息和通信buffer地址
- `aclnn_moe_distribute_dispatch_v2_base.cpp`：新增 ~300行代码，包含 `GetCommMode`、`GetHcclCommChannel`、`CreatMc2Context`、`CreatMc2ContextTensor`、`GetMc2Context`、`SetCommArgs` 等函数，实现950平台MTE模式下通过新的extend路径下发算子
- 新增 `moe_distribute_dispatch_v2_extend` 算子目录，包含op_def、tiling_key、kernel骨架
- V2/V3/V4的dispatch入口统一切换为 `aclnnMoeDistributeDispatchBase`，按平台和通信模式路由到V2或V2Extend

涉及20个文件，其中C/C++文件11个。核心逻辑集中在 `aclnn_moe_distribute_dispatch_v2_base.cpp` 的新增代码。

## 审查发现

共发现11个问题（严重7 / 一般2 / 建议2）

---

### #1 [严重] GetCommMode 在所有平台上无条件调用，导致非950平台功能回退

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp:387-388`
- 规则：红线1.4（变量未初始化/误用）+ 功能回退
- 置信度：较确定 — 已确认 `GetCommMode` 内调用 `HcclRankGraphGetLayers`（见同文件:119），该API在非950平台上可能不支持；即使支持，`hcclHandle` 和 `netLayerNum` 在非950路径中从未使用

问题代码：
```cpp
ret = GetCommMode(groupEp, hcclHandle, netLayerNum);
CHECK_RET(ret == ACLNN_SUCCESS, ret);
```

分析：在修改前，`aclnnMoeDistributeDispatchGetWorkspaceSizeBase` 直接调用 `aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize`，不依赖 `HcclRankGraphGetLayers`。修改后，无论是否为950平台，都先调用 `GetCommMode`（其内部调用 `HcclRankGraphGetLayers`），如果该API在910B上返回失败，`CHECK_RET` 会直接返回错误，导致910B上原本正常的功能被阻断。

修复建议：将 `GetCommMode` 调用移到950 MTE分支内部：
```cpp
if(!is950 || (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0)) {
    getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(...);
} else {
    ret = GetCommMode(groupEp, hcclHandle, netLayerNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // ... 950 extend path
}
```

---

### #2 [严重] GetMc2Context 中 mc2_context 未初始化即使用

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp:322`
- 规则：红线1.4（变量未初始化就使用）
- 置信度：确定 — `Mc2MoeContext mc2_context` 为栈上局部变量（line 302），仅在 `HcclEngineCtxGet` 失败时通过 `CreatMc2Context` 初始化（line 316）；当 `HcclEngineCtxGet` 成功时进入else分支（line 317-319），`mc2_context` 所有成员均为未初始化值

问题代码：
```cpp
hcclBuffSize = mc2_context.winsize;
```

分析：当context已存在（else分支），`mc2_context` 从未被赋值。`hcclBuffSize` 会拿到栈上的随机值，传递给后续的 `aclnnInnerMoeDistributeDispatchV2ExtendGetWorkspaceSize` 作为 `hcclBuffSize` 参数，导致不可预期行为。

修复建议：在else分支中从已存在的device context读回 `mc2_context` 数据，或至少从 `HcclGetHcclBuffer` 获取 `winsize`：
```cpp
} else {
    OP_LOGD("PRINT in else");
    // 从已有context中恢复必要信息
    ret = HcclGetRankId(hcclHandle, &mc2_context.rankId);
    // ... 或用 HcclEngineCtxCopy 反向读取
}
```

---

### #3 [严重] Tensor shape 计算错误，仅覆盖 1/4 的 context 数据

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp:284`
- 规则：红线1.2（数组/缓冲区越界的变体：数据截断）
- 置信度：确定 — `sizeof(Mc2MoeContext)` = 8216字节（已确认结构体布局：4+4+8+8+8192，见 mc2_moe_context.h:10-16），除以 `sizeof(uint32_t)` 得2054；但tensor类型为 `ACL_INT8`（1字节/元素），所以tensor仅描述2054字节，实际数据8216字节

问题代码：
```cpp
int64_t shap[1] = {mc2ContextLength / sizeof(uint32_t)}; // 默认1维
```

分析：op_def中 `mc2_context` 输入的数据类型为 `DT_INT8`（见 `moe_distribute_dispatch_v2_extend_def.cpp` 第37行）。tensor的shape应为字节总数，即 `mc2ContextLength`，而不是除以4。当前代码导致kernel侧只能看到前2054字节（约25%），`windowsIn` 数组的大部分数据被截断，通信地址不完整。

修复建议：
```cpp
int64_t shap[1] = {static_cast<int64_t>(mc2ContextLength)};
```

---

### #4 [严重] CHECK_RET 使用了错误的返回值变量

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp:329`
- 规则：3.1.2（内存安全/逻辑错误）
- 置信度：确定 — `res` 为 `aclnnStatus` 类型（line 304），是 `CreatMc2ContextTensor` 的返回值；`ret` 为 `HcclResult` 类型（line 303），是之前 `HcclEngineCtxGet` 的返回值。当 `CreatMc2ContextTensor` 失败时，返回的是 `ret`（HcclResult类型）而非正确的错误码 `res`

问题代码：
```cpp
CHECK_RET(res == ACLNN_SUCCESS, ret);
```

修复建议：
```cpp
CHECK_RET(res == ACLNN_SUCCESS, res);
```

---

### #5 [严重] windowsIn 数组越界风险：rankDim 无上界校验

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp:246, 262`
- 规则：红线1.2（数组越界）
- 置信度：较确定 — `windowsIn` 数组大小为 `HCCL_HOST_KFC_MAX_RANK_NUM = 1024`（见 mc2_moe_context.h:6,15），`rankDim` 从 `HcclGetRankSize` 获取（见同文件:234），取决于集群配置。当 `rankDim > 1024` 时，line 262 的 `mc2_context->windowsIn[index]` 发生栈缓冲区溢出

问题代码：
```cpp
for(uint64_t index = 0; index < mc2_context->rankDim; index++) {
    // ...
    mc2_context->windowsIn[index] = reinterpret_cast<uint64_t>(tempBuffer);
}
```

修复建议：在循环前校验 `rankDim`：
```cpp
if (mc2_context->rankDim > Mc2Kernel::HCCL_HOST_KFC_MAX_RANK_NUM) {
    OP_LOGE(ACLNN_ERR_INNER, "rankDim %u exceeds max %u",
            mc2_context->rankDim, Mc2Kernel::HCCL_HOST_KFC_MAX_RANK_NUM);
    return ACLNN_ERR_INNER;
}
```

---

### #6 [严重] 使用未初始化指针 links 的值打印日志

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp:141`
- 规则：红线1.4（变量未初始化就使用）
- 置信度：确定 — `links` 在 line 133 声明为 `CommLink * links;`，未赋初值。line 141 的 `OP_LOGD` 读取未初始化指针的值，属于未定义行为

问题代码：
```cpp
OP_LOGD("PRINT CommLink ptr %p", links);
```

修复建议：初始化指针或删除此调试日志：
```cpp
CommLink * links = nullptr;
```

---

### #7 [严重] 格式字符串 %d 与 uint64_t 类型不匹配

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp:226, 309, 312`
- 规则：3.1.3（格式字符串参数匹配）
- 置信度：确定 — `ctxSize` 在 line 211 声明为 `uint64_t`，在 line 307 也声明为 `uint64_t`；`%d` 期望 `int` 类型参数

问题代码：
```cpp
OP_LOGD("PRINT ctxSize: %d", ctxSize);     // line 226
OP_LOGD("PRINT ctxSize:%d",ctxSize);        // line 309
OP_LOGD("PRINT ctxSize after:%d", ctxSize); // line 312
```

修复建议：使用 `%lu` 或 `%llu`：
```cpp
OP_LOGD("PRINT ctxSize: %lu", ctxSize);
```

---

### #8 [一般] NnopbaseSetUserHandle / NnopbaseGetUserHandle 未声明为 weak 符号

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp:61-62`
- 规则：2.1.3（冗余/不一致代码）
- 置信度：待确认 — `NnopbaseSetHcclServerType` 使用了 `__attribute__((weak))`（见同文件:60）且调用前有非空检查；新增的 `NnopbaseSetUserHandle` 和 `NnopbaseGetUserHandle` 没有 weak 属性，如果该符号来自可选链接库，在不支持的平台上会导致链接失败。需确认该库是否在所有构建配置中都链接

问题代码：
```cpp
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void* NnopbaseGetUserHandle(void *executor);
```

修复建议：与 `NnopbaseSetHcclServerType` 保持一致，添加 weak 属性并在调用前检查：
```cpp
extern "C" void __attribute__((weak)) NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void* __attribute__((weak)) NnopbaseGetUserHandle(void *executor);
```

---

### #9 [一般] 使用 C 头文件 `<stdint.h>` 而非 C++ 头文件

- 位置：`mc2/common/inc/mc2_moe_context.h:3`
- 规则：2.2.1（C++ 代码中应使用 C++ 标准头文件）
- 置信度：确定

问题代码：
```cpp
#include <stdint.h>
```

修复建议：
```cpp
#include <cstdint>
```

---

### #10 [建议] 大量调试用 PRINT 日志残留在生产代码中

- 位置：`mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2_base.cpp` 多处（108, 116, 117, 137-140, 160-161, 171, 174, 177, 187, 190, 193, 210, 224-225, 232, 238, 242, 248, 263, 270, 276, 292, 300-301, 308-309, 311-313, 318, 320, 329-330）
- 规则：1.3.3（TODO/FIXME 及调试代码清理）
- 置信度：确定 — 超过30处 `OP_LOGD("PRINT ...")` 日志，带有 `"PRINT"` 前缀和非正式描述（如 "inter to the"），属于开发调试遗留

修复建议：在合入前清理所有 `PRINT` 前缀的调试日志，保留必要的关键路径日志并使用规范的日志描述。

---

### #11 [建议] run.sh 调试脚本提交到仓库

- 位置：`run.sh`（新增文件，57行）
- 规则：1.3.3（冗余代码/调试遗留）
- 置信度：确定 — 该文件为本地编译调试脚本，不应提交到仓库

修复建议：将 `run.sh` 从提交中移除，如有需要可加入 `.gitignore`。

---

## 总结

本MR引入了950平台MTE模式的context结构体切换机制，整体架构合理，但实现中存在多个严重缺陷。最关键的问题是：(1) `GetCommMode` 无条件调用导致非950平台功能回退；(2) context已存在时 `mc2_context` 未初始化即读取 `winsize`；(3) tensor shape计算错误导致仅传递1/4的context数据到kernel侧。建议优先处理7个严重问题，其中6个确定、1个较确定（数组越界取决于rankDim运行时值，但缺少防御性校验）。
