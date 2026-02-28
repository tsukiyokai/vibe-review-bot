# Code Review: PR #2097

| 属性 | 值 |
|------|------|
| 标题 | alltoallmatmul A16W4和A16W8性能修复 |
| 作者 | SimpleBright_Man |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2097](https://gitcode.com/cann/ops-transformer/merge_requests/2097) |
| 审查时间 | 2026-02-27 19:31:50 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | 21ea8649ef63 |
| 发现 | 严重 3 / 一般 1 / 建议 1 |

---

## 变更概述

本 MR 为 alltoallmatmul 算子增加 A16W8 和 A16W4 场景的 tiling 规则（NPU 910B），同时重构了 kernel 侧的 CatlassMatmul 和 Quant 逻辑。主要变更：

- allto_all_matmul_tiling_910b.cpp: 新增 6 个 tiling code 查找表（2/4/8 rank x A16W8/A16W4）、6 个对应 tiling 函数、`SetTilingParam`、`DecodeTilingData`，修改 `DoMmCommTiling` 增加 A16W8/A16W4 分支
- allto_all_matmul_tiling_910b.h: 声明新增函数
- allto_all_matmul.h: 重构 `CatlassMatmul` 用编译期类型分发替代重复代码块；重构 `Quant` 函数从 allToAllSendCoreNum 核扩展到全部 AIV 核，改用 `quantCoreNum` 均分 token
- allto_all_matmul_tiling.h: 新增 `quantCoreNum` 字段
- allto_all_matmul_util.h: 新增 tile shape 常量，初始化 `quantCoreNum`

涉及 5 个文件，约 450 行新增 / 77 行删除。

## 审查发现

共发现 5 个问题（严重 3 / 一般 1 / 建议 1）

---

### #1 [严重] A16W8 场景遗漏 rankSize==4 分支

- 位置：`mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910b.cpp:1187-1192`
- 规则：逻辑缺陷
- 置信度：确定

问题代码：
```cpp
// A16W8 tiling策略
if (info.rankSize == 2 && quantType == TILINGKEY_TPL_A16W8) {
    AlltoAllMatmulNPU910BTwoRankA16W8Tiling(cocTilingData, info);
    return ge::GRAPH_SUCCESS;
}
if (info.rankSize == 8 && quantType == TILINGKEY_TPL_A16W8) {
    AlltoAllMatmulNPU910BEightRankA16W8Tiling(cocTilingData, info);
    return ge::GRAPH_SUCCESS;
}
```

分析：A16W8 只处理了 rankSize==2 和 rankSize==8，缺少 rankSize==4 的分支。但 `AlltoAllMatmulNPU910BFourRankA16W8Tiling` 已经实现（line 1101），头文件中也已声明（line 63），对应的查找表 `g_alltoAllMatmulNPU910BFourRankA16W8tilingCodeMap` 也已定义（line 424）。当 A16W8 + 4 卡时，会 fallthrough 到 basic 的 `DoFourRankTiling`，使用 FP16 的 tiling 规则，导致 A16W8 四卡场景的性能优化完全失效。

对比 A16W4 的写法（line 1197-1207），A16W4 完整覆盖了 2/4/8 三个 rank。这是一个遗漏。

修复建议：
```cpp
// A16W8 tiling策略
if (quantType == TILINGKEY_TPL_A16W8) {
    if (info.rankSize == 2) {
        AlltoAllMatmulNPU910BTwoRankA16W8Tiling(cocTilingData, info);
        return ge::GRAPH_SUCCESS;
    } else if (info.rankSize == 4) {
        AlltoAllMatmulNPU910BFourRankA16W8Tiling(cocTilingData, info);
        return ge::GRAPH_SUCCESS;
    } else if (info.rankSize == 8) {
        AlltoAllMatmulNPU910BEightRankA16W8Tiling(cocTilingData, info);
        return ge::GRAPH_SUCCESS;
    }
}
```

---

### #2 [严重] DoMmCommTiling 在未匹配分支时无返回值（未定义行为）

- 位置：`mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910b.cpp:1221`
- 规则：红线 1.4（变量未初始化 / 未定义行为）
- 置信度：确定

问题代码：
```cpp
    } else if (info.rankSize == 8) {
        DoEightRankTiling(cocTilingData, info);  // 若8卡
        return ge::GRAPH_SUCCESS;
    }
}  // 函数结束，无 return
```

分析：重构前，原代码的最后一个分支是无条件的 `DoEightRankTiling` + `return ge::GRAPH_SUCCESS`，保证了所有路径都有返回值。重构后改为 `else if (info.rankSize == 8)`，如果 rankSize 不是 2/4/8 且不是 A16W8/A16W4，函数走到末尾没有 return 语句，返回值未定义。调用方 `GE_ASSERT_GRAPH_SUCCESS(DoMmCommTiling(...))` 会检查返回值，未定义行为可能导致随机失败或继续执行后续逻辑。

修复建议：在函数末尾添加默认返回。
```cpp
    } else if (info.rankSize == 8) {
        DoEightRankTiling(cocTilingData, info);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(opName_, "Unsupported rankSize: %d", info.rankSize);
    return ge::GRAPH_FAILED;
}
```

---

### #3 [严重] Quant 函数中 `globalAivIdx % quantCoreNum - remainTokenNum` 存在无符号下溢风险

- 位置：`mc2/allto_all_matmul/op_kernel/arch32/allto_all_matmul.h:573-574`
- 规则：红线 1.3（整数溢出/翻转）
- 置信度：待确认

问题代码：
```cpp
dataSrcCoreOffset = remainTokenNum * (quantSizePerCore + tokenSize) + (globalAivIdx % quantCoreNum - remainTokenNum) * (quantSizePerCore);
coreTokenOffset = remainTokenNum * (tokenPercore + 1) + (globalAivIdx % quantCoreNum - remainTokenNum) * tokenPercore;
```

分析：`globalAivIdx` 是 `uint32_t` 类型（line 564），`remainTokenNum` 是 `int32_t`。表达式 `globalAivIdx % quantCoreNum - remainTokenNum` 中，由于 `globalAivIdx % quantCoreNum` 为 `uint32_t`，`remainTokenNum` 会被隐式转换为 `uint32_t`。如果 `globalAivIdx % quantCoreNum < remainTokenNum`，减法结果会发生无符号下溢，得到一个非常大的正数。

在当前 910B 架构下，假设 AIC:AIV = 1:2，`globalAivIdx` 范围是 `[0, aivNum-1]`，`quantCoreNum == aivNum`，此时 `globalAivIdx % quantCoreNum == globalAivIdx`，由于 else 分支条件 `globalAivIdx >= remainTokenNum`，减法不会下溢。但如果未来硬件架构中 AIC:AIV 比例不是 1:2，或 `quantCoreNum` 的计算方式变更，这段代码就会出错。需人工确认 `globalAivIdx < quantCoreNum` 是否恒成立。

修复建议：将 `globalAivIdx` 声明为 `int32_t`（与其他偏移量一致），或在 else 分支内加 assert 保护。
```cpp
int32_t globalAivIdx = static_cast<int32_t>(aicIdx * 2 + aivIdx);
```

---

### #4 [一般] 命名规范：局部变量 `TilingParamMap` 应使用小驼峰

- 位置：`mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910b.cpp:1090, 1103, 1116, 1131, 1145, 1159, 1173`
- 规则：1.1.x（命名规范）
- 置信度：确定

问题代码：
```cpp
std::map<int*, AlltoAllMatmulTilingValue> TilingParamMap = {
```

分析：根据编码规范，局部变量应使用小驼峰命名（`tilingParamMap`）。此问题在原有的 `CalTilingParam` 调用方中已经存在，但本 PR 新增了 7 处同名局部变量，扩大了违规范围。此外 `SetTilingParam` 的形参 `TilingParamMap` 也应改为 `tilingParamMap`。

修复建议：将所有 `TilingParamMap` 重命名为 `tilingParamMap`。

---

### #5 [建议] CatlassMatmul 中注释位置错误

- 位置：`mc2/allto_all_matmul/op_kernel/arch32/allto_all_matmul.h:175`
- 规则：1.3.x（注释风格）
- 置信度：确定

问题代码：
```cpp
    // A16W4的tiling
template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CatlassMatmul()
```

分析：注释 `// A16W4的tiling` 放在了 `CatlassMatmul` 函数上方，但 `CatlassMatmul` 是通用的矩阵乘法函数，并非仅服务于 A16W4。这个注释来自原始代码的上下文（前一个函数结束后的空行区域），误放到了此处。该注释的缩进也与函数模板声明不一致（多了一级缩进）。

修复建议：删除该注释或将其移到正确位置。

---

## 总结

最严重的问题是 #1：A16W8 四卡场景的 tiling 函数已实现但未被调用，导致该场景性能优化完全失效——这应该是一个编码遗漏。#2 `DoMmCommTiling` 缺少默认返回值会引入未定义行为。#3 的无符号混合运算虽然在当前硬件假设下不会触发，但属于脆弱设计。建议优先修复 3 个严重问题。
