# Code Review: PR #2068

| 属性 | 值 |
|------|------|
| 标题 | [mc2] allgather adjust x1 size |
| 作者 | chuguowei |
| 链接 | [https://gitcode.com/cann/ops-transformer/merge_requests/2068](https://gitcode.com/cann/ops-transformer/merge_requests/2068) |
| 审查时间 | 2026-02-27 12:43:31 |
| 审查工具 | Claude Code (`codereview` skill) |
| 基线提交 | ebae8e3ce547 |
| 发现 | 严重 4 / 一般 0 / 建议 1 |

---

## 变更概述

本 MR 为 mc2 AllGather MatMul V2 的 arch35 tiling 逻辑新增了 x1 尺寸校验（通信限制：最多 16 次通信，单次不超 256MB），同时在 fit_balance_tiling 中硬编码了 16 等分切分策略。主要变更：
- all_gather_quant_bmm_tiling.cpp/h: 新增 `CheckX1Size()` 函数，在 `DoOpTiling()` 入口处调用
- all_gather_fit_balance_tiling.cpp: 在 `AdjustLongShortTileLen()` 中用硬编码 16 等分覆盖了原有自适应切分结果
涉及 3 个文件，约 30 行新增。

## 审查发现

共发现 5 个问题（严重 4 / 一般 0 / 建议 1）

---

### #1 [严重] 调试代码遗留：printf 含开发者姓名，且硬编码覆盖自适应切分结果
- 位置：`mc2/all_gather_matmul_v2/op_host/op_tiling/arch35/all_gather_fit_balance_tiling.cpp:83-88`
- 规则：项目级规则（禁止 printf/cout，必须用项目日志宏）+ 2.1.3（冗余/调试代码）
- 置信度：确定

问题代码：
```cpp
    printf("*********** CHUGUOWEI adjust to 16 part.");
    uint64_t value1 = 16;
    tilingM_.cutRes.longTileLen = Ops::Base::CeilDiv(mmInfo_.mValue, value1);
    tilingM_.cutRes.numLongTile = 15U;
    tilingM_.cutRes.shortTileLen = mmInfo_.mValue - tilingM_.cutRes.longTileLen * 15;
    tilingM_.cutRes.numShortTile = 1U;
```

分析：
1. `printf` 包含开发者姓名 "CHUGUOWEI"，是典型的本地调试打印，不应提交到代码仓库。项目要求使用 `OPS_LOG_D` / `OP_LOGE` 等宏。
2. 这段代码无条件覆盖了上一行 `FitTileLengthDiscrete(false, true, true)` 的计算结果，等同于废弃了自适应切分算法，对所有输入强制 16 等分。这看起来是开发调试时的临时代码，不应合入。
3. 该算术本身也有 uint64_t 下溢风险（见 #2）。

修复建议：整段删除。如果确实需要 16 等分逻辑，应有条件守护并使用命名常量，且不能包含 printf。

---

### #2 [严重] uint64_t 减法下溢：shortTileLen 计算在多数 M 值下产生回绕
- 位置：`mc2/all_gather_matmul_v2/op_host/op_tiling/arch35/all_gather_fit_balance_tiling.cpp:87`
- 规则：红线 1.3（整数溢出/翻转）
- 置信度：确定

问题代码：
```cpp
    tilingM_.cutRes.shortTileLen = mmInfo_.mValue - tilingM_.cutRes.longTileLen * 15;
```

分析：`longTileLen = CeilDiv(M, 16)`，即 ceil(M/16)。当 M 不是 16 的整数倍时，`ceil(M/16) * 15` 很可能大于 M，导致 uint64_t 减法回绕为极大值。例如 M=17 时，longTileLen=2，2×15=30 > 17，shortTileLen 回绕为 2^64 - 13。类似地 M=33, 49, 65... 等大量值都会触发。

修复建议：如果保留此逻辑（不建议，见 #1），应改用 `mValue - longTileLen * (value1 - 1)` 并增加断言确保不下溢，或改为 `mValue % longTileLen` 等等价安全写法。

---

### #3 [严重] sizeof(args_.geAType) 取的是枚举对象大小，非数据元素大小
- 位置：`mc2/all_gather_matmul_v2/op_host/op_tiling/arch35/all_gather_quant_bmm_tiling.cpp:429`
- 规则：高价值缺陷模式 #1（sizeof 误用）
- 置信度：较确定 — 已确认 `geAType` 类型为 `ge::DataType`（枚举），定义见 `matmul_formulaic_tiling.h:185`

问题代码：
```cpp
    uint64_t sizeOfA = args_.orgMValue * args_.orgMValue * sizeof(args_.geAType);
```

分析：`args_.geAType` 的类型是 `ge::DataType`（枚举），`sizeof(args_.geAType)` 得到的是枚举对象本身的字节数（通常为 4），而非该枚举所表示的数据类型的元素大小。例如 FP8 的元素大小应为 1 字节，FP16 为 2 字节，但 sizeof 始终返回 4。项目中已有正确的方式获取元素大小：`args_.inputDtypeSize`（在 `all_gather_matmul_tiling_base.cpp:306` 通过 `mc2tiling::GetDataTypeSize()` 赋值）。

修复建议：
```cpp
    uint64_t sizeOfA = args_.orgMValue * args_.orgKValue * args_.inputDtypeSize;
```

---

### #4 [严重] x1 尺寸计算使用 M×M 而非 M×K
- 位置：`mc2/all_gather_matmul_v2/op_host/op_tiling/arch35/all_gather_quant_bmm_tiling.cpp:429`
- 规则：逻辑错误
- 置信度：较确定 — 已确认 `orgMValue` 对应 x1 的 dim0（M 维），`orgKValue` 对应 x1 的 dim1（K 维），分别在 `all_gather_matmul_tiling_base.cpp:281,283` 赋值

问题代码：
```cpp
    uint64_t sizeOfA = args_.orgMValue * args_.orgMValue * sizeof(args_.geAType);
```

分析：函数名为 `CheckX1Size()`，注释说明是检查 x1 的数据量。x1 的 shape 为 [M, K]，数据量应为 M × K × elementSize。但代码写成了 `orgMValue * orgMValue`（M × M），遗漏了 K 维度。这会导致：
- 当 K > M 时，实际数据量被低估，本应拦截的超限场景被放行
- 当 K < M 时，实际数据量被高估，合法场景被误拦截

修复建议：
```cpp
    uint64_t sizeOfA = args_.orgMValue * args_.orgKValue * args_.inputDtypeSize;
```

---

### #5 [建议] 魔鬼数字 16 使用无语义临时变量 value1
- 位置：`mc2/all_gather_matmul_v2/op_host/op_tiling/arch35/all_gather_quant_bmm_tiling.cpp:430`, `mc2/all_gather_matmul_v2/op_host/op_tiling/arch35/all_gather_fit_balance_tiling.cpp:84`
- 规则：2.4.2（魔鬼数字）
- 置信度：确定

问题代码：
```cpp
    uint64_t value1 = 16;
```

分析：两个文件中都用 `value1` 命名一个含义为"最大通信切分次数"的常量。`value1` 完全没有语义，且在 quant_bmm_tiling.cpp 中已经有了用 `constexpr` 定义命名常量的实践（如 `MAX_BYTES_256MB`）。

修复建议：定义为共享常量或至少使用有意义的名字：
```cpp
constexpr uint64_t MAX_COMM_SPLIT_NUM = 16;
```

---

## 总结

本 MR 存在 4 个严重问题。fit_balance_tiling.cpp 中的整段修改是遗留的调试代码（含 printf 和开发者姓名），应整体移除。CheckX1Size() 的核心计算有两个独立的严重错误：sizeof 枚举对象（应使用 inputDtypeSize）和 M×M 维度错误（应为 M×K），叠加后使校验完全失效。建议先修复这 4 个问题再合入。
