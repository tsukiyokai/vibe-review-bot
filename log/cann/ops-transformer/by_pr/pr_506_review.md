# Code Review: PR #506

| 属性 | 值 |
|------|------|
| 标题 | Support Symmetric Memory |
| 作者 | linzhenkang |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/506](https://gitcode.com/cann/hcomm-dev/merge_requests/506) |
| 审查时间 | 2026-02-24 11:07:41 |
| 审查工具 | Claude Code (`codereview` skill) |
| 发现 | 严重 3 / 一般 1 / 建议 2 |

---

## 变更概述

本 MR 为 HCCL 框架实现了 Symmetric Memory（对称内存）支持，允许在 AICPU 展开场景下通过对称内存窗口直接获取远端节点的地址，绕过 ZeroCopy 的远端地址交换流程。主要变更：
- coll_alg_param.h: `OpParam` 新增 `supportSymmetricMemory`、`inputWindow`/`outputWindow`、`inputOffset`/`outputOffset` 字段
- hccl_communicator_host.cc: 新增 `IsSupportSymmetricMemory` 判断函数，在 `ExecOp` 入口处调用并联动 `supportZeroCopy` 标志
- aicpu_communicator.cc: AICPU 侧在 `PrepareUserMemRanges` 和 `ExecOp` 中增加对称内存分支
- aicpu_operator_pub.h: `OpTilingData` 新增对称内存相关字段用于 host-device 传递
- hccl_aicpu_interface.cc/aicpu_hccl_process.cc: 对称内存标志的传递和恢复

涉及 9 个文件，约 104 行新增/修改。

## 审查发现

共发现 6 个问题（严重 3 / 一般 1 / 建议 2）

---

### #1 [严重] 同一作用域内重复声明变量 `ret`

- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:569`
- 规则: C++ 语言规则（同一作用域变量重定义）
- 置信度: 确定

问题代码:

    HcclResult ret = SymmetricMemory::FindSymmetricWindow(opParam.inputPtr, opParam.inputSize, opParam.inputWindow, opParam.inputOffset);  // 第566行
    ...
    HcclResult ret = SymmetricMemory::FindSymmetricWindow(opParam.outputPtr, opParam.outputSize, opParam.outputWindow, opParam.outOffset);  // 第569行

分析: 第 566 行已经声明了 `HcclResult ret`，第 569 行在同一作用域内再次声明 `HcclResult ret`，编译器会报 "redefinition of 'ret'" 错误，无法通过编译。

修复建议: 第 569 行去掉类型声明，改为赋值：

    ret = SymmetricMemory::FindSymmetricWindow(opParam.outputPtr, opParam.outputSize, opParam.outputWindow, opParam.outputOffset);

---

### #2 [严重] 字段名拼写错误：`outOffset` 应为 `outputOffset`

- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:569`
- 规则: 编译正确性
- 置信度: 确定（已确认 `OpParam` 结构体中字段名为 `outputOffset`，见 `coll_alg_param.h:212`，不存在 `outOffset` 字段）

问题代码:

    SymmetricMemory::FindSymmetricWindow(opParam.outputPtr, opParam.outputSize, opParam.outputWindow, opParam.outOffset);

分析: `OpParam` 结构体中定义的字段名是 `outputOffset`（`coll_alg_param.h` 第 212 行），不存在 `outOffset`。此拼写错误将导致编译失败。

修复建议:

    SymmetricMemory::FindSymmetricWindow(opParam.outputPtr, opParam.outputSize, opParam.outputWindow, opParam.outputOffset);

---

### #3 [严重] `PrepareUserMemRanges` 中使用了未定义的标识符 `opParam`

- 位置: `src/framework/device/framework/aicpu_communicator.cc:332, 343`
- 规则: 编译正确性
- 置信度: 确定（已确认函数签名为 `PrepareUserMemRanges(const OpParam &param, ...)`，参数名是 `param`；`HcclCommAicpu` 类中无 `opParam` 成员变量）

问题代码:

    remoteUserInputBaseAddr = HcclGetSymPtr(opParam.inputWindow, peerRank, opParam.inputOffset);   // 第332行
    remoteUserOutputBaseAddr = HcclGetSymPtr(opParam.outputWindow, peerRank, opParam.outputOffset); // 第343行

分析: `PrepareUserMemRanges` 的函数参数名为 `param`（`const OpParam &param`），但新增代码中使用了 `opParam`。该函数作用域内没有名为 `opParam` 的局部变量，`HcclCommAicpu` 类也没有名为 `opParam` 的成员变量。这将导致 "use of undeclared identifier 'opParam'" 编译错误。

修复建议: 将所有 `opParam.` 替换为 `param.`：

    remoteUserInputBaseAddr = HcclGetSymPtr(param.inputWindow, peerRank, param.inputOffset);
    remoteUserOutputBaseAddr = HcclGetSymPtr(param.outputWindow, peerRank, param.outputOffset);

---

### #4 [一般] 关键字后缺少空格，`else` 前后缺少空格

- 位置: `src/framework/device/framework/aicpu_communicator.cc:325, 327, 328, 329, 2234`
- 规则: 1.2（代码格式规范）
- 置信度: 确定（已确认该文件原有代码中不存在 `}else` 模式，此为新引入的格式违规）

问题代码:

    if(!isSymmetricMemory_) {       // 第325行: if后缺空格
    }else {                          // 第327行: else前后缺空格
    for(size_t peerRank = 0; ...     // 第328行: for后缺空格
    if(peerRank != curRank) {        // 第329行: if后缺空格
    }else {                          // 第2234行: else前后缺空格

修复建议: 按编码规范在关键字后、`else` 前后加空格：

    if (!isSymmetricMemory_) {
    } else {
    for (size_t peerRank = 0; ...
    if (peerRank != curRank) {
    } else {

---

### #5 [建议] `IsSupportSymmetricMemory` 谓词函数具有副作用

- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:543`
- 规则: 命令-查询分离原则
- 置信度: 确定

问题代码:

    bool HcclCommunicator::IsSupportSymmetricMemory(OpParam &opParam)

分析: 函数名 `IsSupport...` 暗示这是一个纯查询（predicate），但实际上它通过 `FindSymmetricWindow` 修改了 `opParam` 的 `inputWindow`、`inputOffset`、`outputWindow`、`outputOffset` 字段。当第一个 `FindSymmetricWindow`（input）成功但第二个（output）失败时，函数返回 `false`，但 `opParam` 中 input 相关字段已被修改。虽然这些残留值在 `supportSymmetricMemory=false` 时不会被使用，但设计上容易误导维护者。对比 `IsSupportZeroCopy(const OpParam &opParam)` 使用的是 `const` 引用，语义更清晰。

修复建议: 考虑将函数拆分为查询和设置两个步骤，或至少重命名为 `TrySetupSymmetricMemory` 以反映其副作用语义。

---

### #6 [建议] 对称内存跳过了 Deterministic 模式检查

- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:3989`
- 规则: 业务逻辑
- 置信度: 待确认（需确认对称内存场景是否确实不受 Deterministic 模式约束）

问题代码:

    opParam.supportZeroCopy = opParam.supportSymmetricMemory || (!commConfig_.GetConfigDeterministic() && IsSupportZeroCopy(opParam));

分析: 原逻辑为 `supportZeroCopy = !Deterministic && IsSupportZeroCopy`，即 Deterministic 模式下明确禁用 ZeroCopy。变更后，当 `supportSymmetricMemory=true` 时，`supportZeroCopy` 直接为 `true`，绕过了 Deterministic 检查。如果 Deterministic 模式的语义是"保证计算结果可复现"，需要确认对称内存路径是否满足此约束。

---

## 总结

本 MR 存在 3 个编译级别的严重缺陷（#1 重复声明 `ret`、#2 `outOffset` 拼写错误、#3 `opParam` 未定义标识符），都是确定性问题，当前代码无法通过编译。建议优先修复这 3 个问题后再提交复审。
