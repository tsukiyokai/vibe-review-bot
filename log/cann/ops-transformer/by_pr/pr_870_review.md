# Code Review: PR #870

| 属性 | 值 |
|------|------|
| 标题 | Support symmetric memory for aicpu unflod mode |
| 作者 | linzhenkang |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/870](https://gitcode.com/cann/hcomm-dev/merge_requests/870) |
| 审查时间 | 2026-02-24 10:49:52 |
| 审查工具 | Claude Code (`codereview` skill) |
| 发现 | 严重 5 / 一般 3 / 建议 2 |

---

## 变更概述

本 PR 为 HCCL 框架新增对称内存（Symmetric Memory）特性，支持 AICPU unfold 模式下的对称内存通信。主要变更：

- include/hccl/hccl_comm.h, hccl_types.h: 新增公共 API `HcclCommWindowRegister`/`HcclCommWindowDeRegister`/`HcclMemAlloc`/`HcclMemFree` 及 `HcclWindow` 类型
- src/algorithm/impl/operator/: 四个算子（allgather, allreduce, broadcast, reduce_scatter）的算法选择逻辑中新增 `supportSymmetricMemory` 分支
- src/framework/communicator/impl/symmetric_memory/: 新增核心类 `SymmetricMemory`，实现对称 VA 空间管理、跨 rank 物理内存映射、窗口注册/注销
- src/framework/common/src/hccl_mem_alloc.cc: 新增 VMM 方式的内存分配/释放
- src/framework/communicator/impl/hccl_communicator_host.cc: 集成对称内存的初始化、判定和 zerocopy 路径协调
- src/framework/device/framework/aicpu_communicator.cc: AICPU 侧对称内存的远端地址准备和执行逻辑
- src/pub_inc/aicpu_operator_pub.h: OpTilingData 新增对称内存相关字段
- src/platform/common/externalinput.cc: 新增环境变量 HCCL_SYMMETRIC_MEMORY_STRIDE 的解析

涉及 34 个 C/C++ 文件，约 1200 行新增/修改。

## 审查发现

共发现 10 个问题（严重 5 / 一般 3 / 建议 2）

---

### #1 [严重] 运算符优先级导致条件逻辑错误
- 位置: `src/algorithm/impl/operator/reduce_scatter_operator.cc:481`
- 规则: 红线 — 逻辑正确性
- 置信度: 较确定（已对比其他三个算子的写法，确认只有 reduce_scatter 不一致）

问题代码:

    } else if (isSupportInlineReduce && param.supportSymmetricMemory || (param.supportZeroCopy &&
        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB))) {

分析:
C++ 中 `&&` 优先级高于 `||`，因此此表达式等价于 `(isSupportInlineReduce && param.supportSymmetricMemory) || (param.supportZeroCopy && ...)`。这意味着当 `isSupportInlineReduce` 为 false 但 `supportZeroCopy` 条件满足时，仍会进入该分支——而注释 `isSupportInlineReduce：不申请scratch ==> 不支持非InlineReduce` 暗示 `isSupportInlineReduce` 应作为整个分支的前提条件。

对比其他三个算子（all_gather, all_reduce, broadcast）的写法都是 `param.supportSymmetricMemory || (param.supportZeroCopy && ...)`，只有 reduce_scatter 额外加了 `isSupportInlineReduce &&` 前缀，但括号位置不对。

修复建议（如果意图是 InlineReduce 约束整个分支）:

    } else if (isSupportInlineReduce && (param.supportSymmetricMemory || (param.supportZeroCopy &&
        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB)))) {

---

### #2 [严重] ExchangePhyAddrHandle 多条路径资源泄漏
- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:570, 571, 572, 574, 575, 576, 581`
- 规则: 红线 1.6（资源泄漏）
- 置信度: 确定

问题代码:

    CHK_RET(hrtMalloc((void**)&sendBuff, inputSize));
    CHK_RET(hrtMalloc((void**)&recvBuff, outputSize));
    CHK_RET(hrtMemSyncCopy((void*)sendBuff, inputSize, inputBuff, inputSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    CHK_RET(HcclAllGatherInner(sendBuff, recvBuff, inputCount, dataType, subCommHandle_, stream_->ptr()));
    CHK_RET(hcclStreamSynchronize(stream_->ptr()));
    CHK_RET(hrtMemSyncCopy((void*)outputBuff, outputSize, (void*)recvBuff, outputSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

分析:
`CHK_RET` 在失败时直接 return，不执行后续清理。第 571 行 `hrtMalloc(&recvBuff)` 失败时 `sendBuff` 泄漏；第 572-576 行任一步失败时 `sendBuff` 和 `recvBuff` 均泄漏。此外第 581 行 `hrtFree(sendBuff)` 失败时 `recvBuff` 也泄漏。

修复建议:
使用 goto cleanup 模式或 RAII 封装，确保所有失败路径释放已分配的资源：

    HcclResult ret = HCCL_SUCCESS;
    CHK_RET(hrtMalloc((void**)&sendBuff, inputSize));
    ret = hrtMalloc((void**)&recvBuff, outputSize);
    if (ret != HCCL_SUCCESS) { goto CLEANUP; }
    // ... 同理后续步骤 ...
    CLEANUP:
    if (sendBuff) { (void)hrtFree(sendBuff); }
    if (recvBuff) { (void)hrtFree(recvBuff); }
    return ret;

---

### #3 [严重] HcclMemFree 中间步骤失败导致资源泄漏
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:75, 77`
- 规则: 红线 1.6（资源泄漏）
- 置信度: 确定

问题代码:

    ret = aclrtUnmapMem(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] UnmapMem virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
    ret = aclrtFreePhysical(handle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] FreePhysical handle[%p] failed, ret[%d]", handle, ret), HCCL_E_RUNTIME);

分析:
释放路径应采用 best-effort 策略。当前第 75 行 `aclrtUnmapMem` 失败直接返回，泄漏了 `handle`（通过 `aclrtMemRetainAllocationHandle` 获取）和虚拟地址（未调用 `aclrtReleaseMemAddress`）。第 77 行 `aclrtFreePhysical` 失败直接返回，虚拟地址泄漏。

修复建议:
释放函数应尽力清理所有资源，即使某步失败也继续后续释放：

    HcclResult result = HCCL_SUCCESS;
    ret = aclrtUnmapMem(ptr);
    if (ret != ACL_SUCCESS) { HCCL_ERROR(...); result = HCCL_E_RUNTIME; }
    ret = aclrtFreePhysical(handle);
    if (ret != ACL_SUCCESS) { HCCL_ERROR(...); result = HCCL_E_RUNTIME; }
    ret = aclrtReleaseMemAddress(ptr);
    if (ret != ACL_SUCCESS) { HCCL_ERROR(...); result = HCCL_E_RUNTIME; }
    return result;

---

### #4 [严重] symmetricMemory_ 未做空指针检查即解引用
- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:602, 605`
- 规则: 红线 1.5（空指针解引用）
- 置信度: 较确定（已确认 `symmetricMemory_` 在 `InitSymmetricMemory` 中创建，但 `IsSupportSymmetricMemory` 无 null 防护。若 Init 失败或在特定异常恢复路径中，`symmetricMemory_` 可能为 nullptr）

问题代码:

    HcclResult ret = symmetricMemory_->FindSymmetricWindow(opParam.inputPtr, opParam.inputSize, &opParam.inputWindow, opParam.inputOffset);

分析:
`IsSupportSymmetricMemory` 函数中直接对 `symmetricMemory_` 解引用，未检查是否为 nullptr。同一类的析构函数中已有 `if(symmetricMemory_ != nullptr)` 的检查，说明开发者意识到它可能为空。同样，`RegisterWindow`（第 8838 行）和 `DeregisterWindow`（第 8844 行）也直接解引用了 `symmetricMemory_`，且这两个是外部 API 入口，用户可能在初始化完成前调用。

修复建议:
在 `IsSupportSymmetricMemory` 解引用前增加检查：

    CHK_PRT_RET(symmetricMemory_ == nullptr,
        HCCL_INFO("[%s] symmetricMemory_ is nullptr", __func__), false);

在 `RegisterWindow` 和 `DeregisterWindow` 中增加：

    CHK_SMART_PTR_NULL(symmetricMemory_);

---

### #5 [严重] HcclGetSymPtr 未校验 winHandle 空指针
- 位置: `src/framework/device/framework/aicpu_symmetric_memory.cc:25`
- 规则: 红线 1.5（空指针解引用）
- 置信度: 确定

问题代码:

    SymmetricWindow *symWin = reinterpret_cast<SymmetricWindow *>(winHandle);
    size_t peerOffset = peerRank * symWin->stride + offset;

分析:
如果 `winHandle` 为 nullptr，`reinterpret_cast` 后 `symWin` 为空，`symWin->stride` 解引用会段错误。调用方（aicpu_communicator.cc:2268-2269）虽然对 `HcclGetSymPtr` 的返回值做了空判断，但函数内部会在判断之前就已经崩溃。

修复建议:

    void *HcclGetSymPtr(HcclWindow winHandle, int32_t peerRank, size_t offset)
    {
        if (winHandle == nullptr) {
            HCCL_ERROR("[HcclGetSymPtr] winHandle is nullptr");
            return nullptr;
        }
        // ...
    }

---

### #6 [一般] 格式字符串类型不匹配: %d 用于 size_t
- 位置: `src/framework/cluster_maintenance/health/heartbeat/heartbeat.cc:581`
- 规则: 3.1.3（格式字符串参数匹配）
- 置信度: 确定

问题代码:

    HCCL_INFO("[Heartbeat][UnRegisterToHeartBeat] groupMap_.size[%d]", groupMap_.size());

分析:
`std::map::size()` 返回 `size_t`（64 位系统上为 `uint64_t`），使用 `%d`（32 位有符号整数）读取。虽然实际 map 大小不太可能超过 INT_MAX，但这是未定义行为，且会触发编译器告警。同文件第 50 行使用了 `%llu` 来打印相同类型的值。

修复建议:

    HCCL_INFO("[Heartbeat][UnRegisterToHeartBeat] groupMap_.size[%zu]", groupMap_.size());

---

### #7 [一般] range-based for 循环应使用 const 引用
- 位置: `src/framework/cluster_maintenance/health/heartbeat/heartbeat.cc:578`
- 规则: 2.10.6（只读形参/变量缺 const）
- 置信度: 确定

问题代码:

    for (const auto it : groupMap_) {

分析:
`groupMap_` 是 `std::map`，按值遍历会拷贝每个 `std::pair<const std::string, ...>`，产生不必要的 string 拷贝开销。

修复建议:

    for (const auto& it : groupMap_) {

---

### #8 [一般] 析构函数中冗余的空指针判断
- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:172, 173`
- 规则: 2.1.3（冗余代码）
- 置信度: 确定

问题代码:

    if(symmetricMemory_ != nullptr){
        symmetricMemory_ = nullptr;
    }

分析:
`symmetricMemory_` 是 `std::unique_ptr`，赋值 nullptr 等价于 reset()，而 unique_ptr 的 reset/析构本身就处理了空指针的情况。这个 `if` 判断完全多余。此外 `if(` 缺少空格。更重要的是，手动置空没有必要，因为紧接着对象析构就会自动释放 unique_ptr。

修复建议:
直接删除这三行。unique_ptr 的析构函数会自动处理。

---

### #9 [建议] 注释中引入了拼写错误
- 位置: `include/hccl/hccl_types.h:67`
- 规则: 1.3.1（注释拼写）
- 置信度: 确定

问题代码:

    * @brief HCCL Reduction opperation

分析:
原文是 `operation`（正确），PR 改成了 `opperation`（多了一个 p）。这是一个无意的 typo。

修复建议:

    * @brief HCCL Reduction operation

---

### #10 [建议] 大量注释掉的代码不应提交
- 位置: `src/platform/common/externalinput.cc:1324-1345`
- 规则: 2.1.3（冗余代码）/ 1.3.3（TODO/FIXME）
- 置信度: 确定

问题代码:

    // std::string symmetricMemoryStrideEnv = GET_ENV(MM_ENV_HCCL_SYMMETRIC_MEM_STRIDE);
    // if (symmetricMemoryStrideEnv == "EmptyString") {
    //     g_externalInput.symmetricMemoryStride = SYMMETRIC_MEM_STRIDE_DEFAULT;
    //     ...
    // }
    // CHK_RET(IsAllDigit(symmetricMemoryStrideEnv.c_str()));
    // HcclResult ret = SalStrToULong(symmetricMemoryStrideEnv, HCCL_BASE_DECIMAL, symmetricMemoryStride);

分析:
`ParseSymmetricMemoryStride` 函数体内大部分代码被注释掉，直接硬编码返回默认值。如果环境变量解析功能未完成，应使用 TODO 注释标记并说明原因；如果确认不需要，应直接删除被注释的代码。当前状态下，尽管 `InitEnvVarParam` 中调用了 `ParseSymmetricMemoryStride` 并做了错误上报（`RPT_ENV_ERR`），但函数永远返回 `HCCL_SUCCESS`，错误处理路径是死代码。

修复建议:
如果是临时跳过，添加 `// TODO: 待实现环境变量解析` 注释说明；如果是设计决策（始终使用默认值），删除被注释的代码和 `InitEnvVarParam` 中对应的 `RPT_ENV_ERR` 错误上报逻辑。

---

## 总结

本 PR 实现了对称内存特性的核心框架，包括 VA 空间管理、跨 rank 物理内存映射、算子路径选择等完整链路。代码结构清晰，关键路径的数据流（host OpParam -> OpTilingData -> device OpParam）已正确覆盖。

主要风险集中在资源管理和防御性编程方面：`ExchangePhyAddrHandle` 和 `HcclMemFree` 的多条失败路径存在资源泄漏（#2, #3），`reduce_scatter_operator.cc` 中的运算符优先级可能导致算法选择错误（#1），以及 `symmetricMemory_` 和 `winHandle` 的空指针解引用风险（#4, #5）。

建议优先处理 5 个严重问题，其中 #1 运算符优先级需要与作者确认意图后确定正确的括号写法，其余 4 个为确定性缺陷。
