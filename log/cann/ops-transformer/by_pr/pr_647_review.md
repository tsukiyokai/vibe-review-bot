# Code Review: PR #647

| 属性 | 值 |
|------|------|
| 标题 | Support symmetric memory for aicpu unflod mode |
| 作者 | linzhenkang |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/647](https://gitcode.com/cann/hcomm-dev/merge_requests/647) |
| 审查时间 | 2026-02-24 10:58:37 |
| 审查工具 | Claude Code (`codereview` skill) |
| 发现 | 严重 8 / 一般 3 / 建议 1 |

---

## 变更概述

本 MR 为 HCCL 框架新增对称内存（Symmetric Memory）支持，主要面向 AICPU unfold 模式。通过在多个 rank 之间建立共享的虚拟地址空间映射，使得各 rank 可直接访问对等 rank 的内存，替代传统的零拷贝方案。

主要变更：
- `symmetric_memory.cc/h`：新增 SymmetricMemory 管理器，含 VA 空间分配器、窗口注册/查找、跨 rank handle 交换
- `hccl_mem_alloc.cc/h`：新增 HcclMemAlloc/HcclMemFree 公共 API，基于 ACL VMM 接口实现虚拟内存分配
- `aicpu_symmetric_memory.cc/h`：device 侧对称内存指针计算
- `hccl_communicator_host.cc`：集成 symmetric memory 到算子执行流程（IsSupportSymmetricMemory、InitSymmetricMemory、RegisterWindow）
- `aicpu_communicator.cc`：AICPU 侧 symmetric memory 远程地址准备和本地地址替换
- 算法选择（allgather/allreduce/broadcast/reduce_scatter）：symmetric memory 复用 zero copy 算法路径
- `aicpu_operator_pub.h`：OpTilingData 新增 symmetric memory 相关 tiling 字段
- `externalinput.cc/h`：新增 HCCL_SYMMETRIC_MEMORY_STRIDE 环境变量解析（当前为硬编码默认值）

涉及 35 个 C/C++ 文件，含 6 个新文件。

## 审查发现

共发现 12 个问题（严重 8 / 一般 3 / 建议 1）

---

### #1 [严重] SymmetricWindow.stride 类型截断导致对称内存寻址完全失效

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.h:42`, `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:401`
- 规则: 红线 1.3（整数溢出/翻转）
- 置信度: 确定

分析:

`SymmetricWindow::stride` 声明为 `u32`（最大 4GB），但 `SymmetricMemory` 类的 `stride_` 成员是 `size_t`（64 位）。`InitSymmetricMemory` 中计算 stride 值为 `GetExternalInputSymmetricMemoryStride() * 1024 * 1024 * 1024`，默认 `SYMMETRIC_MEM_STRIDE_DEFAULT = 16`，即 16GB = 17179869184 = 0x400000000。赋值 `pWin->stride = stride_` 时发生静默截断：0x400000000 截断为 u32 后等于 0。

后果：`HcclGetSymPtr` 中 `(peerRank - localRank) * symWin->stride` 恒为 0，所有 rank 计算出相同的 VA 地址，对称内存寻址完全错误。

问题代码:

    // symmetric_memory.h:42
    u32 stride;

    // symmetric_memory.cc:401
    pWin->stride = stride_;  // size_t → u32 截断，16GB → 0

修复建议:

    // symmetric_memory.h
    size_t stride;  // 与 SymmetricMemory::stride_ 类型保持一致

---

### #2 [严重] HcclGetSymPtr 有符号/无符号混合运算导致 peerRank < localRank 时地址计算错误

- 位置: `src/framework/device/framework/aicpu_symmetric_memory.cc:27`
- 规则: 红线 1.3（整数溢出/翻转）
- 置信度: 确定

分析:

`peerRank` 类型为 `int32_t`（有符号），`symWin->localRank` 类型为 `u32`（无符号）。C++ 隐式转换规则将 `peerRank` 提升为 `u32` 后再做减法。当 `peerRank < localRank` 时（如 peerRank=0, localRank=1），`0U - 1U = UINT32_MAX`，乘以 stride 后得到错误地址。

此外，即使 #1 修复后 stride 变为 `size_t`，`(u32 * size_t)` 的乘法结果虽能正确扩展到 64 位，但减法的无符号下溢问题仍然存在。

问题代码:

    size_t peerOffset = symWin->heapOffset + (peerRank - symWin->localRank) * symWin->stride + offset;

修复建议:

    // 使用有符号中间变量或显式转换
    ptrdiff_t rankDiff = static_cast<ptrdiff_t>(peerRank) - static_cast<ptrdiff_t>(symWin->localRank);
    size_t peerOffset = symWin->heapOffset + rankDiff * static_cast<ptrdiff_t>(symWin->stride) + offset;

---

### #3 [严重] 全局变量 va2Handle_ 无线程安全保护

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:20`
- 规则: 红线 1.7（并发安全 / data race）
- 置信度: 确定

分析:

`va2Handle_` 是一个全局 `std::unordered_map`，被 `HcclAddVa2Handle`、`aclMemRetainHandle`、`HcclDelVa2Handle` 三个函数读写。`HcclMemAlloc`/`HcclMemFree` 是公共 API，可能被多个线程并发调用。`std::unordered_map` 不是线程安全的，并发读写构成 data race，属于未定义行为。

问题代码:

    std::unordered_map<void*, void*> va2Handle_{};

修复建议:

    // 方案1: 添加互斥锁
    static std::mutex va2HandleMutex_;
    static std::unordered_map<void*, void*> va2Handle_;
    // 在每个访问函数中 lock_guard

---

### #4 [严重] ExchangePhyAddrHandle 中间步骤失败导致设备内存泄漏

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:549, 551, 552, 553`
- 规则: 红线 1.6（资源泄漏）
- 置信度: 确定

分析:

`sendBuff` 和 `recvBuff` 通过 `hrtMalloc` 分配设备内存。`CHK_RET` 宏在失败时直接 return，不会执行后续的 `hrtFree` 调用。若 `hrtMemSyncCopy`（549行）、`HcclAllGatherInner`（551行）、`hcclStreamSynchronize`（552行）任一失败，两块设备内存均泄漏。此外 `CHK_RET(hrtFree(sendBuff))`（557行）失败时 `recvBuff` 也会泄漏。

问题代码:

    CHK_RET(hrtMalloc((void**)&sendBuff, inputSize));
    CHK_RET(hrtMalloc((void**)&recvBuff, outputSize));
    CHK_RET(hrtMemSyncCopy(...));   // 失败则 sendBuff、recvBuff 泄漏
    CHK_RET(HcclAllGatherInner(...));
    CHK_RET(hcclStreamSynchronize(...));
    CHK_RET(hrtMemSyncCopy(...));

修复建议:

    // 使用 RAII 或 goto 清理模式
    HcclResult ret = hrtMemSyncCopy(...);
    if (ret != HCCL_SUCCESS) {
        (void)hrtFree(sendBuff);
        (void)hrtFree(recvBuff);
        return ret;
    }
    // ... 类似处理后续调用

---

### #5 [严重] new(std::nothrow) 返回值未检查，空指针直接解引用

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:392`
- 规则: 2.16.1（内存分配后未判空）
- 置信度: 确定

分析:

`new (std::nothrow) SymmetricWindow()` 在内存不足时返回 `nullptr`。`std::shared_ptr` 构造后内部指针为 null，但紧接着的 `pWin->userVa = ptr` 等赋值直接解引用该空指针。

问题代码:

    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    pWin->userVa = ptr;  // 若 new 返回 nullptr，此处崩溃

修复建议:

    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    CHK_SMART_PTR_NULL(pWin);
    pWin->userVa = ptr;

---

### #6 [严重] OpTilingData 新增字段 inputWindow/outputWindow 未初始化

- 位置: `src/pub_inc/aicpu_operator_pub.h:705, 707`
- 规则: 红线 1.4（变量未初始化）
- 置信度: 确定

分析:

`inputWindow` 和 `outputWindow` 声明为 `u64` 但缺少默认初始值，同组的其他字段（`isSymmetricMemory`、`inputOffset`、`outputOffset`）都有 `= 0` 初始化。`OpTilingData` 作为 host-device 间传递的 tiling 数据结构，未初始化字段携带栈上随机值，device 侧会将其 `reinterpret_cast<void*>` 作为地址使用。

问题代码:

    u64 inputWindow;
    u64 outputWindow;

修复建议:

    u64 inputWindow = 0;
    u64 outputWindow = 0;

---

### #7 [严重] RegisterWindow/DeregisterWindow 未检查 symmetricMemory_ 空指针

- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:8515, 8520`
- 规则: 红线 1.5（空指针解引用）
- 置信度: 较确定。已确认 `std::make_unique` 可能抛异常（非 nothrow），若异常被上层捕获则 `symmetricMemory_` 保持 `nullptr`；此外用户可能在 communicator 初始化完成前调用 RegisterWindow API。

分析:

`RegisterWindow` 和 `DeregisterWindow` 直接调用 `symmetricMemory_->` 成员方法，没有空指针检查。而 device 侧实现（`hccl_communicator_device.cc`）是空操作（直接 return SUCCESS），不存在此问题。

问题代码:

    HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags)
    {
        return symmetricMemory_->RegisterSymmetricMem(ptr, size, winHandle);
    }

修复建议:

    HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags)
    {
        CHK_SMART_PTR_NULL(symmetricMemory_);
        return symmetricMemory_->RegisterSymmetricMem(ptr, size, winHandle);
    }

---

### #8 [严重] HcclMemAlloc 中 HcclAddVa2Handle 失败后已映射内存和物理内存未清理

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:112`
- 规则: 红线 1.6（资源泄漏）
- 置信度: 确定

分析:

在 `HcclMemAlloc` 末尾，`aclrtMapMem` 成功后调用 `CHK_RET(HcclAddVa2Handle(virPtr, handle))`。`CHK_RET` 在失败时直接 return，此时 virPtr 已映射（aclrtMapMem）、handle 已分配（aclrtMallocPhysical）、虚拟地址已预留（aclrtReserveMemAddress），三者均未清理。

问题代码:

    CHK_RET(HcclAddVa2Handle(virPtr, handle));

修复建议:

    HcclResult addRet = HcclAddVa2Handle(virPtr, handle);
    if (addRet != HCCL_SUCCESS) {
        aclrtUnmapMem(virPtr);
        aclrtFreePhysical(handle);
        aclrtReleaseMemAddress(virPtr);
        return addRet;
    }

---

### #9 [一般] 多处格式字符串与参数类型不匹配

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:89, 91, 95`, `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:240, 266, 375, 380`
- 规则: 3.1.3（格式字符串参数匹配）
- 置信度: 确定

分析:

`size_t` 类型应使用 `%zu`，不应使用 `%llu`（unsigned long long）或 `%u`（unsigned int）或 `%ld`（signed long）。在 LP64 模型下 `size_t` 为 `unsigned long`（8 字节），`%llu` 对应 `unsigned long long`，二者在某些平台上宽度一致但类型不匹配，属于未定义行为。

问题代码:

    // hccl_mem_alloc.cc:89 — size_t 用 %llu
    HCCL_ERROR("...granularity[%llu], ret[%d]", granularity, ret);
    // hccl_mem_alloc.cc:95 — void** 用 %p（应为 *ptr 或 virPtr）
    HCCL_ERROR("...virPtr[%p] size[%llu]...", ptr, allocSize, ret);
    // symmetric_memory.cc:240 — size_t 用 %u
    HCCL_ERROR("...Stride %u is not a multiple of granularity %zu.", stride_, granularity_);
    // symmetric_memory.cc:266 — size_t 用 %u
    HCCL_ERROR("...HcclMemAlloc failed for size[%u].", size);
    // symmetric_memory.cc:375 — size_t 用 %ld
    HCCL_ERROR("...size[%ld]. ", ptr, size);

修复建议:

    // size_t 统一使用 %zu
    HCCL_ERROR("...granularity[%zu], ret[%d]", granularity, ret);
    // void** 应打印解引用后的值或改变量名
    HCCL_ERROR("...virPtr[%p] size[%zu]...", *ptr, allocSize, ret);

---

### #10 [一般] HcclScatterInner 声明缺少 extern 关键字

- 位置: `include/hccl/hccl_comm.h:323`
- 规则: API 声明一致性
- 置信度: 确定

分析:

`hccl_comm.h` 中其他函数声明（HcclCommWindowRegister、HcclMemAlloc 等）都使用 `extern` 关键字，但 `HcclScatterInner` 没有。此外该函数也缺少 doxygen 注释，与文件中其他 API 风格不一致。该声明似乎是无关变更，不应混入此 MR。

问题代码:

    HcclResult HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
        HcclComm comm, aclrtStream stream);

修复建议:

    // 若确需在此文件声明，补齐 extern 和注释；否则移除此无关变更
    extern HcclResult HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount,
        HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);

---

### #11 [一般] HcclMemFree 中类型混用：aclError 变量接收 HcclResult 返回值

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:121`
- 规则: 2.7.1（隐式类型转换）
- 置信度: 确定

分析:

`HcclMemFree` 中 `ret` 声明为 `aclError`，但 `aclMemRetainHandle` 返回 `HcclResult`。虽然两者的成功值都是 0，但错误码体系不同，`ret != ACL_SUCCESS` 的检查在语义上不正确。错误日志 `[HcclMemFree] RetainAllocationHandle` 中的函数名也与实际调用的 `aclMemRetainHandle` 不一致。

问题代码:

    aclError ret = ACL_SUCCESS;
    ...
    ret = aclMemRetainHandle(ptr, &handle);  // 返回 HcclResult，赋给 aclError
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] RetainAllocationHandle..."), HCCL_E_RUNTIME);

修复建议:

    HcclResult hcclRet = aclMemRetainHandle(ptr, &handle);
    CHK_PRT_RET(hcclRet != HCCL_SUCCESS,
        HCCL_ERROR("[HcclMemFree] aclMemRetainHandle virPtr[%p] failed, ret[%d]", ptr, hcclRet), HCCL_E_RUNTIME);

---

### #12 [建议] ParseSymmetricMemoryStride 函数体几乎全部注释掉，仅硬编码默认值

- 位置: `src/platform/common/externalinput.cc:1308-1325`
- 规则: 2.1.3（冗余代码）
- 置信度: 确定

分析:

函数内有约 15 行注释掉的环境变量解析逻辑，仅保留一行 `g_externalInput.symmetricMemoryStride = SYMMETRIC_MEM_STRIDE_DEFAULT;`。大段注释代码不应提交到主分支，应在功能完成后补齐实现或移除注释代码。

问题代码:

    // std::string symmetricMemoryStrideEnv = GET_ENV(MM_ENV_HCCL_SYMMETRIC_MEM_STRIDE);
    // if (symmetricMemoryStrideEnv == "EmptyString") {
    //     ...
    // }
    g_externalInput.symmetricMemoryStride = SYMMETRIC_MEM_STRIDE_DEFAULT;

修复建议:

移除注释代码，保留简洁的默认值赋值，或完成环境变量解析功能。

---

## 总结

本 MR 引入了一个完整的对称内存子系统，架构设计合理（VA 空间预分配 + handle 交换 + 内存映射），但实现存在多个严重缺陷。

最关键的问题是 `SymmetricWindow::stride` 类型截断（#1）：默认 16GB stride 在 u32 字段中被截断为 0，直接导致所有对称内存地址计算返回相同地址，该功能在当前代码下无法正确工作。`HcclGetSymPtr` 的有符号/无符号混合运算（#2）在此基础上雪上加霜。

其次，全局 map 无锁保护（#3）、设备内存泄漏（#4, #8）、new 后未判空（#5）等问题影响稳定性和安全性。

建议优先处理 8 个严重问题，其中 7 个确定，1 个较确定。#1 和 #2 是功能正确性的阻塞项，应在合入前修复。
