# Code Review: PR #567

| 属性 | 值 |
|------|------|
| 标题 | Support symmetric memory for aicpu unflod mode |
| 作者 | linzhenkang |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/567](https://gitcode.com/cann/hcomm-dev/merge_requests/567) |
| 审查时间 | 2026-02-24 11:13:33 |
| 审查工具 | Claude Code (`codereview` skill) |
| 发现 | 严重 6 / 一般 4 / 建议 2 |

---

## 变更概述

本 PR 为 HCCL 框架实现了对称内存（Symmetric Memory）支持，使 AICPU 展开模式下的集合通信算子能通过 VMM（Virtual Memory Management）API 实现跨 rank 的直接内存访问，从而绕过传统的 zero-copy IPC 交换流程。主要变更：

- `symmetric_memory.cc/h`（新增）：对称内存管理器核心实现，包含 VA 空间预留、物理内存映射、窗口注册/注销
- `hccl_mem_alloc.cc/h`（新增）：基于 VMM API 的设备内存分配/释放，维护 VA→handle 映射
- `aicpu_symmetric_memory.cc/h`（新增）：设备端对称指针计算函数 `HcclGetSymPtr`
- `hccl_communicator_host.cc`：集成对称内存初始化、`IsSupportSymmetricMemory` 判断、`PrepareZeroCopy` 旁路
- `aicpu_communicator.cc`：设备端远程内存地址准备和本地指针替换
- `coll_alg_param.h` / `aicpu_operator_pub.h`：OpParam、OpTilingData 结构体新增对称内存字段
- 四个算法 operator（allgather/allreduce/broadcast/reduce_scatter）：算法选择条件扩展
- `externalinput.cc/h`：新增环境变量 `HCCL_SYMMETRIC_MEMORY_STRIDE` 解析
- `hccl_comm.h` / `hccl_types.h`：公共 API 新增 `HcclWindow` 类型、`HcclCommWindowRegister/DeRegister`、`HcclMemAlloc/Free`

涉及 34 个 C/C++ 文件，约 1200 行新增/修改。

## 审查发现

共发现 12 个问题（严重 6 / 一般 4 / 建议 2）

---

### #1 [严重] SymmetricWindow::stride 类型为 u32，无法容纳实际 stride 值（16GB），导致截断为 0

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.h:44`
- 规则: 红线 1.3（整数溢出/翻转）
- 置信度: 确定。已确认 `stride` 为 `u32`（即 `uint32_t`，最大值 ~4.3GB），而 `stride_` 在 `SymmetricMemory` 类中为 `size_t`。默认 stride = 16GB = 0x400000000，超出 u32 范围。

问题代码:

    // symmetric_memory.h:44
    u32 stride;

    // symmetric_memory.cc（RegisterSymmetricMem 中）:
    pWin->stride = stride_;  // size_t → u32 截断

分析:

`SymmetricMemory` 构造函数接收 `size_t stride` 参数，值为 `GetExternalInputSymmetricMemoryStride() * 1024 * 1024 * 1024`（默认 16 * 1024^3 = 0x400000000）。该值赋给 `SymmetricWindow::stride`（u32）时被截断为 0。随后 `HcclGetSymPtr` 使用 `symWin->stride` 计算 peer 偏移量 `(peerRank - localRank) * stride`，乘以 0 后所有 rank 的偏移都变成相同值，导致远程内存访问指向错误地址。这是一个数据损坏/安全性缺陷。

修复建议:

将 `SymmetricWindow::stride` 类型改为 `size_t`：

    size_t stride;

---

### #2 [严重] HcclGetSymPtr 中混合符号运算导致偏移计算错误

- 位置: `src/framework/device/framework/aicpu_symmetric_memory.cc:29`
- 规则: 红线 1.3（整数溢出/翻转）
- 置信度: 较确定。已确认 `peerRank` 为 `int32_t`，`symWin->localRank` 为 `u32`，`symWin->stride` 为 `u32`。

问题代码:

    size_t peerOffset = symWin->heapOffset + (peerRank - symWin->localRank) * symWin->stride + offset;

分析:

当 `peerRank < localRank` 时，表达式 `(peerRank - symWin->localRank)` 中 `int32_t` 被隐式提升为 `u32`（C++ 整数提升规则），产生一个巨大的无符号数（如 peerRank=0, localRank=1 → 0xFFFFFFFF）。该值再乘以 stride 并在 u32 中回绕，然后零扩展到 `size_t`，与有符号计算的结果完全不同。即使 #1 修复后 stride 改为 size_t，子表达式 `(int32_t - u32)` 仍在 u32 域中计算，结果被提升到 size_t 时零扩展而非符号扩展。

修复建议:

明确使用有符号类型进行差值计算：

    ptrdiff_t rankDiff = static_cast<ptrdiff_t>(peerRank) - static_cast<ptrdiff_t>(symWin->localRank);
    size_t peerOffset = symWin->heapOffset + rankDiff * static_cast<ptrdiff_t>(symWin->stride) + offset;

同时建议将 `HcclGetSymPtr` 的 `peerRank` 参数改为 `u32`，与调用方类型一致（`PrepareUserMemRanges` 中 peerRank 为 `size_t`，`ExecOp` 中 `localUsrRankId` 为 `u32`）。

---

### #3 [严重] heapOffset 计算下溢：alignedPtr - ptr 为负值赋给 size_t

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:398`
- 规则: 红线 1.3（整数溢出/翻转）
- 置信度: 较确定。已确认 `alignedPtr = ptr & ~(granularity_ - 1)` 即向下对齐，因此 `alignedPtr <= ptr`。

问题代码:

    pWin->heapOffset = (uintptr_t)alignedPtr - (uintptr_t)ptr + offset;

分析:

`alignedPtr` 是 `ptr` 向下对齐到 granularity 的结果，所以 `alignedPtr <= ptr`。`(uintptr_t)alignedPtr - (uintptr_t)ptr` 在 ptr 未对齐时为负值，但 `uintptr_t` 是无符号类型，产生一个极大的无符号数（如 ptr 偏移 4096 → 0xFFFFFFFFFFFFE000）。加上 offset 后 heapOffset 仍然是错误值。后续 `HcclGetSymPtr` 基于此计算 peer VA 地址时会指向完全错误的内存位置。

修复建议:

将减法方向反转，计算 ptr 相对于对齐基地址的正偏移量，并在 `HcclGetSymPtr` 中正确组合：

    pWin->heapOffset = offset;
    pWin->ptrDiff = (uintptr_t)ptr - (uintptr_t)alignedPtr;  // 新增字段，正值

或直接基于 offset 和 ptrDiff 重新设计地址计算逻辑。

---

### #4 [严重] 全局变量 va2Handle_ 无线程安全保护，并发访问为 UB

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:20`
- 规则: 红线 1.7（并发安全 / data race）
- 置信度: 确定。`va2Handle_` 是文件级全局变量，三个函数直接读写，无任何锁保护。`HcclMemAlloc`/`HcclMemFree` 是公共 API。

问题代码:

    std::unordered_map<void*, void*> va2Handle_{};

分析:

`HcclMemAlloc` 和 `HcclMemFree` 是 `hccl_comm.h` 中暴露的公共 C API，可被多线程并发调用。三个操作函数 `HcclAddVa2Handle`、`aclMemRetainHandle`、`HcclDelVa2Handle` 均直接访问 `va2Handle_` 而无互斥保护。`std::unordered_map` 的并发读写是未定义行为。

修复建议:

添加 mutex 保护，并修正命名（全局变量应使用 `g_` 前缀）：

    static std::mutex g_va2HandleMutex;
    static std::unordered_map<void*, void*> g_va2Handle;

    HcclResult HcclAddVa2Handle(void *devPtr, void *handle)
    {
        std::lock_guard<std::mutex> lock(g_va2HandleMutex);
        // ...
    }

---

### #5 [严重] new (std::nothrow) 后未检查 null 即使用

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:396`
- 规则: 2.16.1（内存分配后未判空）
- 置信度: 确定

问题代码:

    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    pWin->userVa = ptr;

分析:

`new (std::nothrow)` 在内存不足时返回 `nullptr`。`std::shared_ptr` 会持有 nullptr，下一行立即 `pWin->userVa = ptr` 触发空指针解引用。应在使用前检查。

修复建议:

    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    CHK_SMART_PTR_NULL(pWin);
    pWin->userVa = ptr;

---

### #6 [严重] RegisterWindow/DeregisterWindow 未检查 symmetricMemory_ 是否为 null

- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:8521, 8526`
- 规则: 红线 1.5（空指针解引用）
- 置信度: 较确定。已确认 `symmetricMemory_` 在 `InitSymmetricMemory` 中通过 `make_unique` 创建，但 `RegisterWindow`/`DeregisterWindow` 可在 `InitSymmetricMemory` 之前被调用（它们是公共 API，通过 `hcclComm::RegisterWindow` 暴露）。

问题代码:

    HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags)
    {
        return symmetricMemory_->RegisterSymmetricMem(ptr, size, winHandle);
    }

分析:

`symmetricMemory_` 仅在 `InitSymmetricMemory` 中初始化，而该函数在 `RankGraph::Init` 之后才被调用。如果用户在通信域未完全初始化前调用 `HcclCommWindowRegister`，`symmetricMemory_` 仍为 nullptr，直接解引用导致崩溃。

修复建议:

    HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags)
    {
        CHK_SMART_PTR_NULL(symmetricMemory_);
        return symmetricMemory_->RegisterSymmetricMem(ptr, size, winHandle);
    }

---

### #7 [一般] HcclMemFree 中返回值类型混用：aclMemRetainHandle 返回 HcclResult 但赋给 aclError

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:125`
- 规则: 3.1.3（类型安全）
- 置信度: 确定。已确认 `aclMemRetainHandle`（该文件第 38 行定义）返回 `HcclResult`，但 `HcclMemFree` 中 `ret` 声明为 `aclError`，用 `ret != ACL_SUCCESS` 判断。

问题代码:

    aclError ret = ACL_SUCCESS;
    // ...
    ret = aclMemRetainHandle(ptr, &handle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] RetainAllocationHandle ..."), HCCL_E_RUNTIME);

分析:

`aclMemRetainHandle` 是本文件定义的函数，返回 `HcclResult`（HCCL 错误码体系），但被赋给 `aclError` 变量并与 `ACL_SUCCESS`（ACL 错误码体系）比较。虽然两者的成功值都是 0，但错误码值域不同，在失败路径上判断可能不正确。日志中函数名 "RetainAllocationHandle" 也与实际调用的 "aclMemRetainHandle" 不一致。

修复建议:

    HcclResult hcclRet = aclMemRetainHandle(ptr, &handle);
    CHK_PRT_RET(hcclRet != HCCL_SUCCESS,
        HCCL_ERROR("[HcclMemFree] aclMemRetainHandle virPtr[%p] failed, ret[%d]", ptr, hcclRet), HCCL_E_RUNTIME);

---

### #8 [一般] OpTilingData 新增字段 inputWindow / outputWindow 未初始化

- 位置: `src/pub_inc/aicpu_operator_pub.h:706, 708`
- 规则: 红线 1.4（变量未初始化）
- 置信度: 较确定。已确认 `OpTilingData` 通过 `HostMem::alloc` 分配且不会自动零初始化。该结构体通过 H2D 传输到设备端。

问题代码:

    u64 inputWindow;
    u64 outputWindow;

分析:

同一段中 `inputOffset = 0` 和 `outputOffset = 0` 有默认值，但 `inputWindow` 和 `outputWindow` 没有。虽然在 `AicpuInitOpTilingDataFromOpParam`（hccl_communicator.cc:1689）中会显式赋值，但在 `AicpuInitOpTilingDataBuf`（hccl_communicator_host.cc:6835 路径）中若非对称内存场景，这些字段可能未被初始化就被传输到设备端。未初始化的随机值可能导致设备端误判。

修复建议:

    u64 inputWindow = 0;
    u64 outputWindow = 0;

---

### #9 [一般] 多处格式字符串与参数类型不匹配

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:88, 90`, `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:240, 266, 375`
- 规则: 3.1.3（格式字符串参数匹配）
- 置信度: 确定。已确认日志宏直接传递给 printf 式函数，无类型适配。

问题代码:

    // hccl_mem_alloc.cc:88 — granularity 是 size_t，用 %llu
    HCCL_ERROR("... granularity[%llu], ret[%d]", granularity, ret);

    // symmetric_memory.cc:240 — stride_ 是 size_t，用 %u
    HCCL_ERROR("... Stride %u is not a multiple ...", stride_, granularity_);

    // symmetric_memory.cc:266 — size 是 size_t，用 %u
    HCCL_ERROR("... failed for size[%u].", size);

    // symmetric_memory.cc:375 — size 是 size_t，用 %ld
    HCCL_ERROR("... size[%ld]. ", ptr, size);

分析:

`size_t` 的标准格式说明符是 `%zu`。使用 `%u`（32位）读取 64 位 `size_t` 会导致参数栈偏移，后续参数全部错位，输出垃圾值。使用 `%llu` 在 LP64 平台上也是类型不匹配（`unsigned long` vs `unsigned long long`），虽然宽度相同但属于未定义行为。

修复建议:

所有 `size_t` 类型参数统一使用 `%zu`。

---

### #10 [一般] HcclScatterInner 声明误插入公共 API 头文件，且缺少 extern 关键字

- 位置: `include/hccl/hccl_comm.h:322`
- 规则: 头文件组织规范
- 置信度: 确定

问题代码:

    HcclResult HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
        HcclComm comm, aclrtStream stream);

分析:

`include/hccl/hccl_comm.h` 是外部公共 API 头文件。该函数名包含 `Inner` 后缀，属于内部 API，不应暴露在外部头文件中。同时其他函数均使用 `extern` 关键字声明，此函数缺少 `extern`。且该声明与对称内存特性无关，疑似误合入。

修复建议:

将此声明移至内部头文件（如 `pkg_inc/hccl/hccl_inner.h`），或添加 `extern` 关键字并确认是否应暴露。

---

### #11 [建议] 多个新文件末尾缺少换行符

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:138`, `src/framework/common/src/hccl_mem_alloc.h:23`, `src/framework/communicator/impl/symmetric_memory/symmetric_memory.h:97`, `src/framework/device/framework/aicpu_symmetric_memory.cc:32`, `src/framework/device/framework/aicpu_symmetric_memory.h:19`, `test/llt/ut/single_test/impl/ut_hccl_mem_alloc.cc:195`
- 规则: 编码规范（POSIX 文件格式）
- 置信度: 确定

分析:

POSIX 标准要求文本文件以换行符结尾。缺少换行符可能导致部分编译器产生警告，也会使 diff 工具在追加内容时产生额外噪声。

修复建议:

在每个文件末尾添加一个空行。

---

### #12 [建议] ParseSymmetricMemoryStride 函数中大量代码被注释掉

- 位置: `src/platform/common/externalinput.cc:1308-1323`
- 规则: 2.1.3（冗余代码）
- 置信度: 确定

问题代码:

    // std::string symmetricMemoryStrideEnv = GET_ENV(MM_ENV_HCCL_SYMMETRIC_MEM_STRIDE);
    // if (symmetricMemoryStrideEnv == "EmptyString") {
    //     g_externalInput.symmetricMemoryStride = SYMMETRIC_MEM_STRIDE_DEFAULT;
    // ...
    g_externalInput.symmetricMemoryStride = SYMMETRIC_MEM_STRIDE_DEFAULT;
    // CHK_RET(IsAllDigit(symmetricMemoryStrideEnv.c_str()));
    // HcclResult ret = SalStrToULong(symmetricMemoryStrideEnv, HCCL_BASE_DECIMAL, symmetricMemoryStride);

分析:

当前函数体中 16 行代码被注释掉，仅剩一行 hardcode 赋默认值。环境变量解析逻辑被完全禁用，意味着用户无法通过环境变量调整 stride 大小。若为开发中间态，应在提交前清理或添加 TODO 说明。

修复建议:

完成环境变量解析逻辑或清理注释代码，仅保留必要的实现。

---

## 总结

本 PR 实现了对称内存的核心框架，但存在多个严重的数值计算缺陷。#1（stride 类型截断为 0）、#2（混合符号运算）、#3（heapOffset 下溢）三个问题相互叠加，导致 `HcclGetSymPtr` 计算的远程地址完全错误，对称内存功能在当前代码状态下无法正常工作。#4（全局变量 data race）在并发场景下会导致未定义行为。建议优先修复 6 个严重问题（其中 4 个确定、2 个较确定），然后处理 4 个一般问题。
