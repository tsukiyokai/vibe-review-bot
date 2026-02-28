# Code Review: PR #895

| 属性 | 值 |
|------|------|
| 标题 | Support symmetric memory for aicpu unflod mode |
| 作者 | linzhenkang |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/895](https://gitcode.com/cann/hcomm-dev/merge_requests/895) |
| 审查时间 | 2026-02-24 10:42:06 |
| 审查工具 | Claude Code (`codereview` skill) |
| 发现 | 严重 5 / 一般 1 / 建议 1 |

---

## 变更概述

本 MR 为 HCCL 框架实现了 AICPU 展开模式下的对称内存(Symmetric Memory)支持，主要变更：
- 新增 `SymmetricMemory` 和 `SymmetricMemoryAgent` 类，负责对称 VA 空间预留、物理内存跨 rank 映射、Window 注册/注销
- 新增公共 API `HcclCommSymWinRegister`/`HcclCommSymWinDeregister`/`HcclCommSymWinGet`/`HcommSymWinGetPeerPointer`
- 算法选择层(allgather/allreduce/reduce_scatter)增加 `supportSymmetricMemory` 分支，对称内存优先于 ZeroCopy
- 新增 `HcclMemAlloc`/`HcclMemFree` 封装 VMM 物理内存分配
- Device 侧 `aicpu_communicator` 增加对称内存地址重映射逻辑
- 通信域配置版本升至 V10，新增 `hcclSymWinMaxMemSizePerRank` 配置项

涉及 47 个文件（41 个 C/C++），约 2200 行新增/修改。

## 审查发现

共发现 7 个问题（严重 5 / 一般 1 / 建议 1）

---

### #1 [严重] 析构函数中遍历容器同时被调用函数删除元素，迭代器失效导致未定义行为
- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:192`
- 规则: 红线 1.2（数组/容器越界）、内存安全
- 置信度: 确定。已确认 `DeregisterSymmetricMem` 在第 516 行调用 `windowMap_.erase(devWin)`，而析构函数正在对 `windowMap_` 做 range-for 遍历。

问题代码:

    for (auto& pair : windowMap_) {
        DeregisterSymmetricMem(pair.first);
    }

分析: range-for 等价于使用 `begin()`/`end()` 迭代器遍历 `windowMap_`。`DeregisterSymmetricMem` 内部（第 516 行）调用 `windowMap_.erase(devWin)`，使当前迭代器失效。后续迭代访问已失效迭代器，属于 C++ 标准定义的未定义行为，极大概率导致 crash 或数据损坏。

修复建议:

    while (!windowMap_.empty()) {
        DeregisterSymmetricMem(windowMap_.begin()->first);
    }

---

### #2 [严重] IsSupportSymmetricMemory 未检查 symmetricMemory_ 空指针，superPodNum_ > 1 时必然崩溃
- 位置: `src/framework/communicator/impl/hccl_communicator_host.cc:613`
- 规则: 红线 1.5（空指针解引用）
- 置信度: 确定。已确认 `InitSymmetricMemory()`（第 8870 行）在 `superPodNum_ > 1` 时直接返回 HCCL_SUCCESS 而不创建 `symmetricMemory_`；但 `IsSupportSymmetricMemory` 仅检查了 `deviceType_ == 910_93`，未检查 `superPodNum_`，也未检查 `symmetricMemory_` 是否为空。

问题代码:

    HcclResult ret = symmetricMemory_->FindSymmetricWindow(opParam.inputPtr, opParam.inputSize, &opParam.inputSymWindow, &opParam.inputOffset);

分析: 触发路径：910_93 设备 + superPodNum_ > 1 + aicpuUnfoldMode + OP_BASE 模式 + deviceNumPerAggregation_ > 1 → 所有前置 CHK_PRT_RET 均通过 → 对 null `symmetricMemory_` 调用成员函数 → crash。

修复建议: 在函数开头增加空指针检查：

    if (symmetricMemory_ == nullptr) {
        HCCL_INFO("[%s] symmetricMemory_ is nullptr, not support", __func__);
        return false;
    }

---

### #3 [严重] HcclMemFree 中间步骤失败时直接返回，导致物理内存和 VA 地址泄漏
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:80, 82`
- 规则: 红线 1.6（资源泄漏）
- 置信度: 确定

问题代码:

    ret = aclrtUnmapMem(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] UnmapMem virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
    ret = aclrtFreePhysical(handle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] FreePhysical handle[%p] failed, ret[%d]", handle, ret), HCCL_E_RUNTIME);
    ret = aclrtReleaseMemAddress(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] ReleaseMemAddress virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);

分析: 三步清理操作（unmap → free physical → release VA）的每一步失败都直接返回，跳过后续清理。例如 `aclrtUnmapMem` 失败时，物理内存（handle）和 VA 地址（ptr）均未释放。与同文件中 `HcclMemAlloc` 的正确做法（第 53-61 行，失败时依次清理已分配资源）形成对比。

修复建议: 改为记录错误但继续执行所有清理步骤：

    HcclResult finalRet = HCCL_SUCCESS;
    ret = aclrtUnmapMem(ptr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[HcclMemFree] UnmapMem virPtr[%p] failed, ret[%d]", ptr, ret);
        finalRet = HCCL_E_RUNTIME;
    }
    ret = aclrtFreePhysical(handle);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[HcclMemFree] FreePhysical handle[%p] failed, ret[%d]", handle, ret);
        finalRet = HCCL_E_RUNTIME;
    }
    ret = aclrtReleaseMemAddress(ptr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[HcclMemFree] ReleaseMemAddress virPtr[%p] failed, ret[%d]", ptr, ret);
        finalRet = HCCL_E_RUNTIME;
    }
    return finalRet;

---

### #4 [严重] 格式字符串 %u 用于 size_t 参数，64 位平台上参数宽度不匹配
- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:287, 293, 381`
- 规则: 3.1.3（格式字符串参数匹配）
- 置信度: 确定。`size_t` 在 LP64/aarch64 上为 8 字节（unsigned long），`%u` 仅读取 4 字节（unsigned int）。

问题代码:

    // 第 287 行
    HCCL_ERROR("[SymmetricMemory][AllocSymmetricMem] HcclMemAlloc failed for size[%u].", size);
    // 第 293 行
    HCCL_ERROR("[SymmetricMemory][AllocSymmetricMem] RegisterSymmetricMem failed for ptr[%p], size[%u].", ptr, size);
    // 第 381 行
    HCCL_ERROR("[SymmetricMemory][GetMemoryInfo] baseVaSize %u is not a multiple of granularity %zu.",
        *baseVaSize, granularity_);

分析: `size` 和 `*baseVaSize` 均为 `size_t`（8 字节），`%u` 预期 `unsigned int`（4 字节）。在 variadic 函数调用中，这造成参数读取宽度错误：要么截断高 4 字节（值 >= 4GB 时打印错误），要么导致后续 `%` 占位符读到错位的参数。对于内存管理模块，size >= 4GB 是正常场景。

修复建议: 将 `%u` 改为 `%zu`（或项目惯例的 `%llu` 并强制转换为 `unsigned long long`）：

    HCCL_ERROR("[SymmetricMemory][AllocSymmetricMem] HcclMemAlloc failed for size[%zu].", size);

---

### #5 [严重] HcclCommSymWinGet 缺少 offset 参数空指针检查
- 位置: `src/framework/op_base/src/op_base.cc:4616`
- 规则: 红线 1.5（空指针解引用）
- 置信度: 确定。函数对 `comm`、`ptr`、`winHandle` 均做了 `CHK_PTR_NULL` 检查，唯独遗漏 `offset`。已确认下层 `FindSymmetricWindow` 在匹配成功时直接写入 `*offset`（symmetric_memory.cc:527），且本函数在第 4627 行日志中解引用 `*offset`。

问题代码:

    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(ptr);
    CHK_PTR_NULL(winHandle);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] size is 0, please check size value", __func__), HCCL_E_PARA);

分析: 若调用方传入 `offset = nullptr`，`FindSymmetricWindow` 内部对 `*offset` 的写入或日志中 `*offset` 的读取将触发空指针写/读。此函数是公共 API，需防御性检查所有输出参数。

修复建议: 在现有检查后补充：

    CHK_PTR_NULL(offset);

---

### #6 [一般] 全局变量 winHandle2comm 未遵循 g_ 前缀命名规范
- 位置: `src/framework/op_base/src/op_base.cc:4567`
- 规则: 1.1.5（全局变量命名）
- 置信度: 确定。同一处的 `g_winHandleMtx`（第 4568 行）正确使用了 `g_` 前缀，但 `winHandle2comm` 遗漏。

问题代码:

    std::unordered_map<CommSymWindow, HcclComm> winHandle2comm;
    std::mutex g_winHandleMtx; // 保护 winHandle2comm

修复建议:

    std::unordered_map<CommSymWindow, HcclComm> g_winHandle2comm;

同时更新所有引用处（第 4583, 4584, 4601, 4602, 4603, 4611 行）。

---

### #7 [建议] 多个新增文件末尾缺少换行符
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:89`, `src/framework/common/src/hccl_mem_alloc.h:33`, `src/framework/communicator/impl/symmetric_memory/symmetric_memory.h:134`, `src/framework/communicator/impl/symmetric_memory/symmetric_memory_agent.cc:249`, `src/framework/communicator/impl/symmetric_memory/symmetric_memory_agent.h:101`
- 规则: POSIX 文本文件规范
- 置信度: 确定（diff 末尾标注 `\ No newline at end of file`）

分析: POSIX 标准要求文本文件以换行符结尾。部分编译器/工具可能产生警告。

修复建议: 在每个文件末尾添加一个空行。

---

## 总结

本 MR 实现了完整的对称内存管理功能，包括 VA 空间分配器、跨 rank 物理内存映射、与算法选择层的集成。代码结构清晰，错误处理路径考虑较全面，UT 覆盖面广。

建议优先处理 5 个严重问题：其中 #1（析构中迭代器失效）和 #2（superPodNum_ > 1 时空指针崩溃）为必现的运行时崩溃，风险最高；#3（HcclMemFree 资源泄漏）会在异常路径累积泄漏物理内存；#4（格式字符串 %u）在 size >= 4GB 时打印错误值并可能导致后续占位符错位。
