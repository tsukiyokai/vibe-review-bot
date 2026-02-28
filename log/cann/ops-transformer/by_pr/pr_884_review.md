# Code Review: PR #884

| 属性 | 值 |
|------|------|
| 标题 | Support symmetric memory for aicpu unflod mode |
| 作者 | linzhenkang |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/884](https://gitcode.com/cann/hcomm-dev/merge_requests/884) |
| 审查时间 | 2026-02-24 10:42:48 |
| 审查工具 | Claude Code (`codereview` skill) |
| 发现 | 严重 6 / 一般 4 / 建议 2 |

---

## 变更概述

本 MR 为 HCCL 框架层和算法层实现了对称内存（Symmetric Memory）支持，用于 AICPU unfold 模式下的通信优化。主要变更：

- `include/hccl/hccl_comm.h`, `hccl_types.h`: 新增 `HcclWindow` 类型和 `HcclCommWindowRegister`/`HcclMemAlloc`/`HcclCommGetSymPtr` 等公共 API
- `src/framework/communicator/impl/symmetric_memory/`: 新增 `SymmetricMemory` 类，实现 VA 空间预留、PA 句柄交换、对称映射管理
- `src/framework/common/src/hccl_mem_alloc.cc`: 新增 `HcclMemAlloc`/`HcclMemFree`，基于 VMM API 做虚拟内存分配
- `src/algorithm/impl/operator/`: 四个集合算子的算法选择逻辑新增 `supportSymmetricMemory` 分支
- `src/framework/communicator/impl/hccl_communicator_host.cc`: 新增 `IsSupportSymmetricMemory`、`InitSymmetricMemory` 等方法
- `src/framework/device/framework/aicpu_communicator.cc`: AICPU 侧对称内存远端地址准备逻辑
- `src/pub_inc/aicpu_operator_pub.h`: `OpTilingData` 结构体新增对称内存相关字段

涉及 40 个文件，约 1300 行新增/修改。

## 审查发现

共发现 12 个问题（严重 6 / 一般 4 / 建议 2）

---

### #1 [严重] 运算符优先级错误导致 isSupportInlineReduce 检查从 zeroCopy 路径遗漏

- 位置: `src/algorithm/impl/operator/reduce_scatter_operator.cc:481`
- 规则: 红线 1.3（逻辑错误）
- 置信度: 确定

原始代码要求 `param.supportZeroCopy && isSupportInlineReduce && (拓扑条件)`，即 zeroCopy 路径必须检查 `isSupportInlineReduce`（注释明确说"不申请scratch ==> 不支持非InlineReduce"）。

问题代码:

    isSupportInlineReduce && param.supportSymmetricMemory || (param.supportZeroCopy &&
        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB))

由于 `&&` 优先级高于 `||`，实际解析为 `(isSupportInlineReduce && param.supportSymmetricMemory) || (param.supportZeroCopy && (拓扑条件))`。zeroCopy 分支不再检查 `isSupportInlineReduce`，与原始意图矛盾。其他三个算子（allgather/allreduce/broadcast）原本就没有 `isSupportInlineReduce` 条件，不受影响。

修复建议:

    isSupportInlineReduce && (param.supportSymmetricMemory ||
        (param.supportZeroCopy &&
        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB)))

分析: 已对比 master 分支原始代码确认。原始代码为 `param.supportZeroCopy && isSupportInlineReduce && (...)`，`isSupportInlineReduce` 是进入该分支的必要条件。

---

### #2 [严重] GetSymmetricPtr 的 symPtr 参数按值传递，调用方永远拿不到结果

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:503`, `include/hccl/hccl_comm.h:355`
- 规则: 红线 1.4（值传递悬垂/无效输出）
- 置信度: 确定

问题代码:

    // symmetric_memory.cc:489
    HcclResult SymmetricMemory::GetSymmetricPtr(void* ptr, size_t size, void** win, void *symPtr)
    // ...
    symPtr = pWin->userVa;  // 修改局部副本，调用方看不到

    // hccl_comm.h:355
    extern HcclResult HcclCommGetSymPtr(HcclComm comm, void *ptr, size_t size, HcclWindow *winHandle, void *symPtr);

`symPtr` 是 `void*` 值传递形参，赋值 `symPtr = pWin->userVa` 仅修改栈上副本。公共 API `HcclCommGetSymPtr` 也存在同样问题，调用方永远无法获取对称内存指针。

修复建议:

    // 函数签名改为 void **symPtr
    HcclResult GetSymmetricPtr(void* ptr, size_t size, void** win, void **symPtr);
    // 赋值改为
    *symPtr = pWin->userVa;

    // 公共 API 同步修改
    extern HcclResult HcclCommGetSymPtr(HcclComm comm, void *ptr, size_t size, HcclWindow *winHandle, void **symPtr);

---

### #3 [严重] ExchangePhyAddrHandle 中间步骤失败导致设备内存泄漏

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:628-635`
- 规则: 红线 1.6（资源泄漏）
- 置信度: 确定

问题代码:

    CHK_RET(hrtMalloc((void**)&sendBuff, inputSize));
    CHK_RET(hrtMalloc((void**)&recvBuff, outputSize));
    CHK_RET(hrtMemSyncCopy(...));
    CHK_RET(HcclAllGatherInner(...));
    CHK_RET(hcclStreamSynchronize(...));
    CHK_RET(hrtMemSyncCopy(...));

`CHK_RET` 失败时直接 return，但 `sendBuff` 和 `recvBuff` 未被释放。例如：
- 第二个 `hrtMalloc` 失败 → `sendBuff` 泄漏
- `HcclAllGatherInner` 失败 → `sendBuff` 和 `recvBuff` 都泄漏

修复建议: 使用 RAII 或 goto 清理模式，确保所有错误路径释放已分配的缓冲区。

---

### #4 [严重] new(std::nothrow) 返回 nullptr 后立即解引用

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:411-412`
- 规则: 红线 1.5（空指针解引用）、规则 2.16.1（内存分配后未判空）
- 置信度: 确定

问题代码:

    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    pWin->userVa = baseUserVa;

`new (std::nothrow)` 分配失败返回 `nullptr`，`shared_ptr` 持有 `nullptr`，下一行 `pWin->userVa` 即为空指针解引用。

修复建议:

    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    CHK_PTR_NULL(pWin.get());
    pWin->userVa = baseUserVa;

---

### #5 [严重] HcclMemFree 部分失败导致资源泄漏

- 位置: `src/framework/common/src/hccl_mem_alloc.cc:77-82`
- 规则: 红线 1.6（资源泄漏）
- 置信度: 确定

问题代码:

    ret = aclrtUnmapMem(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, ..., HCCL_E_RUNTIME);
    ret = aclrtFreePhysical(handle);
    CHK_PRT_RET(ret != ACL_SUCCESS, ..., HCCL_E_RUNTIME);
    ret = aclrtReleaseMemAddress(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, ..., HCCL_E_RUNTIME);

释放步骤间用 `CHK_PRT_RET` 提前返回。如果 `aclrtUnmapMem` 失败，物理句柄 `handle` 和虚拟地址 `ptr` 都不会释放。释放操作应尽力执行所有步骤（best-effort），记录错误但不提前返回。

修复建议:

    HcclResult result = HCCL_SUCCESS;
    aclError aclRet = aclrtUnmapMem(ptr);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[HcclMemFree] UnmapMem virPtr[%p] failed, ret[%d]", ptr, aclRet);
        result = HCCL_E_RUNTIME;
    }
    aclRet = aclrtFreePhysical(handle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[HcclMemFree] FreePhysical handle[%p] failed, ret[%d]", handle, aclRet);
        result = HCCL_E_RUNTIME;
    }
    aclRet = aclrtReleaseMemAddress(ptr);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[HcclMemFree] ReleaseMemAddress virPtr[%p] failed, ret[%d]", ptr, aclRet);
        result = HCCL_E_RUNTIME;
    }
    return result;

---

### #6 [严重] strcpy_s 返回值未检查

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:609`
- 规则: 2.18.6（安全函数返回值必须检查）
- 置信度: 确定

问题代码:

    strcpy_s(config.hcclCommName, sizeof(config.hcclCommName), identifier.c_str());

`identifier` 是通信域名拼接 `"::symmetric_memory_sub_comm"` 后的字符串，可能超过 `config.hcclCommName` 的缓冲区大小，且 `strcpy_s` 的返回值未检查。

修复建议:

    CHK_SAFETY_FUNC_RET(strcpy_s(config.hcclCommName, sizeof(config.hcclCommName), identifier.c_str()));

---

### #7 [一般] 格式字符串类型不匹配

- 位置: `src/framework/cluster_maintenance/health/heartbeat/heartbeat.cc:581`
- 规则: 3.1.3（格式字符串参数匹配）
- 置信度: 确定

问题代码:

    HCCL_INFO("[Heartbeat][UnRegisterToHeartBeat] groupMap_.size[%d]", groupMap_.size());

`groupMap_.size()` 返回 `size_t`，用 `%d` (int) 格式化是类型不匹配。在 64 位系统上 `size_t` 为 8 字节，`%d` 读取 4 字节，可能输出错误值。

修复建议:

    HCCL_INFO("[Heartbeat][UnRegisterToHeartBeat] groupMap_.size[%zu]", groupMap_.size());

---

### #8 [一般] 格式字符串类型不匹配: stride_ 是 size_t 但用 %u 格式化

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:237, 242`
- 规则: 3.1.3
- 置信度: 较确定。已确认 `stride_` 声明为 `size_t`（见 symmetric_memory.h:102）

问题代码:

    // 237行
    HCCL_ERROR("[SymmetricMemory][Init] Stride %u is not a multiple of granularity %zu.", stride_, granularity_);
    // 242行
    HCCL_ERROR("[SymmetricMemory][Init] aclrtReserveMemAddress failed to reserve %zu bytes. stride: %u, rankSize: %u.",
               totalHeapSize, stride_, rankSize_);

`stride_` 类型为 `size_t`，但使用 `%u` (`unsigned int`) 格式化。同样 `AllocSymmetricMem` 中的 `size` 参数也用了 `%u`。

修复建议: 将 `%u` 改为 `%zu`。

---

### #9 [一般] for 循环按值拷贝 map 元素而非引用

- 位置: `src/framework/cluster_maintenance/health/heartbeat/heartbeat.cc:578`
- 规则: 2.10.6（只读形参缺 const 引用）
- 置信度: 确定

问题代码:

    for (const auto it : groupMap_) {

`groupMap_` 是 `std::map<std::string, ...>`，每次迭代都会拷贝 key-value pair（包括 `std::string` 拷贝），纯为日志输出无此必要。

修复建议:

    for (const auto& it : groupMap_) {

---

### #10 [一般] 引入注释拼写错误

- 位置: `include/hccl/hccl_types.h:67`
- 规则: 1.3.x（注释规范）
- 置信度: 确定

问题代码:

    * @brief HCCL Reduction opperation

原文为 `operation`，PR 改为 `opperation`（多了一个 p）。

修复建议: 改回 `operation`。

---

### #11 [建议] Dump 函数使用 HCCL_ERROR 级别输出调试信息

- 位置: `src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc:39-48`
- 规则: HCCL 日志规范
- 置信度: 确定

问题代码:

    void Dump(const char* tag) {
        HCCL_ERROR("[%s] === VA Allocator Dump (Total: %zu) ===", tag, totalSize_);
        // ... 多行 HCCL_ERROR ...
    }

`Dump` 是调试打印函数（在 Reserve 失败时调用），不应使用 `HCCL_ERROR` 级别。ERROR 日志会触发告警/上报，调试信息应使用 `HCCL_INFO` 或 `HCCL_DEBUG`。

修复建议: 将 `HCCL_ERROR` 改为 `HCCL_INFO`（或如需保留在生产环境可见，用 `HCCL_WARNING`）。

---

### #12 [建议] HcclScatterInner 声明缺少 extern 关键字和 doxygen 注释

- 位置: `include/hccl/hccl_comm.h:322`
- 规则: 1.3.x（注释规范）
- 置信度: 确定

问题代码:

    HcclResult HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
        HcclComm comm, aclrtStream stream);

该文件中所有其他 C 函数声明都使用 `extern` 关键字并配有 doxygen 注释（如 `HcclCommWindowRegister`、`HcclMemAlloc` 等），此处既没有 `extern` 也没有注释。且函数名含 `Inner` 暗示是内部接口，不应放在外部公共头文件中。

修复建议: 要么添加 `extern` 和 doxygen 注释，要么将该声明移至内部头文件（如 `pkg_inc/hccl/hccl_inner.h`）。

---

## 总结

本 MR 实现了对称内存的核心功能，整体架构（VA 空间预留 + PA 句柄交换 + 对称映射）合理，懒初始化（`EnsureInit`/`call_once`）设计良好。但存在 6 个严重问题需优先修复：

1. reduce_scatter 运算符优先级 bug 会导致运行时选错算法路径（#1）
2. `GetSymmetricPtr` 按值传递导致 API 功能完全失效（#2）
3. `ExchangePhyAddrHandle` 和 `HcclMemFree` 的资源泄漏在异常路径下会累积（#3, #5）
4. `new(std::nothrow)` 未判空可能导致空指针崩溃（#4）
5. `strcpy_s` 返回值未检查可能导致缓冲区截断后静默继续（#6）

建议优先处理 6 个严重问题，其中全部为确定级别。
