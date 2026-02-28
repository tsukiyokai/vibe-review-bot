# Code Review: PR #110

| 属性 | 值 |
|------|------|
| 标题 | HcclMemAlloc |
| 作者 | linzhenkang |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/110](https://gitcode.com/cann/hcomm-dev/merge_requests/110) |
| 审查时间 | 2026-02-24 11:14:54 |
| 审查工具 | Claude Code (`codereview` skill) |
| 发现 | 严重 3 / 一般 3 / 建议 2 |

---

## 变更概述

本 MR 为 HCCL 新增虚拟内存分配/释放公共 API（`HcclMemAlloc` / `HcclMemFree`），主要变更：
- `inc/hccl/hccl_comm.h`: 新增两个 API 声明及一个不相关的 `HcclScatterInner` 声明
- `src/framework/common/src/hccl_mem_alloc.cc`: 实现文件，基于 ACL 虚拟内存 API 完成"预留地址→分配物理内存→映射"三步操作
- `src/framework/common/src/hccl_mem_alloc.h`: 头文件，定义 ALIGN_SIZE 对齐宏
- 3 个 stub 文件和 1 个 UT 文件: 新增 ACL API stub 和 14 个单元测试用例

涉及 9 个文件（7 个 C/C++ 文件），约 400 行新增代码。

## 审查发现

共发现 8 个问题（严重 3 / 一般 3 / 建议 2）

---

### #1 [严重] HcclMemAlloc 入口缺少 ptr 空指针检查
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:19`
- 规则: 红线 1.5（空指针解引用）
- 置信度: 确定

问题代码:

    HcclResult HcclMemAlloc(void **ptr, size_t size)
    {
        CHK_PRT_RET(size == 0, HCCL_ERROR("[HcclMemAlloc] size is zero"), HCCL_E_PARA);

函数入口只检查了 `size == 0`，未检查 `ptr == nullptr`。若调用者传入 `nullptr`，第 43 行 `aclrtReserveMemAddress(ptr, ...)` 会尝试向 `*ptr` 写入地址，触发空指针解引用崩溃。即使 ACL 内部不崩溃，第 47 行 `void *virPtr = *ptr` 也必然崩溃。

修复建议:

    HcclResult HcclMemAlloc(void **ptr, size_t size)
    {
        CHK_PTR_NULL(ptr);
        CHK_PRT_RET(size == 0, HCCL_ERROR("[HcclMemAlloc] size is zero"), HCCL_E_PARA);

---

### #2 [严重] HcclMemFree 中间步骤失败时资源泄漏
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:77, 78, 79, 80`
- 规则: 红线 1.6（资源泄漏）
- 置信度: 确定

问题代码:

    ret = aclrtUnmapMem(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("..."), HCCL_E_RUNTIME);
    ret = aclrtFreePhysical(handle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("..."), HCCL_E_RUNTIME);

`CHK_PRT_RET` 在条件成立时直接 return，导致后续清理步骤被跳过。具体泄漏路径：
- `aclrtUnmapMem` 失败 → `handle`（已通过 `aclMemRetainAllocationHandle` 获取）未释放，物理内存和虚拟地址均泄漏
- `aclrtFreePhysical` 失败 → `aclrtReleaseMemAddress(ptr)` 被跳过，虚拟地址泄漏

对比同函数中 `HcclMemAlloc` 的错误处理（第 50-60 行），`HcclMemAlloc` 在 `aclrtMallocPhysical`/`aclrtMapMem` 失败时正确执行了逆序清理。`HcclMemFree` 应采用相同模式。

修复建议:

    ret = aclrtUnmapMem(ptr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[HcclMemFree] UnmapMem virPtr[%p] failed, ret[%d]", ptr, ret);
        aclrtReleaseMemAddress(ptr);
        return HCCL_E_RUNTIME;
    }
    ret = aclrtFreePhysical(handle);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[HcclMemFree] FreePhysical handle[%p] failed, ret[%d]", handle, ret);
        aclrtReleaseMemAddress(ptr);
        return HCCL_E_RUNTIME;
    }
    ret = aclrtReleaseMemAddress(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] ReleaseMemAddress virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);

---

### #3 [严重] 格式字符串类型不匹配：%llu 用于 size_t
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:39, 41, 45, 51, 57`
- 规则: 3.1.3（格式字符串参数匹配）
- 置信度: 确定

问题代码:

    HCCL_ERROR("[HcclMemAlloc] GetAllocationGranularity failed, granularity[%llu], ret[%d]", granularity, ret);
    HCCL_INFO("[HcclMemAlloc] deviceId[%d], granularity[%llu], allocSize[%llu].", deviceId, granularity, allocSize);

`granularity` 和 `allocSize` 均为 `size_t` 类型。在 LP64 ABI 下，`size_t` = `unsigned long`（8 字节），`%llu` 期望 `unsigned long long`。二者是不同类型，属于未定义行为。已确认 `HCCL_ERROR`/`HCCL_INFO` 底层使用 printf 风格格式化（见 `log.h:68-76`）。

补充说明：代码库中 `%llu` 用于 `size_t` 是一种广泛存在的模式（grep 可见大量同类用法），在目标平台（LP64 Linux aarch64/x86_64）上二者均为 8 字节，实际不会产生运行时问题。但代码库中同样存在使用 `%zu` 的正确写法。

修复建议: 将 `%llu` 替换为 `%zu`：

    HCCL_ERROR("[HcclMemAlloc] GetAllocationGranularity failed, granularity[%zu], ret[%d]", granularity, ret);
    HCCL_INFO("[HcclMemAlloc] deviceId[%d], granularity[%zu], allocSize[%zu].", deviceId, granularity, allocSize);

全文件共 5 处需修改（第 39、41、45、51、57 行）。

---

### #4 [一般] 不相关的 HcclScatterInner 声明混入本 PR
- 位置: `inc/hccl/hccl_comm.h:291`
- 规则: PR 单一职责原则
- 置信度: 确定

问题代码:

    HcclResult HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
        HcclComm comm, aclrtStream stream);

该声明与 HcclMemAlloc/HcclMemFree 功能无关，且：(1) 缺少 `extern` 关键字（同文件其他函数均有）；(2) 缺少 doxygen 注释（同文件其他函数均有）；(3) 函数名含 "Inner" 暗示内部接口，不应出现在公共头文件中。

修复建议: 将该声明移至单独的 PR，或移至内部头文件。

---

### #5 [一般] if 关键字后缺少空格
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:50, 56`
- 规则: 1.2（格式规范：关键字与括号之间加空格）
- 置信度: 确定

问题代码:

    if(ret != ACL_SUCCESS) {

修复建议:

    if (ret != ACL_SUCCESS) {

---

### #6 [一般] ALIGN_SIZE 宏使用 GCC statement expression 且 tab 缩进
- 位置: `src/framework/common/src/hccl_mem_alloc.h:17`
- 规则: 2.1（C++14 标准兼容性）、1.2.1（缩进使用空格）
- 置信度: 较确定。已确认项目使用 C++14 标准（见 CLAUDE.md），GCC statement expression `({...})` 是 GCC 扩展，非标准 C++。

问题代码:

    #define ALIGN_SIZE(size, align) \
    	({ \
            (size) = (((size) + (align) - 1) / (align)) * (align);\
    	})

两个问题：(1) `({...})` 是 GCC statement expression，非标准 C++14；(2) 宏体内使用 tab 缩进，与项目的 4 空格缩进风格不一致。

修复建议: 移除不必要的 statement expression 包装，改用空格缩进：

    #define ALIGN_SIZE(size, align) \
        ((size) = (((size) + (align) - 1) / (align)) * (align))

---

### #7 [建议] 日志中 virPtr 标签与实际参数不匹配
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:45`
- 规则: 可维护性
- 置信度: 确定

问题代码:

    HCCL_ERROR("[HcclMemAlloc] ReserveMemAddress failed, "
        "virPtr[%p] size[%llu], ret[%d]", ptr, allocSize, ret)

日志标签写 `virPtr[%p]`，但传入的参数是 `ptr`（`void**`，输出参数的地址），不是虚拟内存地址。此时 `aclrtReserveMemAddress` 已失败，`*ptr` 值不可靠。

修复建议:

    HCCL_ERROR("[HcclMemAlloc] ReserveMemAddress failed, "
        "ptr[%p] size[%zu], ret[%d]", ptr, allocSize, ret)

---

### #8 [建议] 新文件末尾缺少换行符
- 位置: `src/framework/common/src/hccl_mem_alloc.cc:87`, `src/framework/common/src/hccl_mem_alloc.h:22`, `test/llt/ut/single_test/impl/ut_hccl_mem_alloc.cc:195`
- 规则: POSIX 文本文件规范
- 置信度: 确定

三个新增文件末尾均缺少换行符（diff 中可见 `\ No newline at end of file` 标记）。POSIX 标准要求文本文件以换行符结尾，缺少可能导致某些工具处理异常。

修复建议: 在每个文件末尾添加一个空行。

---

## 总结

本 PR 实现了 HcclMemAlloc/HcclMemFree 虚拟内存管理 API，整体思路清晰（预留地址→分配物理→映射），HcclMemAlloc 的错误处理路径（逆序清理）设计正确，UT 覆盖了主要的错误分支。

核心问题集中在两处：(1) HcclMemAlloc 入口缺少 `ptr` 空指针检查，作为公共 API 必须防御 null 输入；(2) HcclMemFree 使用 `CHK_PRT_RET` 导致中间步骤失败时跳过后续资源释放，应改为与 HcclMemAlloc 一致的 if-cleanup 模式。建议优先修复这 3 个严重问题。
