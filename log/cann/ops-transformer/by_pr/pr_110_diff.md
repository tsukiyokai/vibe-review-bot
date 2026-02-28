# PR #110: HcclMemAlloc

- 作者: linzhenkang
- 分支: temp -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/110
- 描述: new HcclMemAlloc and Free with test

## 变更文件 (9 个, 其中 C/C++ 文件 7 个)

- [modified] inc/hccl/hccl_comm.h (+19, -0) *
- [modified] src/framework/common/src/CMakeLists.txt (+1, -0)
- [added] src/framework/common/src/hccl_mem_alloc.cc (+87, -0) *
- [added] src/framework/common/src/hccl_mem_alloc.h (+22, -0) *
- [modified] test/llt/aicpu_kfc/stub/llt_aicpu_kfc_stub.cc (+26, -0) *
- [modified] test/llt/ut/single_test/impl/CMakeLists.txt (+1, -0)
- [added] test/llt/ut/single_test/impl/ut_hccl_mem_alloc.cc (+195, -0) *
- [modified] test/stub/framework_stub/llt_hccl_stub.cc (+26, -0) *
- [modified] test/ut/stub/llt_hccl_stub.cc (+26, -0) *

## Diff 内容

### inc/hccl/hccl_comm.h
```diff
@@ -288,6 +288,25 @@ extern HcclResult HcclSetCommConfig(HcclComm comm, HcclConfig config, HcclConfig
 extern HcclResult HcclGetCommConfig(HcclComm comm, HcclConfig config, HcclConfigValue *configValue);
 #endif
 
+HcclResult HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
+    HcclComm comm, aclrtStream stream);
+
+/**
+ * @brief Allocates virtual memory and maps it to physical memory.
+ *
+ * @param ptr A pointer to a void pointer that will receive the allocated virtual memory address.
+ * @param size The size of the virtual memory to allocate.
+ * @return HcclResult
+ */
+extern HcclResult HcclMemAlloc(void **ptr, size_t size);
+
+/**
+ * @brief Releases virtual memory and its mapped physical memory.
+ *
+ * @param ptr A pointer identifying the virtual memory address to be released.
+ * @return HcclResult
+ */
+extern HcclResult HcclMemFree(void *ptr);
 #ifdef __cplusplus
 }
 #endif // __cplusplus

```

### src/framework/common/src/hccl_mem_alloc.cc
```diff
@@ -0,0 +1,87 @@
+/*
+ * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
+ * This file is a part of the CANN Open Software.
+ * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
+ * Please refer to the License for details. You may not use this file except in compliance with the License.
+ * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
+ * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
+ * See LICENSE in the root of the software repository for the full text of the License.
+ */
+
+#include "hccl_mem_alloc.h"
+
+using namespace hccl;
+
+#ifdef __cplusplus
+extern "C" {
+#endif  // __cplusplus
+
+HcclResult HcclMemAlloc(void **ptr, size_t size)
+{
+    CHK_PRT_RET(size == 0, HCCL_ERROR("[HcclMemAlloc] size is zero"), HCCL_E_PARA);
+
+    aclError ret = ACL_SUCCESS;
+    int32_t deviceId;
+    ret = aclrtGetDevice(&deviceId);
+    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemAlloc] GetDevice failed, ret[%d]", ret), HCCL_E_RUNTIME);
+    aclrtPhysicalMemProp prop;
+    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
+    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
+    prop.memAttr = ACL_HBM_MEM_HUGE;
+    prop.location.id = deviceId;
+    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
+    prop.reserve = 0;
+
+    size_t allocSize = size;
+    size_t granularity = 0;
+    ret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
+    CHK_PRT_RET(ret != ACL_SUCCESS || granularity == 0,
+        HCCL_ERROR("[HcclMemAlloc] GetAllocationGranularity failed, granularity[%llu], ret[%d]", granularity, ret), HCCL_E_RUNTIME);
+    ALIGN_SIZE(allocSize, granularity);
+    HCCL_INFO("[HcclMemAlloc] deviceId[%d], granularity[%llu], allocSize[%llu].", deviceId, granularity, allocSize);
+
+    ret = aclrtReserveMemAddress(ptr, allocSize, 0, nullptr, 1);
+    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemAlloc] ReserveMemAddress failed, "
+        "virPtr[%p] size[%llu], ret[%d]", ptr, allocSize, ret), HCCL_E_RUNTIME);
+
+    void *virPtr = *ptr;
+    aclrtDrvMemHandle handle;
+    ret = aclrtMallocPhysical(&handle, allocSize, &prop, 0);
+    if(ret != ACL_SUCCESS) {
+        HCCL_ERROR("[HcclMemAlloc] MallocPhysical failed, size[%llu], ret[%d]", allocSize, ret);
+        aclrtReleaseMemAddress(virPtr);
+        return HCCL_E_RUNTIME;
+    }
+    ret = aclrtMapMem(virPtr, allocSize, 0, handle, 0);
+    if(ret != ACL_SUCCESS) {
+        HCCL_ERROR("[HcclMemAlloc] MapMem virPtr[%p] size[%llu] handle[%p] failed, ret[%d]", virPtr, allocSize, handle, ret);
+        aclrtFreePhysical(handle);
+        aclrtReleaseMemAddress(virPtr);
+        return HCCL_E_RUNTIME;
+    }
+
+    return HCCL_SUCCESS;
+}
+
+HcclResult HcclMemFree(void *ptr)
+{
+    if (ptr == nullptr) {
+        HCCL_DEBUG("[HcclMemFree] virPtr is nullptr.");
+        return HCCL_SUCCESS;
+    }
+    aclError ret = ACL_SUCCESS;
+    aclrtDrvMemHandle handle;
+    ret = aclMemRetainAllocationHandle(ptr, &handle);
+    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] RetainAllocationHandle virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
+    HCCL_INFO("[HcclMemFree] virPtr[%p], handle[%p]", ptr, handle);
+    ret = aclrtUnmapMem(ptr);
+    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] UnmapMem virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
+    ret = aclrtFreePhysical(handle);
+    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] FreePhysical handle[%p] failed, ret[%d]", handle, ret), HCCL_E_RUNTIME);
+    ret = aclrtReleaseMemAddress(ptr);
+    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] ReleaseMemAddress virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
+    return HCCL_SUCCESS;
+}
+#ifdef __cplusplus
+}
+#endif // __cplusplus
\ No newline at end of file

```

### src/framework/common/src/hccl_mem_alloc.h
```diff
@@ -0,0 +1,22 @@
+/*
+ * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
+ * This file is a part of the CANN Open Software.
+ * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
+ * Please refer to the License for details. You may not use this file except in compliance with the License.
+ * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
+ * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
+ * See LICENSE in the root of the software repository for the full text of the License.
+ */
+
+#ifndef HCCL_MEM_ALLOC_H
+#define HCCL_MEM_ALLOC_H
+
+#include <hccl_comm.h>
+#include "config.h"
+
+#define ALIGN_SIZE(size, align) \
+	({ \
+        (size) = (((size) + (align) - 1) / (align)) * (align);\
+	})
+
+#endif // HCCL_MEM_ALLOC_H
\ No newline at end of file

```

### test/llt/aicpu_kfc/stub/llt_aicpu_kfc_stub.cc
```diff
@@ -2114,6 +2114,32 @@ aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
     return ACL_SUCCESS;
 }
 
+aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags)
+{
+    (void)handle;
+    (void)size;
+    (void)prop;
+    (void)flags;
+    return ACL_SUCCESS;
+}
+
+aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *prop, aclrtMemGranularityOptions option, size_t *granularity)
+{
+    (void)prop;
+    (void)option;
+    if(granularity != nullptr) {
+        *granularity = 2097152;
+    }
+    return ACL_SUCCESS;
+}
+
+aclError aclMemRetainAllocationHandle(void *virPtr, aclrtDrvMemHandle *handle)
+{
+    (void)virPtr;
+    (void)handle;
+    return ACL_SUCCESS;
+}
+
 aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
 {
     (void)virPtr;

```

### test/llt/ut/single_test/impl/ut_hccl_mem_alloc.cc
```diff
@@ -0,0 +1,195 @@
+/**
+ * Copyright (c) 2025 Huawei Technologies Co., Ltd.
+ * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
+ * CANN Open Software License Agreement Version 2.0 (the "License").
+ * Please refer to the License for details. You may not use this file except in compliance with the License.
+ * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
+ * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
+ * See LICENSE in the root of the software repository for the full text of the License.
+ */
+
+#include "gtest/gtest.h"
+#include <mockcpp/mockcpp.hpp>
+#include <iostream>
+
+#define private public
+#define protected public
+#include "hccl_comm.h"
+#undef private
+#undef protected
+
+using namespace std;
+
+constexpr size_t TWO_M = 2097152;
+
+class MemAllocTest : public testing::Test {
+protected:
+    static void SetUpTestCase()
+    {
+        std::cout << "MemAllocTest Testcase SetUP" << std::endl;
+    }
+    static void TearDownTestCase()
+    {
+        std::cout << "MemAllocTest Testcase TearDown" << std::endl;
+    }
+    virtual void SetUp()
+    {
+        std::cout << "A MemAllocTest SetUP" << std::endl;
+    }
+    virtual void TearDown()
+    {
+        GlobalMockObject::verify();
+        std::cout << "A MemAllocTest TearDown" << std::endl;
+    }
+};
+
+TEST_F(MemAllocTest, ut_HcclMemAlloc_When_Normal_Expect_ReturnHCCL_SUCCESS)
+{
+    void *ptr = nullptr;
+    size_t size = TWO_M;
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemAlloc_When_SizeIsZero_Expect_ReturnHCCL_E_PARA)
+{
+    void *ptr = nullptr;
+    size_t size = 0;
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    EXPECT_EQ(ret, HCCL_E_PARA);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemAlloc_When_GetDeviceFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtGetDevice)
+    .stubs()
+    .will(returnValue(500000));
+
+    void *ptr = nullptr;
+    size_t size = TWO_M;
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemAlloc_When_GetGranularityFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtMemGetAllocationGranularity)
+    .stubs()
+    .will(returnValue(500000));
+
+    void *ptr = nullptr;
+    size_t size = TWO_M;
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemAlloc_When_GranularityIsZero_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtMemGetAllocationGranularity)
+    .stubs()
+    .will(returnValue(ACL_SUCCESS));
+
+    void *ptr = nullptr;
+    size_t size = TWO_M;
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemAlloc_When_ReserveMemAddressFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtReserveMemAddress)
+    .stubs()
+    .will(returnValue(500000));
+
+    void *ptr = nullptr;
+    size_t size = TWO_M + 1;            // 对齐测试
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemAlloc_When_MallocPhysicalFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtMallocPhysical)
+    .stubs()
+    .will(returnValue(500000));
+
+    void *ptr = nullptr;
+    size_t size = TWO_M;
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemAlloc_When_MapMemFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtMapMem)
+    .stubs()
+    .will(returnValue(500000));
+
+    void *ptr = nullptr;
+    size_t size = TWO_M;
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemFree_When_Normal_Expect_ReturnHCCL_SUCCESS)
+{
+    int temp = 0;
+    void *ptr = &temp;
+    HcclResult ret = HcclMemFree(ptr);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemFree_When_PtrIsNull_Expect_ReturnHCCL_SUCCESS)
+{
+    void *ptr = nullptr;
+    HcclResult ret = HcclMemFree(ptr);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemFree_When_RetainAllocationHandleFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclMemRetainAllocationHandle)
+    .stubs()
+    .will(returnValue(500000));
+
+    int temp = 0;
+    void *ptr = &temp;
+    HcclResult ret = HcclMemFree(ptr);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemFree_When_UnmapMemFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtUnmapMem)
+    .stubs()
+    .will(returnValue(500000));
+
+    int temp = 0;
+    void *ptr = &temp;
+    HcclResult ret = HcclMemFree(ptr);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemFree_When_FreePhysicalFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtFreePhysical)
+    .stubs()
+    .will(returnValue(500000));
+
+    int temp = 0;
+    void *ptr = &temp;
+    HcclResult ret = HcclMemFree(ptr);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
+
+TEST_F(MemAllocTest, ut_HcclMemFree_When_ReleaseMemAddressFailed_Expect_ReturnHCCL_E_RUNTIME)
+{
+    MOCKER(aclrtReleaseMemAddress)
+    .stubs()
+    .will(returnValue(500000));
+
+    int temp = 0;
+    void *ptr = &temp;
+    HcclResult ret = HcclMemFree(ptr);
+    EXPECT_EQ(ret, HCCL_E_RUNTIME);
+}
\ No newline at end of file

```

### test/stub/framework_stub/llt_hccl_stub.cc
```diff
@@ -1923,6 +1923,32 @@ aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
     return ACL_SUCCESS;
 }
 
+aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags)
+{
+    (void)handle;
+    (void)size;
+    (void)prop;
+    (void)flags;
+    return ACL_SUCCESS;
+}
+
+aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *prop, aclrtMemGranularityOptions option, size_t *granularity)
+{
+    (void)prop;
+    (void)option;
+    if(granularity != nullptr) {
+        *granularity = 2097152;
+    }
+    return ACL_SUCCESS;
+}
+
+aclError aclMemRetainAllocationHandle(void *virPtr, aclrtDrvMemHandle *handle)
+{
+    (void)virPtr;
+    (void)handle;
+    return ACL_SUCCESS;
+}
+
 aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
 {
     (void)virPtr;

```

### test/ut/stub/llt_hccl_stub.cc
```diff
@@ -2036,6 +2036,32 @@ aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
     return ACL_SUCCESS;
 }
 
+aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags)
+{
+    (void)handle;
+    (void)size;
+    (void)prop;
+    (void)flags;
+    return ACL_SUCCESS;
+}
+
+aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *prop, aclrtMemGranularityOptions option, size_t *granularity)
+{
+    (void)prop;
+    (void)option;
+    if(granularity != nullptr) {
+        *granularity = 2097152;
+    }
+    return ACL_SUCCESS;
+}
+
+aclError aclMemRetainAllocationHandle(void *virPtr, aclrtDrvMemHandle *handle)
+{
+    (void)virPtr;
+    (void)handle;
+    return ACL_SUCCESS;
+}
+
 aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
 {
     (void)virPtr;

```

> 注: 以下非 C/C++ 文件未纳入审查: src/framework/common/src/CMakeLists.txt, test/llt/ut/single_test/impl/CMakeLists.txt
