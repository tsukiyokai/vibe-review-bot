# PR #895: Support symmetric memory for aicpu unflod mode

- 作者: linzhenkang
- 分支: pr_663 -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/895
- 描述: Support symmetric memory for aicpu unflod mode

## 变更文件 (47 个, 其中 C/C++ 文件 41 个)

- [modified] include/hccl/hccl_comm.h (+33, -0) *
- [modified] include/hccl/hccl_types.h (+16, -1) *
- [modified] include/hccl/hcomm_primitives.h (+12, -0) *
- [modified] src/algorithm/impl/operator/all_gather_operator.cc (+3, -3) *
- [modified] src/algorithm/impl/operator/all_reduce_operator.cc (+4, -4) *
- [modified] src/algorithm/impl/operator/reduce_scatter_operator.cc (+6, -6) *
- [modified] src/algorithm/pub_inc/coll_alg_param.h (+5, -0) *
- [modified] src/framework/common/src/CMakeLists.txt (+1, -0)
- [added] src/framework/common/src/hccl_mem_alloc.cc (+89, -0) *
- [added] src/framework/common/src/hccl_mem_alloc.h (+33, -0) *
- [modified] src/framework/common/src/launch_aicpu.cc (+1, -0) *
- [modified] src/framework/communicator/comm_config.cc (+22, -9) *
- [modified] src/framework/communicator/hccl_comm.cc (+22, -0) *
- [modified] src/framework/communicator/impl/CMakeLists.txt (+2, -1)
- [modified] src/framework/communicator/impl/aclgraph/zero_copy_acl_graph.cc (+1, -1) *
- [modified] src/framework/communicator/impl/hccl_communicator.cc (+4, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator.h (+8, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator_device.cc (+25, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator_host.cc (+92, -1) *
- [added] src/framework/communicator/impl/symmetric_memory/CMakeLists.txt (+8, -0)
- [added] src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc (+603, -0) *
- [added] src/framework/communicator/impl/symmetric_memory/symmetric_memory.h (+134, -0) *
- [added] src/framework/communicator/impl/symmetric_memory/symmetric_memory_agent.cc (+249, -0) *
- [added] src/framework/communicator/impl/symmetric_memory/symmetric_memory_agent.h (+101, -0) *
- [modified] src/framework/device/framework/CMakeLists.txt (+1, -0)
- [modified] src/framework/device/framework/aicpu_communicator.cc (+104, -8) *
- [modified] src/framework/device/framework/aicpu_communicator.h (+7, -0) *
- [modified] src/framework/device/framework/aicpu_hccl_process.cc (+5, -0) *
- [added] src/framework/device/framework/aicpu_symmetric_memory.cc (+47, -0) *
- [added] src/framework/device/framework/aicpu_symmetric_memory.h (+15, -0) *
- [modified] src/framework/device/hccl_aicpu_interface.cc (+3, -1) *
- [modified] src/framework/inc/comm_config_pub.h (+5, -1) *
- [modified] src/framework/inc/hccl_comm_pub.h (+4, -0) *
- [modified] src/framework/op_base/src/op_base.cc (+65, -0) *
- [modified] src/platform/common/externalinput.cc (+1, -1) *
- [modified] src/pub_inc/aicpu_operator_pub.h (+5, -0) *
- [modified] test/ut/depends/include/acl/acl_rt.h (+12, -0) *
- [modified] test/ut/framework/communicator/impl/CMakeLists.txt (+2, -1)
- [added] test/ut/framework/communicator/impl/symmetric_memory/CMakeLists.txt (+62, -0)
- [added] test/ut/framework/communicator/impl/symmetric_memory/main.cc (+19, -0) *
- [added] test/ut/framework/communicator/impl/symmetric_memory/ut_aicpu_communicator.cc (+129, -0) *
- [added] test/ut/framework/communicator/impl/symmetric_memory/ut_allgather_manager.cc (+301, -0) *
- [added] test/ut/framework/communicator/impl/symmetric_memory/ut_hccl_communicator_host.cc (+153, -0) *
- [added] test/ut/framework/communicator/impl/symmetric_memory/ut_hccl_mem_alloc.cc (+196, -0) *
- [added] test/ut/framework/communicator/impl/symmetric_memory/ut_symmetric_memory.cc (+1245, -0) *
- [modified] test/ut/framework/communicator/main.cc (+1, -1) *
- [modified] test/ut/stub/llt_hccl_stub.cc (+67, -0) *

## Diff 内容

### include/hccl/hccl_comm.h
```diff
@@ -220,6 +220,7 @@ inline void HcclCommConfigInit(HcclCommConfig *config)
     config->hcclRetryEnable[0] = '\0';
     config->hcclRetryParams[0] = '\0';
     config->hcclBufferName[0] = '\0';
+    config->hcclSymWinMaxMemSizePerRank = HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE;
 }
 
 /**
@@ -288,6 +289,38 @@ extern HcclResult HcclGroupStart();
  */
 extern HcclResult HcclGroupEnd();
 
+/**
+ * @brief Register a memory window for HCCL communication.
+ *
+ * @param comm A pointer identifying the communication resource based on.
+ * @param addr A pointer identifying the user memory address.
+ * @param size A size_t identifying the size of memory window.
+ * @param winHandle A pointer identifying the registered memory window handle.
+ * @param flag The flag of this memory window, now only support 0
+ * @return HcclResult
+ */
+extern HcclResult HcclCommSymWinRegister(HcclComm comm, void *addr, uint64_t size, CommSymWindow *winHandle, uint32_t flag);
+
+/**
+ * @brief Deregister a memory window for HCCL communication.
+ *
+ * @param winHandle A pointer identifying the registered memory window handle.
+ * @return HcclResult
+ */
+extern HcclResult HcclCommSymWinDeregister(CommSymWindow winHandle);
+
+/**
+ * @brief Get symmetric memory offset and window for HCCL communication.
+ *
+ * @param comm A pointer identifying the communication resource based on.
+ * @param ptr A pointer identifying the user memory address.
+ * @param size A size_t identifying the size of memory window.
+ * @param winHandle A pointer identifying the registered memory window handle.
+ * @param offset A size_t identifying the offset of symmetric memory heap.
+ * @return HcclResult
+ */
+extern HcclResult HcclCommSymWinGet(HcclComm comm, void *ptr, size_t size, CommSymWindow *winHandle, size_t *offset);
+
 #ifdef __cplusplus
 }
 #endif // __cplusplus

```

### include/hccl/hccl_types.h
```diff
@@ -60,6 +60,19 @@ typedef void *HcclComm;
  */
 typedef void *HcclConn;
 
+/**
+ * @brief handle to HCCL Window
+ */
+typedef void *CommSymWindow;
+
+/**
+ * @brief Symmetric Memory Flag
+ */
+typedef enum {
+    HCCL_WIN_DEFAULT = 0,       /**< 先不支持，预留 */
+    HCCL_WIN_COLL_SYMMETRIC = 1 /**< 启用对称内存 */
+} symmetricMemoryFlag;
+
 /**
  * @brief HCCL Reduction operation
  */
@@ -120,7 +133,7 @@ typedef struct HcclRootInfoDef {
 
 const uint32_t HCCL_COMM_CONFIG_INFO_BYTES = 24;
 const uint32_t HCCL_COMM_CONFIG_MAGIC_WORD = 0xf0f0f0f0;
-const uint32_t HCCL_COMM_CONFIG_VERSION = 9;
+const uint32_t HCCL_COMM_CONFIG_VERSION = 10;
 const uint32_t HCCL_COMM_DEFAULT_BUFFSIZE = 200;
 const uint32_t HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET = 0xffffffff;
 const uint32_t HCCL_COMM_DEFAULT_DETERMINISTIC = 0;
@@ -130,6 +143,7 @@ const uint32_t HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0;
 const uint32_t HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xffffffff;
 const uint32_t HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xffffffff;
 const int32_t HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET = 0xffffffff;
+const uint64_t HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE = 16ULL;
 
 typedef struct HcclCommConfigDef {
     char reserved[HCCL_COMM_CONFIG_INFO_BYTES];
@@ -148,6 +162,7 @@ typedef struct HcclCommConfigDef {
     char hcclRetryEnable[HCCL_COMM_RETRY_ENABLE_MAX_LENGTH];
     char hcclRetryParams[HCCL_COMM_RETRY_PARAMS_MAX_LENGTH];
     char hcclBufferName[BUFFER_NAME_MAX_LENGTH];
+    uint64_t hcclSymWinMaxMemSizePerRank; // 对称内存预留VA大小, 单位GB
 } HcclCommConfig;
 
 typedef enum {

```

### include/hccl/hcomm_primitives.h
```diff
@@ -15,6 +15,7 @@
 #include <securec.h>
 #include <arpa/inet.h>
 #include "acl/acl_rt.h"
+#include <hccl_types.h>
 
 #ifdef __cplusplus
 extern "C" {
@@ -342,6 +343,17 @@ extern int32_t HcommAcquireComm(const char* commId);
  */
 extern int32_t HcommReleaseComm(const char* commId);
 
+/**
+ * @brief Get symmetric memory pointer.
+ *
+ * @param winHandle A pointer identifying the registered memory window handle.
+ * @param offset A size_t identifying the offset of symmetric memory heap.
+ * @param peerRank A integer identifying the identify for the peer rank.
+ * @param ptr A pointer identifying the symmetric memory heap address.
+ * @return HcclResult
+ */
+extern HcclResult HcommSymWinGetPeerPointer(CommSymWindow winHandle, size_t offset, int peerRank, void** ptr);
+
 #define HCOMM_PRIMITIVES_H_MODIFIED
 
 

```

### src/algorithm/impl/operator/all_gather_operator.cc
```diff
@@ -350,10 +350,10 @@ HcclResult AllGatherOperator::SelectAlgfor91093(const OpParam& param, std::strin
     } else if (smallCountOptimMultiPod) {
         algName = "AllGatherComm";
         algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
-    } else if (smallCountOptimMultiServer || smallCountOptimSingleServer) {
+    } else if (!param.supportSymmetricMemory && (smallCountOptimMultiServer || smallCountOptimSingleServer)) {
         algName = "AllGatherSmallCount";
-    } else if (param.supportZeroCopy &&
-        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB)) {
+    } else if (param.supportSymmetricMemory || (param.supportZeroCopy &&
+        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB))) {
         const u32 SEVER_NUM_FOUR = 4;
         constexpr u64 RING_EXCHANGE_PIPELINE_DATA_SIZE_MIN = 2 * 1024 * 1024;
         HcclAlgoType configAlgTypeLevel2 = topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLGATHER)[HCCL_ALGO_LEVEL_2];

```

### src/algorithm/impl/operator/all_reduce_operator.cc
```diff
@@ -663,13 +663,13 @@ HcclResult AllReduceOperator::SelectAlgfor91093(const OpParam& param, std::strin
         algName = "AllReduceComm";
     } else if (multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffDeviceNumMode_) {
         algName = "AllReduceARSFor91093Executor";
-    } else if (useHostComm || smallCountOptimMultiServer || smallCountOptimMultiPod) {
+    } else if (!param.supportSymmetricMemory && (useHostComm || smallCountOptimMultiServer || smallCountOptimMultiPod)) {
         algName = "AllReduceComm";
         algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
-    } else if (smallCountOptimSingleServer) {
+    } else if (!param.supportSymmetricMemory && smallCountOptimSingleServer) {
         algName = "AllReduceMeshSmallCountExecutor";
-    } else if (param.supportZeroCopy &&
-        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize > HCCL_MID_COUNT_16_MB * serverNum_)) {
+    } else if (param.supportSymmetricMemory || (param.supportZeroCopy &&
+        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize > HCCL_MID_COUNT_16_MB * serverNum_))) {
         algName = "AllReduceRingZerocopyExecutor";
     } else {
         if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {

```

### src/algorithm/impl/operator/reduce_scatter_operator.cc
```diff
@@ -470,16 +470,16 @@ HcclResult ReduceScatterOperator::SelectAlgfor91093(const OpParam& param, std::s
          algName = "ReduceScatterComm";
     } else if (multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffDeviceNumMode_) {
         algName = "ReduceScatterARSFor91093Executor";
-    } else if (smallCountOptimMultiPod || useHostComm || (smallCountOptimMultiServer && !isPowOfTwo &&
-        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_256_KB))) {
+    } else if (!param.supportSymmetricMemory && (smallCountOptimMultiPod || useHostComm || (smallCountOptimMultiServer && !isPowOfTwo &&
+        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_256_KB)))) {
         algName = "ReduceScatterComm";
         algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
-    } else if (smallCountOptimSingleServer ||
+    } else if (!param.supportSymmetricMemory && (smallCountOptimSingleServer ||
         (smallCountOptimMultiServer && isPowOfTwo &&
-        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] * serverNum_ <= smallCountMultiServerThreshold))) {
+        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] * serverNum_ <= smallCountMultiServerThreshold)))) {
         algName = "ReduceScatterDeterExecutor";
-    } else if (param.supportZeroCopy && isSupportInlineReduce &&    // 不申请scratch ==> 不支持非InlineReduce
-        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB)) {
+    } else if (isSupportInlineReduce && (param.supportSymmetricMemory || (param.supportZeroCopy &&    // isSupportInlineReduce：不申请scratch ==> 不支持非InlineReduce
+        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB)))) {
         const u32 SEVER_NUM_FOUR = 4;
         constexpr u64 RING_EXCHANGE_PIPELINE_DATA_SIZE_MIN = 2 * 1024 * 1024;
         HcclAlgoType configAlgTypeLevel2 = topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)[HCCL_ALGO_LEVEL_2];

```

### src/algorithm/pub_inc/coll_alg_param.h
```diff
@@ -213,6 +213,11 @@ struct OpParam {
     u32 nRecv = 0;
     u32 iSend = 0; // index of send
     u32 iRecv = 0; // index of recv
+    bool supportSymmetricMemory = false;
+    void* inputSymWindow = nullptr;
+    u64 inputOffset = 0;
+    void* outputSymWindow = nullptr;
+    u64 outputOffset = 0;
 
     inline HcclDataType GetDataType() const
     {

```

### src/framework/common/src/hccl_mem_alloc.cc
```diff
@@ -0,0 +1,89 @@
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
+#include "hccl_mem_alloc.h"
+using namespace hccl;
+
+#ifdef __cplusplus
+extern "C" {
+#endif  // __cplusplus
+
+HcclResult HcclMemAlloc(void **ptr, size_t size)
+{
+    CHK_PTR_NULL(ptr);
+    CHK_PRT_RET(size == 0, HCCL_ERROR("[HcclMemAlloc] size is zero"), HCCL_E_PARA);
+
+    aclError ret = ACL_SUCCESS;
+    int32_t deviceId;
+    ret = aclrtGetDevice(&deviceId);
+    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemAlloc] GetDevice failed, ret[%d]", ret), HCCL_E_RUNTIME);
+
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
+    HCCL_INFO("[HcclMemAlloc] deviceId[%d], granularity[%llu], size[%llu], allocSize[%llu].", deviceId, granularity, size, allocSize);
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
+    HCCL_INFO("[HcclMemAlloc]Start to MapMem virPtr[%p], handle[%p]", virPtr, handle);
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
+    ret = aclrtMemRetainAllocationHandle(ptr, &handle);
+    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcclMemFree] RetainAllocationHandle virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
+    HCCL_INFO("[HcclMemFree]Start to UnmapMem virPtr[%p], handle[%p]", ptr, handle);
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
@@ -0,0 +1,33 @@
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
+#ifndef HCCL_MEM_ALLOC_H
+#define HCCL_MEM_ALLOC_H
+
+#include <hccl_comm.h>
+#include "hccl_comm_pub.h"
+#include "config.h"
+
+#define ALIGN_SIZE(size, align) \
+    ({ \
+        (size) = (((size) + (align) - 1) / (align)) * (align);\
+    })
+
+#ifdef __cplusplus
+extern "C" {
+#endif  // __cplusplus
+
+HcclResult HcclMemAlloc(void **ptr, size_t size);
+HcclResult HcclMemFree(void *ptr);
+
+#ifdef __cplusplus
+}
+#endif  // __cplusplus
+#endif // HCCL_MEM_ALLOC_H
\ No newline at end of file

```

### src/framework/common/src/launch_aicpu.cc
```diff
@@ -11,6 +11,7 @@
 #include <iostream>
 #include <fstream>
 #include <string>
+#include <cstdint>
 #include "launch_aicpu.h"
 #include "log.h"
 #include "env_config.h"

```

### src/framework/communicator/comm_config.cc
```diff
@@ -33,7 +33,8 @@ CommConfig::CommConfig(const std::string &commName)
       retryMaxCnt_(GetExternalInputRetryMaxCnt()),
       retryHoldTime_(GetExternalInputRetryHoldTime()),
       retryIntervalTime_(GetExternalInputRetryIntervalTime()),
-      bufferName_("")
+      bufferName_(""),
+      symmetricMemoryStride_(HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE)
 {
     InitAlgoConfig();
     InitRetryEnable();
@@ -54,7 +55,9 @@ CommConfig::CommConfig()
       execTimeOutSetByConfig_(false),
       retryMaxCnt_(GetExternalInputRetryMaxCnt()),
       retryHoldTime_(GetExternalInputRetryHoldTime()),
-      retryIntervalTime_(GetExternalInputRetryIntervalTime())
+      retryIntervalTime_(GetExternalInputRetryIntervalTime()),
+      bufferName_(""),
+      symmetricMemoryStride_(HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE)
 {
     InitAlgoConfig();
     InitRetryEnable();
@@ -113,8 +116,8 @@ HcclResult CommConfig::Load(const HcclCommConfig *userConfig)
     HCCL_RUN_INFO("[Load] comm config info of [%s]: configSize[%llu], version[%u], opExpansionMode[%u]", commName_.c_str(),
         configHandle.info.configSize, configHandle.info.version, configHandle.opExpansionMode);
     HCCL_RUN_INFO("[Load] comm config of [%s]: bufferSize[%llu], deterministic[%u], trafficClass[%u], serviceLevel[%u]"
-        ", execTimeOut[%u]s, bufferName[%s]",
-        commName_.c_str(), bufferSize_, deterministic_, trafficClass_, serviceLevel_, execTimeOut_, bufferName_.c_str());
+        ", execTimeOut[%u]s, bufferName[%s], symmetricMemoryStride[%llu]",
+        commName_.c_str(), bufferSize_, deterministic_, trafficClass_, serviceLevel_, execTimeOut_, bufferName_.c_str(), symmetricMemoryStride_);
     return HCCL_SUCCESS;
 }
 
@@ -139,18 +142,18 @@ HcclResult CommConfig::CheckMagicWord(const CommConfigHandle &config)
 
 HcclResult CommConfig::SetConfigByVersion(const CommConfigHandle &config)
 {
-    if (config.info.version > CommConfigVersion::COMM_CONFIG_VERSION_EIGHT) {
+    if (config.info.version > CommConfigVersion::COMM_CONFIG_VERSION_TEN) {
         // 传入的config的版本高于当前版本，警告不支持的配置项将被忽略
         HCCL_WARNING("[SetConfigByVersion] The version of provided config[%u] is higher than the current version[%u], "
             "unsupported configuration will be ignored.",
             config.info.version,
-            CommConfigVersion::COMM_CONFIG_VERSION_EIGHT);
-    } else if (config.info.version < CommConfigVersion::COMM_CONFIG_VERSION_EIGHT) {
+            CommConfigVersion::COMM_CONFIG_VERSION_TEN);
+    } else if (config.info.version < CommConfigVersion::COMM_CONFIG_VERSION_TEN) {
         // 传入的config的版本低于当前版本，警告高版本支持的配置项将被忽略
         HCCL_WARNING("[SetConfigByVersion] The version of provided config[%u] is lower than the current version[%u], "
             "configurations supported by later versions will be ignored.",
             config.info.version,
-            CommConfigVersion::COMM_CONFIG_VERSION_EIGHT);
+            CommConfigVersion::COMM_CONFIG_VERSION_TEN);
     }
 
     if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_ONE) {
@@ -215,6 +218,11 @@ HcclResult CommConfig::SetConfigByVersion(const CommConfigHandle &config)
         // 版本大于等于9
         CHK_RET(SetConfigBufferName(config));
     }
+
+    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_TEN) {
+        // 版本大于等于10，支持配置对称内存每个rank的预留VA大小
+        symmetricMemoryStride_ = config.symmetricMemoryStride;
+    }
     HCCL_INFO("NSLBDP-VERSION config.info.version = [%u] .", config.info.version);
     return HCCL_SUCCESS;
 }
@@ -623,7 +631,7 @@ HcclResult CommConfig::SetSpecificAlgTypeConfig(std::vector<std::string> &algos)
         algoConfig_[HcclCMDType::HCCL_CMD_ALLTOALL];
     return HCCL_SUCCESS;
 }
- 
+
 HcclResult CommConfig::SetConfigExecTimeOut(s32 execTimeOut)
 {
     execTimeOut_ = execTimeOut;
@@ -744,4 +752,9 @@ const std::string& CommConfig::GetConfigBufferName() const
 {
     return bufferName_;
 }
+
+u64 CommConfig::GetConfigSymmetricMemoryStride() const
+{
+    return symmetricMemoryStride_;
+}
 }
\ No newline at end of file

```

### src/framework/communicator/hccl_comm.cc
```diff
@@ -1500,4 +1500,26 @@ bool hcclComm::IsCommunicatorV2()
     }
     return false;
 }
+
+HcclResult hcclComm::RegisterWindow(void* ptr, size_t size, CommSymWindow *winHandle)
+{
+    CHK_SMART_PTR_NULL(communicator_);
+    CHK_RET(communicator_->RegisterWindow(ptr, size, winHandle));
+    return HCCL_SUCCESS;
+}
+
+HcclResult hcclComm::DeregisterWindow(CommSymWindow winHandle)
+{
+    CHK_SMART_PTR_NULL(communicator_);
+    CHK_RET(communicator_->DeregisterWindow(winHandle));
+    return HCCL_SUCCESS;
+}
+
+HcclResult hcclComm::GetCommSymWin(void* ptr, size_t size, CommSymWindow *winHandle, size_t *offset)
+{
+    CHK_SMART_PTR_NULL(communicator_);
+    CHK_RET(communicator_->GetCommSymWin(ptr, size, winHandle, offset));
+    return HCCL_SUCCESS;
+}
+
 }  // namespace hccl

```

### src/framework/communicator/impl/aclgraph/zero_copy_acl_graph.cc
```diff
@@ -70,7 +70,7 @@ bool ZeroCopyAclGraph::SetAclGraphZeroCopyMode(
             deviceType);
         return false;
     }
-    if (opParam.isZeroCopy || opParam.supportZeroCopy) {
+    if (opParam.isZeroCopy || opParam.supportZeroCopy || opParam.supportSymmetricMemory) {
         HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl can't support graph zero copy mode and operator "
                   "zero copy at the same time.");
         return false;

```

### src/framework/communicator/impl/hccl_communicator.cc
```diff
@@ -1703,6 +1703,10 @@ namespace hccl
         opTilingData->isInplacePreSync = static_cast<u8>(isInplacePreSync_);
         opTilingData->isPostSync = static_cast<u8>(isPostSync_);
         opTilingData->userStreamId = opParam.stream.id();
+        opTilingData->inputSymWindow = reinterpret_cast<u64>(opParam.inputSymWindow);
+        opTilingData->inputOffset = opParam.inputOffset;
+        opTilingData->outputSymWindow = reinterpret_cast<u64>(opParam.outputSymWindow);
+        opTilingData->outputOffset = opParam.outputOffset;
         return HCCL_SUCCESS;
     }
 

```

### src/framework/communicator/impl/hccl_communicator.h
```diff
@@ -56,6 +56,7 @@
 #include "comm_config_pub.h"
 #include "new/hccl_dispatcher_ctx.h"
 #include "rank_graph.h"
+#include "symmetric_memory/symmetric_memory.h"
 
 namespace hccl {
 using ServRankInfo_t = std::map<std::string, std::vector<RankInfo_t> >;
@@ -477,6 +478,10 @@ public:
     HcclResult GroupSyncMainstream(std::unordered_map<u32, std::vector<u64>> &sendIdx2Byte, std::unordered_map<u32, std::vector<u64>> &recvIdx2Byte);
     HcclResult GroupSubstreamsSync();
     void SetReleaseChannel(std::function<HcclResult()> releaseChannel);
+    HcclResult RegisterWindow(void* ptr, size_t size, CommSymWindow *winHandle);
+    HcclResult DeregisterWindow(CommSymWindow winHandle);
+    HcclResult InitSymmetricMemory();
+    HcclResult GetCommSymWin(void* ptr, size_t size, CommSymWindow *winHandle, size_t *offset);
 private:
 
     bool IsEnableRoce();
@@ -776,6 +781,7 @@ private:
     bool GetSupportHDCommunicate();
     HcclResult InitOpRetry();
     HcclResult InitOpResPara();
+    bool IsSupportSymmetricMemory(OpParam &opParam);
     bool IsSupportZeroCopy(const OpParam &opParam);
     HcclResult PrepareZeroCopy(const std::string &algName, const AlgDesc &algDesc, OpParam &opParam);
     HcclResult UpdateZeroCopy(const OpParam &opParam, const AlgResourceResponse &algResource);
@@ -1115,6 +1121,8 @@ private:
     std::function<bool()> getAicpuCommState_; // 获取自定义算子aicpu通信域是否初始化
     bool isInvalidComm_ { false };
     std::function<HcclResult()> releaseChannel_ = nullptr;
+    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_;
+    std::unique_ptr<SymmetricMemory> symmetricMemory_;
 };
 }  // end namespace hccl
 #endif  // HCCL_IMPL_BASE_H

```

### src/framework/communicator/impl/hccl_communicator_device.cc
```diff
@@ -142,6 +142,11 @@ namespace hccl
         return HCCL_SUCCESS;
     }
 
+    bool HcclCommunicator::IsSupportSymmetricMemory(OpParam &opParam)
+    {
+        return false;
+    }
+
     bool HcclCommunicator::IsSupportZeroCopy(const OpParam &opParam)
     {
         return false;
@@ -1590,4 +1595,24 @@ namespace hccl
     {
         return cclBufferManager_;
     }
+
+    HcclResult HcclCommunicator::InitSymmetricMemory()
+    {
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, CommSymWindow *winHandle)
+    {
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult HcclCommunicator::DeregisterWindow(CommSymWindow winHandle)
+    {
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult HcclCommunicator::GetCommSymWin(void* ptr, size_t size, CommSymWindow *winHandle, size_t *offset)
+    {
+        return HCCL_SUCCESS;
+    }
 }

```

### src/framework/communicator/impl/hccl_communicator_host.cc
```diff
@@ -51,6 +51,7 @@
 #include "snapshot_control.h"
 #include "comm_topo_desc.h"
 #include "rt_external.h"
+#include "externalinput.h"
 
 using namespace std;
 constexpr u32 MODULE_NUM_FOUR = 4;
@@ -73,6 +74,7 @@ namespace hccl
     constexpr u32 SINGLE_PROCESS_MAX_PORT = 65535;
     constexpr u32 TYPE_USER_MEM = 1;
     constexpr u32 NON_BATCH_WRITE_MAX_STREAM_NUM = 19U;
+    constexpr u64 GIGABYTE_TO_BYTE = 1024ULL * 1024ULL * 1024ULL;
     enum TransferMemInfoIdx
     {
         TRANSFER_MEM_INFO_KEY_IDX = 0,
@@ -373,6 +375,7 @@ namespace hccl
         if (deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_93){
             CHK_RET(RegisterToSnapshot());
         }
+        CHK_RET(InitSymmetricMemory());
         return HCCL_SUCCESS;
     }
 
@@ -401,6 +404,7 @@ namespace hccl
         attrCollector_.GetTopoAttr(topoAttr);
         CHK_RET(rankGraph_.Init(topoAttr));
         CHK_RET(SaveTopoDesc(params.identifier));
+        CHK_RET(InitSymmetricMemory());
         return HCCL_SUCCESS;
     }
 
@@ -583,6 +587,42 @@ namespace hccl
         return HCCL_SUCCESS;
     }
 
+    bool HcclCommunicator::IsSupportSymmetricMemory(OpParam &opParam)
+    {
+        HCCL_INFO("[%s] aicpuUnfold[%d], workflowMode[%d], deviceType[%d], "
+            "deviceNumPerAggregation_[%d], multiModuleDiffDeviceNumMode_[%d], tag[%s].",
+            __func__, opParam.aicpuUnfoldMode, GetWorkflowMode(), deviceType_,
+            deviceNumPerAggregation_, multiModuleDiffDeviceNumMode_, opParam.tag.c_str());
+
+        // 只支持aicpu展开、单算子模式、910_93芯片
+        CHK_PRT_RET(!opParam.aicpuUnfoldMode,
+                    HCCL_INFO("[%s] aicpuUnfold:%d not support symmetric memory", __func__, opParam.aicpuUnfoldMode), false);
+        CHK_PRT_RET(GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE,
+                    HCCL_INFO("[%s] workflowMode:%d not support symmetric memory", __func__, GetWorkflowMode()), false);
+        CHK_PRT_RET(deviceType_ != DevType::DEV_TYPE_910_93,
+                    HCCL_INFO("[%s] deviceType:%d not support symmetric memory", __func__, deviceType_), false);
+
+        // 判断拓扑逻辑是否支持symmetric memory
+        // 每个节点只有一张卡或节点间非对称场景不支持对称内存
+        CHK_PRT_RET(deviceNumPerAggregation_ == 1 || multiModuleDiffDeviceNumMode_,
+                    HCCL_INFO("[%s] deviceNumPerAggregation[%u], multiModuleDiffDeviceNumMode_[%d] not support symmetric memory",
+                              __func__, deviceNumPerAggregation_, multiModuleDiffDeviceNumMode_),
+                    false);
+
+        // 判断输入输出地址是否都注册为对称内存
+        HcclResult ret = symmetricMemory_->FindSymmetricWindow(opParam.inputPtr, opParam.inputSize, &opParam.inputSymWindow, &opParam.inputOffset);
+        CHK_PRT_RET(ret != HCCL_SUCCESS || opParam.inputSymWindow == nullptr,
+                    HCCL_INFO("[%s] input[%p] size[%llu] is not support symmetric memory", __func__, opParam.inputPtr, opParam.inputSize), false);
+        ret = symmetricMemory_->FindSymmetricWindow(opParam.outputPtr, opParam.outputSize, &opParam.outputSymWindow, &opParam.outputOffset);
+        CHK_PRT_RET(ret != HCCL_SUCCESS || opParam.outputSymWindow == nullptr,
+                    HCCL_INFO("[%s] output[%p] size[%llu] is not support symmetric memory", __func__, opParam.outputPtr, opParam.outputSize), false);
+        
+        HCCL_INFO("[HcclCommunicator][IsSupportSymmetricMemory] opParam.inputPtr[%p], inputOffset[%llu], inputSymWindow[%p]", opParam.inputPtr, opParam.inputOffset, opParam.inputSymWindow);
+        HCCL_INFO("[HcclCommunicator][IsSupportSymmetricMemory] opParam.outputPtr[%p], outputOffset[%llu], outputSymWindow[%p]", opParam.outputPtr, opParam.outputOffset, opParam.outputSymWindow);
+
+        return true;
+    }
+
     bool HcclCommunicator::IsSupportZeroCopy(const OpParam &opParam)
     {
         HCCL_INFO("[%s] aicpuUnfold[%d], workflowMode[%d], deviceType[%d], "
@@ -617,9 +657,16 @@ namespace hccl
     HcclResult HcclCommunicator::PrepareZeroCopy(const std::string &algName, const AlgDesc &algDesc, OpParam &opParam)
     {
         if (!algDesc.isZeroCopy) {
+            opParam.supportSymmetricMemory = false;     //  对称内存选择零拷贝算法，若未选择零拷贝算法，对称内存使能关闭，确保aicpu侧不走对称内存分支
             HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] algName[%s] not support zerocopy.", algName.c_str());
             return HCCL_SUCCESS;
         }
+
+        if (opParam.supportSymmetricMemory) {
+            HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] algName[%s] symmetric memory is enabled, not use zerocopy.",
+                      algName.c_str());
+            return HCCL_SUCCESS;
+        }
         // ARS特性不支持零拷贝
         if ((opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opParam.opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
                 opParam.opType == HcclCMDType::HCCL_CMD_ALLREDUCE) && deviceType_ == DevType::DEV_TYPE_910_93 && 
@@ -4304,7 +4351,8 @@ namespace hccl
         }
 
         ForceProf(opParam.isCapture);
-        opParam.supportZeroCopy = !commConfig_.GetConfigDeterministic() && IsSupportZeroCopy(opParam);
+        opParam.supportSymmetricMemory = IsSupportSymmetricMemory(opParam);
+        opParam.supportZeroCopy = !opParam.supportSymmetricMemory && !commConfig_.GetConfigDeterministic() && IsSupportZeroCopy(opParam);
         opParam.aclGraphZeroCopyEnable = GetConfigAclGraphZeroCopyEnable();
         bool isInGraphCaptureZeroCopy = false;
         zeroCopyAclGraph_->SetRetryEnable(retryEnable_);
@@ -7005,6 +7053,7 @@ namespace hccl
         opTilingData->isZeroCopy = opParam.isZeroCopy;
         opTilingData->isCapture = opParam.isCapture;
         opTilingData->orderLaunchMode = GetOrderLaunchMode(opParam.isCapture);
+        opTilingData->isSymmetricMemory = opParam.supportSymmetricMemory;
         // 有没有存在对应的Notify
         CHK_RET(InitAndCheckAicpuOrderNotify(opTilingData->orderLaunchMode));
         CHK_RET(BuildHierarchicalAlgOption(opTilingData->ahcConfInfo));
@@ -8815,4 +8864,46 @@ namespace hccl
     {
         return cclBufferManager_;
     }
+
+    HcclResult HcclCommunicator::InitSymmetricMemory()
+    {
+        if (superPodNum_ > 1) {
+            HCCL_DEBUG("[InitSymmetricMemory] Cross-SuperNode not support symmetric memory");
+            return HCCL_SUCCESS;
+        }
+        if (deviceType_ != DevType::DEV_TYPE_910_93) {
+            HCCL_DEBUG("[%s] deviceType:%d not support symmetric memory", __func__, deviceType_);
+            return HCCL_SUCCESS;
+        }
+        
+        u64 stride = commConfig_.GetConfigSymmetricMemoryStride() * GIGABYTE_TO_BYTE;
+        HCCL_RUN_INFO("InitSymmetricMemory, comm identifier[%s], userRank[%u], userRankSize[%u], stride[%llu], devicePhyId[%u].",
+            identifier_.c_str(), realUserRank_, userRankSize_, stride, devicePhyId_);
+        
+        symmetricMemoryAgent_ = std::make_shared<SymmetricMemoryAgent>(socketManager_, devicePhyId_,
+            deviceLogicId_, localVnicIp_, rankInfoList_, realUserRank_, useSuperPodMode_, identifier_);
+        CHK_SMART_PTR_NULL(symmetricMemoryAgent_);
+
+        symmetricMemory_ = std::make_unique<SymmetricMemory>(realUserRank_, userRankSize_, stride, symmetricMemoryAgent_);
+        CHK_SMART_PTR_NULL(symmetricMemory_);
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, CommSymWindow *winHandle)
+    {
+        CHK_SMART_PTR_NULL(symmetricMemory_);
+        return symmetricMemory_->RegisterSymmetricMem(ptr, size, winHandle);
+    }
+
+    HcclResult HcclCommunicator::DeregisterWindow(CommSymWindow winHandle)
+    {
+        CHK_SMART_PTR_NULL(symmetricMemory_);
+        return symmetricMemory_->DeregisterSymmetricMem(winHandle);
+    }
+
+    HcclResult HcclCommunicator::GetCommSymWin(void* ptr, size_t size, CommSymWindow *winHandle, size_t *offset)
+    {
+        CHK_SMART_PTR_NULL(symmetricMemory_);
+        return symmetricMemory_->FindSymmetricWindow(ptr, size, winHandle, reinterpret_cast<u64*>(offset));
+    }
 }

```

### src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc
```diff
@@ -0,0 +1,603 @@
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
+#include "symmetric_memory.h"
+#include <algorithm> // for std::max
+#include <cstddef>
+#include <list>      // for SimpleVaAllocator
+#include "hccl_comm.h"
+
+namespace hccl 
+{
+/**
+ * @brief (内部) 简单的VA空间分配器
+ */
+class SymmetricMemory::SimpleVaAllocator {
+    std::list<FreeBlock> freeList_; // 按offset排序的空闲块
+    std::mutex mutex_;
+    size_t totalSize_;
+
+public:
+    SimpleVaAllocator() : totalSize_(0) {}
+    ~SimpleVaAllocator() { 
+        Destroy();
+    }
+
+    // 增加调试打印函数
+    void Dump(const char* tag) {
+        HCCL_ERROR("[%s] === VA Allocator Dump (Total: %zu) ===", tag, totalSize_);
+        size_t freeSum = 0;
+        int i = 0;
+        for (auto &block : freeList_) {
+            HCCL_ERROR("  Block[%d]: offset %zu (0x%zx) -> size %zu (0x%zx) | end: %zu", 
+                i++, block.offset, block.offset, block.size, block.size, block.offset + block.size);
+            freeSum += block.size;
+        }
+        HCCL_ERROR("  Total Free: %zu (%.2f%%)", freeSum, (double)freeSum / totalSize_ * 100.0);
+        HCCL_ERROR("==========================================");
+    }
+
+    HcclResult Init(size_t size) {
+        std::lock_guard<std::mutex> lock(mutex_);
+        CHK_PRT_RET(size == 0, HCCL_ERROR("[SimpleVaAllocator][Init] invalid size: 0"), HCCL_E_PARA);
+        totalSize_ = size;
+        freeList_.push_back({0, size});
+        return HCCL_SUCCESS;
+    }
+
+    void Destroy() {
+        std::lock_guard<std::mutex> lock(mutex_);
+        freeList_.clear();
+        totalSize_ = 0;
+    }
+
+    HcclResult Reserve(size_t size, size_t align, size_t &offset) {
+        std::lock_guard<std::mutex> lock(mutex_);
+        
+        // Debug: 打印请求信息
+        HCCL_INFO("[VAAllocator] Request Reserve: size %zu, align %zu", size, align);
+
+        for (auto it = freeList_.begin(); it != freeList_.end(); ++it) {
+            size_t start = it->offset;
+            size_t end = it->offset + it->size;
+
+            // 计算对齐后的offset
+            size_t alignedOffset = (start + align - 1) & ~(align - 1);
+            
+            // 检查对齐后的空间是否足够
+            if (alignedOffset < end && (end - alignedOffset) >= size) {
+                // 找到了
+                offset = alignedOffset;
+                
+                size_t frontPad = alignedOffset - start;
+                size_t backPad = (end) - (alignedOffset + size);
+                
+                HCCL_INFO("[VAAllocator] Found Block: [0x%zx, 0x%zx], Need aligned: 0x%zx. FrontPad: %zu, BackPad: %zu",
+                    start, end, alignedOffset, frontPad, backPad);
+
+                auto to_erase = it;
+                // 先插入后部碎片（如果存在）
+                if (backPad > 0) {
+                    // 后部碎片应该插入在to_erase之后
+                    freeList_.insert(std::next(to_erase), 
+                                    {alignedOffset + size, backPad});
+                }
+                
+                // 再插入前部碎片（如果存在）
+                if (frontPad > 0) {
+                    // 前部碎片插入在to_erase之前
+                    freeList_.insert(to_erase, {start, frontPad});
+                }
+                
+                // 最后删除原空闲块
+                freeList_.erase(to_erase);
+                return HCCL_SUCCESS;
+            } else {
+                // 只有当块看起来比较大但因为对齐无法满足时才打印，避免刷屏
+                if (it->size >= size) {
+                    HCCL_DEBUG("[VAAllocator] Block [0x%zx, 0x%zx] size %zu skipped. AlignedOffset 0x%zx overlaps end or insufficient.",
+                        start, end, it->size, alignedOffset);
+                }
+            }
+        }
+
+        // 失败时打印当前内存布局，极大概率是碎片化导致
+        HCCL_ERROR("[VAAllocator] Failed to reserve size %zu with align %zu. No suitable block found.", size, align);
+        Dump("Reserve Failed");
+        
+        return HCCL_E_MEMORY;
+    }
+
+    HcclResult Release(size_t offset, size_t size) {
+        std::lock_guard<std::mutex> lock(mutex_);
+        // 边界检查
+        if (offset + size > totalSize_) {
+            HCCL_ERROR("[VAAllocator] Release out of range. off %zu + size %zu > total %zu", offset, size, totalSize_);
+            return HCCL_E_PARA;
+        }
+
+        // 找到插入位置并合并
+        auto it = freeList_.begin();
+        while (it != freeList_.end() && it->offset < offset) {
+            ++it;
+        }
+        
+	    // 检查重叠,直接报错
+ 	    // 与前一块重叠
+ 	    if (it != freeList_.begin()) {
+ 	        auto prevIt = std::prev(it);
+ 	        if (prevIt->offset <=  offset && prevIt->offset + prevIt->size >= offset + size) { //  完全重叠表示释放空闲区域
+                HCCL_WARNING("[VAAllocator] Releasing block[0x%zx, size %zu] is free", offset, size);
+ 	            return HCCL_SUCCESS;     
+ 	        }
+ 	        if (prevIt->offset + prevIt->size > offset) {
+                HCCL_ERROR("[VAAllocator] Releasing block[0x%zx, size %zu] overlaps with the previous block.", offset, size);
+ 	            return HCCL_E_PARA;
+ 	        }
+ 	    }
+ 	    // 与后一块重叠
+ 	    if (it != freeList_.end() && it->offset < offset + size) {
+ 	        if (it->offset <= offset && it->offset + it->size >= offset + size) { //  完全重叠表示释放空闲区域
+                HCCL_WARNING("[VAAllocator] Releasing block[0x%zx, size %zu] is free", offset, size);
+ 	            return HCCL_SUCCESS;
+ 	        }
+            HCCL_ERROR("[VAAllocator] Releasing block[0x%zx, size %zu] overlaps with the next block.", offset, size);
+ 	        return HCCL_E_PARA;
+ 	    }
+        
+        // 插入新释放的块
+        auto newIt = freeList_.insert(it, {offset, size});
+        HCCL_INFO("[VAAllocator] Releasing block[0x%zx, size %zu]", offset, size);
+
+        // 尝试与后一块合并
+        if (std::next(newIt) != freeList_.end()) {
+            auto nextIt = std::next(newIt);
+            if (newIt->offset + newIt->size == nextIt->offset) {
+                newIt->size += nextIt->size;
+                freeList_.erase(nextIt);
+            }
+        }
+        // 尝试与前一块合并
+        if (newIt != freeList_.begin()) {
+            auto prevIt = std::prev(newIt);
+            if (prevIt->offset + prevIt->size == newIt->offset) {
+                prevIt->size += newIt->size;
+                freeList_.erase(newIt);
+            }
+        }
+        return HCCL_SUCCESS;
+    }
+};
+
+SymmetricMemory::SymmetricMemory(u32 rank, u32 rankSize, size_t stride, std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent)
+    : rank_(rank),
+      rankSize_(rankSize),
+      stride_(stride), 
+      vaAllocator_(new (std::nothrow) SimpleVaAllocator()),
+      symmetricMemoryAgent_(std::move(symmetricMemoryAgent))
+{
+    remoteShareablePids.resize(rankSize_, 0);
+}
+
+SymmetricMemory::~SymmetricMemory() 
+{
+    HCCL_INFO("[SymmetricMemory][~SymmetricMemory] begin");
+    for (auto& pair : windowMap_) {
+        DeregisterSymmetricMem(pair.first);
+    }
+    windowMap_.clear();
+    sortedWindows_.clear();
+
+    if (heapBase_) {
+        if (aclrtReleaseMemAddress(heapBase_) != ACL_SUCCESS) {
+            HCCL_ERROR("[SymmetricMemory][~SymmetricMemory] Failed to release symmetric heap VA: %p", heapBase_);
+        }
+    }
+
+    HCCL_INFO("[SymmetricMemory][~SymmetricMemory] end");
+}
+
+HcclResult SymmetricMemory::EnsureInit() {
+    std::call_once(init_flag_, [this]() {
+        initResult_ = Init();
+    });
+    return initResult_;
+}
+
+HcclResult SymmetricMemory::Init() 
+{
+    CHK_SMART_PTR_NULL(vaAllocator_);
+    CHK_SMART_PTR_NULL(symmetricMemoryAgent_);
+
+    CHK_PRT_RET(rankSize_ < 2, HCCL_ERROR("[SymmetricMemory][Init] single rank communicator"), HCCL_E_PARA);
+    CHK_PRT_RET(stride_ == 0, HCCL_ERROR("[SymmetricMemory][Init] invalid stride: 0"), HCCL_E_PARA);
+
+    size_t free = 0;
+    size_t total = 0;
+    aclError acl_ret = aclrtGetMemInfo(ACL_HBM_MEM_HUGE, &free, &total); // 获取当前进程总的物理内存大小
+    CHK_PRT_RET(acl_ret != ACL_SUCCESS,
+        HCCL_ERROR("[SymmetricMemory][Init] aclrtGetMemInfo failed, ret=[%d]", acl_ret), HCCL_E_INTERNAL);
+    CHK_PRT_RET(stride_ > total,
+        HCCL_ERROR("[SymmetricMemory][Init] Stride[%llu] is out of total[%llu].", stride_, total), HCCL_E_PARA);
+
+    acl_ret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity_);
+    CHK_PRT_RET(acl_ret != ACL_SUCCESS,
+        HCCL_ERROR("[SymmetricMemory][Init] Get memory granularity failed, ret=[%d]", acl_ret), HCCL_E_INTERNAL);
+
+    CHK_PRT_RET(granularity_ == 0, HCCL_ERROR("[SymmetricMemory][Init] Invalid memory granularity: 0"), HCCL_E_INTERNAL);
+
+    CHK_PRT_RET(stride_ % granularity_ != 0,
+        HCCL_ERROR("[SymmetricMemory][Init] Stride %llu is not a multiple of granularity %zu.", stride_, granularity_), HCCL_E_PARA);
+
+    size_t totalHeapSize = (size_t)stride_ * rankSize_;    // 每个rank都预留一个总大小为 totalHeapSize 的VA空间。
+    void* hintPtr = reinterpret_cast<void*>(targetStartTB);
+
+    if (aclrtReserveMemAddressNoUCMemory(&heapBase_, totalHeapSize, 0, hintPtr, 0) != ACL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][Init] aclrtReserveMemAddress failed to reserve %zu bytes. stride: %llu, rankSize: %u.",
+                   totalHeapSize, stride_, rankSize_);
+        return HCCL_E_INTERNAL;
+    }
+    //  初始化VA分配器 (管理本地rank的stride_大小空间，即管理偏移量。
+    //  这是一个集合调用，所有rank上的vaAllocator_状态将保持一致（前提是 SimpleVaAllocator 是确定性的）
+    CHK_RET(vaAllocator_->Init(stride_));
+
+    CHK_RET(symmetricMemoryAgent_->Init());
+    CHK_RET(GetAllRankPid());
+
+    HCCL_INFO("[SymmetricMemory][Init] SymmetricMemory initialized. Rank[%u], Local Heap Base: %p, Stride: %llu, RankSize: %u.",
+               rank_, heapBase_, stride_, rankSize_);
+
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemory::GetAllRankPid()
+{
+    int32_t localPid{0};    // 当前进程号
+    if (aclrtDeviceGetBareTgid(&localPid) != ACL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][GetAllRankPid] Failed to get pid");
+        return HCCL_E_DRV;
+    }
+    HCCL_INFO("[SymmetricMemory][GetAllRankPid] Local pid: %d.", localPid);
+
+    CHK_RET(symmetricMemoryAgent_->ExchangeInfo((void*)&localPid, (void*)remoteShareablePids.data(), sizeof(localPid)));
+
+    std::string pidStr;
+    for (u32 i = 0; i < remoteShareablePids.size(); i++) {
+        pidStr += std::to_string(remoteShareablePids[i]);
+        pidStr += "; ";
+    }
+    HCCL_INFO("[SymmetricMemory][GetAllRankPid] remote pids: %s", pidStr.c_str());
+
+    return HCCL_SUCCESS;
+}
+
+void* SymmetricMemory::AllocSymmetricMem(size_t size)
+{
+    void* devWin = nullptr;
+    void *ptr = nullptr;
+    HcclResult ret = HcclMemAlloc(&ptr, size);
+    if (ret != HCCL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][AllocSymmetricMem] HcclMemAlloc failed for size[%u].", size);
+        return nullptr;
+    }
+
+    ret = RegisterSymmetricMem(ptr, size, &devWin);
+    if (ret != HCCL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][AllocSymmetricMem] RegisterSymmetricMem failed for ptr[%p], size[%u].", ptr, size);
+        (void)HcclMemFree(ptr);
+        return nullptr;
+    }
+    return devWin;
+}
+
+HcclResult SymmetricMemory::FreeSymmetricMem(void* devWin)
+{
+    std::shared_ptr<SymmetricWindow> pWin = windowMap_[devWin];
+    if (pWin == nullptr) {
+        return HCCL_SUCCESS;
+    }
+
+    void* userPtr = pWin->userVa;
+    CHK_RET(DeregisterSymmetricMem(devWin));
+    CHK_RET(HcclMemFree(userPtr));
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemory::AddSymmetricWindow(std::shared_ptr<SymmetricWindow> &win)
+{
+    CHK_RET(hrtMalloc(&win->devWin, sizeof(SymmetricWindow)));
+    CHK_RET(hrtMemSyncCopy(win->devWin, sizeof(SymmetricWindow), 
+        win.get(), sizeof(SymmetricWindow), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
+
+    sortedWindows_.push_back(win);
+    std::sort(sortedWindows_.begin(), sortedWindows_.end(), 
+        [](const std::shared_ptr<SymmetricWindow>& a, const std::shared_ptr<SymmetricWindow>& b) {
+            return ((uintptr_t)a->userVa < (uintptr_t)b->userVa) || 
+                (((uintptr_t)a->userVa == (uintptr_t)b->userVa) && (a->userSize < b->userSize));
+    });
+
+    windowMap_[win->devWin] = win;
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemory::DeleteSymmetricWindow(std::shared_ptr<SymmetricWindow> &win)
+{
+    auto it = std::find_if(sortedWindows_.begin(), sortedWindows_.end(),
+        [&win](const std::shared_ptr<SymmetricWindow>& w) {
+            return w.get() == win.get();
+        });
+    if (it != sortedWindows_.end()) {
+        CHK_PRT(hrtFree(win->devWin));
+        windowMap_.erase(win->devWin);
+        sortedWindows_.erase(it);
+    }
+
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemory::DeleteSymmetricWindow(void* devWin)
+{
+    auto it = windowMap_.find(devWin);
+    if (it != windowMap_.end()) {
+        std::shared_ptr<SymmetricWindow> win = it->second;
+        CHK_PRT(hrtFree(win->devWin));
+        windowMap_.erase(it);
+
+        auto vecIt = std::find_if(sortedWindows_.begin(), sortedWindows_.end(),
+            [&win](const std::shared_ptr<SymmetricWindow>& w) {
+                return w.get() == win.get();
+            });
+        if (vecIt != sortedWindows_.end()) {
+            sortedWindows_.erase(vecIt);
+        }
+    }
+
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemory::GetMemoryInfo(void* ptr, size_t size, void** baseUserVa, size_t* baseVaSize, aclrtDrvMemHandle* paHandle)
+{
+    CHK_PTR_NULL(ptr);
+    CHK_PRT_RET(size == 0, HCCL_ERROR("[SymmetricMemory][GetMemoryInfo] Invalid size: 0."), HCCL_E_PARA);
+
+    // 打印当前注册请求的关键信息
+    HCCL_INFO("[SymmetricMemory][GetMemoryInfo] Request: ptr=%p, size=%zu, granularity=%zu", 
+        ptr, size, granularity_);
+
+    if(aclrtMemGetAddressRange(ptr, baseUserVa, baseVaSize) != 0) {
+        HCCL_ERROR("[SymmetricMemory][GetMemoryInfo] aclrtMemGetAddressRange failed for ptr[%p], size[%zu]. ", ptr, size);
+        return HCCL_E_PARA;
+    }
+    CHK_PTR_NULL(*baseUserVa);
+    CHK_PRT_RET(*baseVaSize == 0, HCCL_ERROR("[SymmetricMemory][GetMemoryInfo] Invalid baseVaSize: 0."), HCCL_E_PARA);
+    CHK_PRT_RET(*baseVaSize % granularity_ != 0,
+        HCCL_ERROR("[SymmetricMemory][GetMemoryInfo] baseVaSize %u is not a multiple of granularity %zu.",
+        *baseVaSize, granularity_), HCCL_E_PARA);
+
+    if (aclrtMemRetainAllocationHandle(*baseUserVa, paHandle) != 0) {
+        HCCL_ERROR("[SymmetricMemory][GetMemoryInfo] MemRetainAllocationHandle failed for ptr[%p], size[%zu]. ", ptr, size);
+        return HCCL_E_PARA;
+    }
+    CHK_PTR_NULL(*paHandle);
+
+    if (reinterpret_cast<uintptr_t>(ptr) + size > reinterpret_cast<uintptr_t>(*baseUserVa) +  *baseVaSize) {
+        HCCL_ERROR("[SymmetricMemory][GetMemoryInfo] ptr=%p size=%zu exceeds  block [baseUserVa=%p, size=%zu]", 
+           ptr, size, *baseUserVa, *baseVaSize);
+        return HCCL_E_PARA;
+    }
+
+    HCCL_INFO("[SymmetricMemory][GetMemoryInfo] Retained paHandle[%p] for baseUserVa[%p],  baseVaSize[%zu]. Total Stride: %zu",
+        *paHandle, *baseUserVa,  *baseVaSize, stride_);
+
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemory::RegisterSymmetricMem(void* ptr, size_t size, void** devWin)
+{
+    CHK_RET(EnsureInit());
+    CHK_PTR_NULL(devWin);
+    void* baseUserVa = nullptr;
+    size_t baseVaSize = 0;
+    aclrtDrvMemHandle paHandle;
+    CHK_RET(GetMemoryInfo(ptr, size, &baseUserVa, &baseVaSize, &paHandle));
+
+    std::shared_ptr<PaMappingInfo> paMapInfo;
+    auto it = paMappingMap_.find(paHandle);
+    if (it != paMappingMap_.end()) {
+        paMapInfo = it->second;
+        paMapInfo->refCount++;
+        HCCL_INFO("PA handle[%p], refCount[%d]", paHandle, paMapInfo->refCount);
+    }else {
+        size_t offset = 0;
+        // 使用 granularity_ (通常是2MB) 作为对齐参数
+        if (vaAllocator_->Reserve( baseVaSize, granularity_, offset) != HCCL_SUCCESS) {
+            HCCL_ERROR("[SymmetricMemory][RegisterSymmetricMem] Failed to reserve VA space. "
+                "Req alignedSize: %zu (0x%zx), Align: %zu. Total Stride: %zu. "
+                "Is fragmentation too high or stride too small?", 
+                 baseVaSize,  baseVaSize, granularity_, stride_);
+            return HCCL_E_MEMORY;
+        }
+        paMapInfo = std::make_shared<PaMappingInfo>();
+        paMapInfo->paHandle = paHandle;
+        paMapInfo->origAllocBaseVa = baseUserVa;
+        paMapInfo->origAllocSize = baseVaSize;
+        paMapInfo->heapBaseOffset = offset;
+        paMapInfo->refCount = 1;
+        paMappingMap_.emplace(paHandle, paMapInfo);
+    }
+    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
+    pWin->userVa = baseUserVa;
+    pWin->userSize = baseVaSize;
+    pWin->baseVa = static_cast<uint8_t*>(heapBase_) + paMapInfo->heapBaseOffset;
+    pWin->alignedHeapOffset = paMapInfo->heapBaseOffset;
+    pWin->alignedSize =  baseVaSize;
+    pWin->localRank = rank_;
+    pWin->rankSize = rankSize_;
+    pWin->stride = stride_;
+    pWin->paHandle = paHandle;
+
+    HcclResult ret = RegisterInternal(paHandle, paMapInfo->heapBaseOffset,  baseVaSize);
+    if (ret != HCCL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory] RegisterInternal Failed!");
+        goto INTERNAL_ERROR;
+    }
+    ret = AddSymmetricWindow(pWin);
+    if (ret != HCCL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory] AddSymmetricWindow Failed!");
+        goto INTERNAL_ERROR;
+    }
+
+    *devWin = pWin->devWin;
+    return HCCL_SUCCESS;
+
+INTERNAL_ERROR:
+    if (paMapInfo->refCount == 1) {
+        HCCL_ERROR("[SymmetricMemory] Releasing offset 0x%zx", paMapInfo->heapBaseOffset);
+        (void)vaAllocator_->Release(paMapInfo->heapBaseOffset,  baseVaSize);
+        paMappingMap_.erase(paHandle);
+    } else {
+        paMapInfo->refCount--;
+    }
+    return ret;
+}
+
+HcclResult SymmetricMemory::DeregisterSymmetricMem(void* devWin)
+{
+    HcclResult ret = HCCL_SUCCESS;
+    if (devWin == nullptr) {
+        HCCL_INFO("[SymmetricMemory] DeregisterSymmetricMem sucessed devWin[nullptr]");
+        return ret;
+    }
+
+    for (auto it = sortedWindows_.begin(); it != sortedWindows_.end();) {
+        if ((*it)->devWin != devWin) {
+            it++;
+            continue;
+        }
+
+        std::shared_ptr<PaMappingInfo> paMapInfo = paMappingMap_[(*it)->paHandle];
+        if (paMapInfo->refCount == 1) {
+            for (u32 i = 0; i < rankSize_; i++) {
+                void* virPtr = static_cast<uint8_t*>(heapBase_) + (stride_ * i) + (*it)->alignedHeapOffset;
+                aclrtDrvMemHandle handle;
+                aclError aclRet = aclrtMemRetainAllocationHandle(virPtr, &handle);
+                if (aclRet != ACL_SUCCESS) {
+                    HCCL_ERROR("[SymmetricMemory][DeregisterSymmetricMem] MemRetainAllocationHandle failed for ptr[%p], rank[%u], ret[%d].", virPtr, i, aclRet);
+                    ret = HCCL_E_DRV;
+                    continue;
+                }
+                HCCL_INFO("[SymmetricMemory][DeregisterSymmetricMem] Start to UnmapMem virPtr[%p], handle[%p], rank[%u].", virPtr, handle, i);
+                aclRet = aclrtUnmapMem(virPtr);
+                if (aclRet != ACL_SUCCESS) {
+                    HCCL_ERROR("[SymmetricMemory][DeregisterSymmetricMem] Failed to unmap mem for rank %u at va %p, ret[%d].", i, virPtr, aclRet);
+                    ret = HCCL_E_DRV;
+                }
+                aclRet = aclrtFreePhysical(handle);
+                if (aclRet != ACL_SUCCESS) {
+                    HCCL_ERROR("[SymmetricMemory][DeregisterSymmetricMem] FreePhysical handle[%p] failed, ret[%d], rank[%u].", handle, aclRet, i);
+                    ret = HCCL_E_DRV;
+                }
+            }
+            vaAllocator_->Release((*it)->alignedHeapOffset, (*it)->alignedSize);
+            paMappingMap_.erase((*it)->paHandle);
+        } else {
+            paMapInfo->refCount--;
+        }
+        
+        it = sortedWindows_.erase(it);
+        windowMap_.erase(devWin);
+        CHK_RET(hrtFree(devWin));
+        break;
+    }
+
+    return ret;
+}
+
+HcclResult SymmetricMemory::FindSymmetricWindow(void* ptr, size_t size, void** win, u64 *offset)
+{
+    uintptr_t userVaStart = reinterpret_cast<uintptr_t>(ptr);
+    uintptr_t userVaEnd = userVaStart + size;
+
+    // 遍历所有窗口
+    for (const auto& pWin : sortedWindows_) {
+        uintptr_t winStart = reinterpret_cast<uintptr_t>(pWin->userVa);
+        if (winStart > userVaStart) {
+            return HCCL_E_NOT_FOUND;
+        }
+
+        if (userVaStart >= winStart && userVaEnd <= winStart + pWin->userSize) {
+            *win = pWin->devWin;
+            *offset = userVaStart - winStart;
+            return HCCL_SUCCESS;
+        }
+    }
+
+    return HCCL_E_NOT_FOUND;
+}
+
+// --- Private Methods ---
+HcclResult SymmetricMemory::RegisterInternal(aclrtDrvMemHandle &paHandle, size_t offset, size_t mapSize)
+{
+    aclrtMemFabricHandle shareableHandle;
+    if (aclrtMemExportToShareableHandleV2(paHandle, 0, 
+        ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, (void*)&shareableHandle) != ACL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][RegisterInternal] Failed to export shareable handle. offset: %zu, size: %zu",
+            offset, mapSize);
+        return HCCL_E_DRV;
+    }
+    if(aclrtMemSetPidToShareableHandleV2((void*)&shareableHandle, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC,
+        remoteShareablePids.data(), remoteShareablePids.size()) != ACL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][RegisterInternal] Failed to aclrtMemSetPidToShareableHandleV2");
+        return HCCL_E_DRV;
+    }
+
+    ShareableInfo shareableInfo{offset, mapSize, shareableHandle};
+    std::vector<ShareableInfo> remoteShareableInfos(rankSize_);
+
+    CHK_RET(symmetricMemoryAgent_->ExchangeInfo((void*)&shareableInfo, (void*)remoteShareableInfos.data(), sizeof(ShareableInfo)));
+    for (u32 i = 0; i < rankSize_; i++) {
+        if (remoteShareableInfos[i].offset != offset || remoteShareableInfos[i].size != mapSize) {
+            HCCL_ERROR("[SymmetricMemory][RegisterInternal] rank[%u]:[offset: %llu, mapSize: %llu] is not equal to "
+            "rank[%u]:[offset: %llu, mapSize: %llu]. Please ensure collective invocation!", rank_, offset, mapSize,
+            i, remoteShareableInfos[i].offset, remoteShareableInfos[i].size);
+            return HCCL_E_INTERNAL;
+        }
+    }
+
+    u32 i = 0;
+    if(paMappingMap_[paHandle]->refCount == 1) {
+        aclrtDrvMemHandle importedHandle;
+        for (; i < rankSize_; i++) {
+            void* targetVa = static_cast<uint8_t*>(heapBase_) + (stride_ * i) + offset;
+            if (i == rank_) {
+                importedHandle = paHandle;
+            } else if (aclrtMemImportFromShareableHandleV2((void*)&(remoteShareableInfos[i].handle), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0,
+                &importedHandle) != ACL_SUCCESS) {
+                HCCL_ERROR("[SymmetricMemory][RegisterInternal] Failed to import handle from rank %u.", i);
+                goto MAP_ERROR;
+            }
+
+            if (aclrtMapMem(targetVa, mapSize, 0, importedHandle, 0) != ACL_SUCCESS) {
+                HCCL_ERROR("[SymmetricMemory][RegisterInternal] Failed to map mem for rank %u at va %p.", i, targetVa);
+                goto MAP_ERROR;
+            }
+            HCCL_INFO("[SymmetricMemory][RegisterInternal] success to Mapmem for rank %u at va %p to handle[%p].", i, targetVa, importedHandle);
+        }
+    }
+    return HCCL_SUCCESS;
+
+MAP_ERROR:
+    for (u32 j = 0; j < i; j++) {
+        (void)aclrtUnmapMem(static_cast<uint8_t*>(heapBase_) + (stride_ * j) + offset);
+    }
+    return HCCL_E_DRV;
+}
+
+} // namespace hccl

```

### src/framework/communicator/impl/symmetric_memory/symmetric_memory.h
```diff
@@ -0,0 +1,134 @@
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
+#ifndef SYMMETRIC_MEMORY_H
+#define SYMMETRIC_MEMORY_H
+
+#include "log.h"
+#include <vector>
+#include <unordered_map>
+#include <map>
+#include <mutex>
+#include <functional>
+#include <memory>
+#include "symmetric_memory_agent.h"
+#include "hccl_mem_alloc.h"
+
+// HCCL
+#include "hccl/base.h"
+#include "hccl_comm.h"
+#include "adapter_rts_common.h"
+#include "hccl_inner.h"
+
+// NPU VMM API
+#include "acl/acl_rt.h"
+
+namespace hccl {
+
+struct FreeBlock {
+    size_t offset;
+    size_t size;
+};
+
+struct ShareableInfo {
+    size_t offset;
+    size_t size;
+    aclrtMemFabricHandle handle;
+};
+
+struct SymmetricWindow {
+    void* userVa;
+    size_t userSize;
+
+    void* baseVa; // 对应userVa在对称堆上的地址, 是不一定对齐的
+    size_t alignedHeapOffset;
+    size_t alignedSize;
+    u32 localRank;
+    u32 rankSize;
+    size_t stride;
+    aclrtDrvMemHandle paHandle;
+
+    void* devWin; // device端结构体
+};
+
+struct PaMappingInfo {
+    // 唯一标识：PA 句柄
+    aclrtDrvMemHandle paHandle;
+
+    // 这里需要记录原始 allocation 的起始 VA (例如 0x1000) 和总大小 (100MB)
+    void* origAllocBaseVa;
+    size_t origAllocSize;
+
+    // 对称堆上的映射信息
+    // 这块物理内存在对称堆上的起始 offset (例如在 heapBase + 0x5000)
+    size_t heapBaseOffset;
+
+    // 引用计数：有多少个 Window 正在复用这块物理内存
+    // 当 refCount 降为 0 时，才执行 Unmap 和 Release VA
+    u32 refCount;
+};
+
+/**
+ * @brief 对称内存管理器
+ * 负责对称VA空间的预留、注册、映射和查找。
+ */
+class SymmetricMemory {
+public:
+    SymmetricMemory(u32 rank, u32 rankSize, size_t stride, std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent);
+    ~SymmetricMemory();
+
+    // 禁止拷贝和赋值
+    SymmetricMemory(const SymmetricMemory&) = delete;
+    SymmetricMemory& operator=(const SymmetricMemory&) = delete;
+    HcclResult EnsureInit();
+    void* AllocSymmetricMem(size_t size);
+    HcclResult FreeSymmetricMem(void* devWin);
+    HcclResult GetMemoryInfo(void* ptr, size_t size, void** baseUserVa, size_t* baseVaSize, aclrtDrvMemHandle* paHandle);
+    HcclResult RegisterSymmetricMem(void* ptr, size_t size, void** devWin);
+    HcclResult DeregisterSymmetricMem(void* devWin);
+    HcclResult FindSymmetricWindow(void* ptr, size_t size, void** win, u64 *offset);
+
+private:
+    HcclResult Init();
+    HcclResult GetAllRankPid();
+    HcclResult RegisterInternal(aclrtDrvMemHandle &paHandle, size_t offset, size_t mapSize);
+    HcclResult AddSymmetricWindow(std::shared_ptr<SymmetricWindow> &win);
+    HcclResult DeleteSymmetricWindow(std::shared_ptr<SymmetricWindow> &win);
+    HcclResult DeleteSymmetricWindow(void* devWin);
+
+private:
+    // VA空间分配器 (Pimpl)
+    std::once_flag init_flag_;
+    u32 rank_{0};
+    u32 rankSize_{0};
+    size_t stride_{0};      // 每个Rank的VA空间大小
+    void* heapBase_{nullptr};  // 对称VA空间的总基地址 (所有rank相同)
+    size_t granularity_{0};
+    class SimpleVaAllocator;
+    std::unique_ptr<SimpleVaAllocator> vaAllocator_;
+    HcclResult initResult_{HCCL_E_INTERNAL}; // 存储Init()的结果
+    std::vector<std::shared_ptr<SymmetricWindow>> sortedWindows_;
+    std::map<void*, std::shared_ptr<SymmetricWindow>> windowMap_; // device指针到host SymmetricWindow 的映射
+    std::unordered_map<aclrtDrvMemHandle, std::shared_ptr<PaMappingInfo>> paMappingMap_;
+    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_;
+    std::vector<int32_t> remoteShareablePids;   // 所有rank进程号
+    aclrtPhysicalMemProp prop = {              // 内存信息，用来获取内存映射的粒度
+        ACL_MEM_HANDLE_TYPE_NONE,
+        ACL_MEM_ALLOCATION_TYPE_PINNED,
+        ACL_HBM_MEM_HUGE,
+        {0, ACL_MEM_LOCATION_TYPE_DEVICE},
+        0
+    };
+    size_t targetStartTB = 40ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL;   //  从40TB处预留虚拟内存
+};
+
+} // namespace hccl
+
+#endif // SYMMETRIC_MEMORY_H
\ No newline at end of file

```

### src/framework/communicator/impl/symmetric_memory/symmetric_memory_agent.cc
```diff
@@ -0,0 +1,249 @@
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
+#include "symmetric_memory_agent.h"
+#include <chrono>
+
+namespace hccl {
+using namespace std;
+
+const string STR_IPC_MEM_EXCHANGE = "Exchange_Info";
+constexpr u32 USLEEP_ONE_THOUSAND = 1000;
+
+SymmetricMemoryAgent::SymmetricMemoryAgent(const std::unique_ptr<HcclSocketManager> &socketManager, u32 devicePhyId,
+    s32 deviceLogicId, const HcclIpAddress &localVnicIp, const std::vector<RankInfo> &rankInfoList, u32 userRank,
+    bool useSuperPodMode, const std::string &identifier)
+    : socketManager_(socketManager), devicePhyId_(devicePhyId), deviceLogicId_(deviceLogicId),
+      localVnicIp_(localVnicIp), rankInfoList_(rankInfoList), userRank_(userRank), rankSize_(rankInfoList.size()),
+      useSuperPodMode_(useSuperPodMode), identifier_(identifier)
+{
+    if (rankSize_ >=2) {    // 当前数据交换算法使用超节点内大平面ring算法，需要和“左右”两边的rank建链
+        leftRank_ = (userRank_ - 1 + rankSize_) % rankSize_;
+        rightRank_ = (userRank_ + 1) % rankSize_;
+    }
+}
+
+SymmetricMemoryAgent::~SymmetricMemoryAgent() {
+    threadRun_ = false;
+    if (recvThread_ && recvThread_->joinable()) {
+        recvThread_->join();
+        recvThread_ = nullptr;
+    }
+    if (vnicPortCtx_ != nullptr) {
+        HcclNetCloseDev(vnicPortCtx_);
+        vnicPortCtx_ = nullptr;
+    }
+}
+
+HcclResult SymmetricMemoryAgent::Init() {
+    CHK_PRT_RET(rankSize_ < 2, HCCL_ERROR("[SymmetricMemoryAgent][Init] single rank communicator"), HCCL_E_PARA);
+    CHK_RET(EstablishSockets());
+    CHK_RET(InitRecvThread());
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemoryAgent::InitRecvThread() {
+    threadRun_ = true;
+    recvThread_.reset(new (std::nothrow) std::thread(&SymmetricMemoryAgent::DealWithRequest, this));
+    CHK_SMART_PTR_NULL(recvThread_);
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemoryAgent::EstablishSockets()
+{
+    CHK_PRT_RET((vnicPortCtx_ != nullptr),
+        HCCL_ERROR("[SymmetricMemoryAgent][Init] already initd"), HCCL_E_PARA);
+    CHK_RET(HcclNetOpenDev(&vnicPortCtx_, NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, localVnicIp_));
+    CHK_PTR_NULL(vnicPortCtx_);
+
+    HCCL_INFO("[SymmetricMemoryAgent][EstablishSockets] userRank[%u], leftRank_[%u], rightRank_[%u], rankSize_[%u]",
+        userRank_, leftRank_, rightRank_, rankSize_);
+    for (size_t i = 0; i < rankInfoList_.size(); i++) {
+        if (rankInfoList_[i].userRank == leftRank_ || rankInfoList_[i].userRank == rightRank_) {
+            HcclRankLinkInfo remoteLinkInfo;
+            RankInfo dstRankInfo = rankInfoList_[i];
+            remoteLinkInfo.userRank = dstRankInfo.userRank;
+            remoteLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
+            remoteLinkInfo.ip = HcclIpAddress(dstRankInfo.devicePhyId);
+            if (useSuperPodMode_) {
+                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID,
+                    dstRankInfo.superDeviceId, remoteLinkInfo.ip));
+            } else {
+                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
+                    dstRankInfo.devicePhyId, remoteLinkInfo.ip));
+            }
+            // 通信域未分配端口则使用默认端口
+            remoteLinkInfo.port =
+                dstRankInfo.deviceVnicPort == HCCL_INVALID_PORT ? HETEROG_CCL_PORT : dstRankInfo.deviceVnicPort;
+            remoteLinkInfo.socketsPerLink = 1;
+            string newTag = GenerateSocketTag(devicePhyId_, rankInfoList_[i].devicePhyId);
+            std::vector<std::shared_ptr<HcclSocket> > tmpSockets;
+            HcclResult ret = socketManager_->CreateSingleLinkSocket(
+                newTag, vnicPortCtx_, remoteLinkInfo, tmpSockets, false, true);
+            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Create][DestSockets]Create single link sockets failed, "
+                "local rank[%u], remote rank[%u]", userRank_, rankInfoList_[i].userRank), ret);
+            if (tmpSockets.size() != 1) {
+                HCCL_ERROR("[SymmetricMemoryAgent][CreateVnic] socket number[%llu] is not 1 as expected!", tmpSockets.size());
+                return HCCL_E_INTERNAL;
+            }
+            // 设置强制断链为关闭，避免进程退出时recv失败
+            tmpSockets[0]->SetForceClose(false);
+            mapRankIdconnectedSockets_[remoteLinkInfo.userRank] = (tmpSockets[0]);
+            mapRankId2DevPhyId_[remoteLinkInfo.userRank] = remoteLinkInfo.devicePhyId;
+        }
+    }
+
+    for (const auto& kv : mapRankIdconnectedSockets_) {
+        CHK_PRT_RET(socketManager_->WaitLinkEstablish(kv.second) != HCCL_SUCCESS,
+            HCCL_ERROR("[SymmetricMemoryAgent][EstablishSockets] tag[%s] socket establish failed", kv.second->GetTag().c_str()),
+            HCCL_E_INTERNAL);
+    }
+    return HCCL_SUCCESS;
+}
+
+std::string SymmetricMemoryAgent::GenerateSocketTag(u32 localRank, u32 remoteRank)
+{
+    u32 small = localRank;
+    u32 large = remoteRank;
+
+    if (localRank > remoteRank) {
+        small = remoteRank;
+        large = localRank;
+    }
+
+    // Socket构造规则：前缀 + identifier + small + large
+    std::string tag = STR_IPC_MEM_EXCHANGE + "_" + identifier_ 
+        + "_" + std::to_string(small) + ":" + std::to_string(large);
+    return tag;
+}
+
+HcclResult SymmetricMemoryAgent::ExchangeInfo(void *inputPtr, void *outputPtr, u64 inputSize)
+{
+    CHK_PTR_NULL(inputPtr);
+    CHK_PTR_NULL(outputPtr);
+    CHK_PRT_RET(inputSize == 0, HCCL_ERROR("Input size is 0"), HCCL_E_PARA);
+    // 校验 inputSize 是否超过协议载荷上限
+    CHK_PRT_RET(inputSize > PACKET_DATA_MAX_LEN, 
+        HCCL_ERROR("Input size %lu exceeds max payload %u", inputSize, PACKET_DATA_MAX_LEN), HCCL_E_PARA);
+    // 校验是否建链成功
+    CHK_PRT_RET(mapRankIdconnectedSockets_.find(rightRank_) == mapRankIdconnectedSockets_.end(),
+        HCCL_ERROR("[ExchangeInfo] rightRank_%u socket not found in map", rightRank_), HCCL_E_INTERNAL);
+    CHK_PRT_RET(mapRankIdconnectedSockets_.find(leftRank_) == mapRankIdconnectedSockets_.end(),
+        HCCL_ERROR("[ExchangeInfo] leftRank_%u socket not found in map", leftRank_), HCCL_E_INTERNAL);
+
+    HCCL_INFO("[SymmetricMemoryAgent] start to ExchangeInfo, inputPtr[%p], outputPtr[%p], inputSize[%llu]", inputPtr, outputPtr, inputSize);
+    
+    // 重置本轮状态
+    outputDataPtr_ = static_cast<u8*>(outputPtr);
+    currentInputSize_ = inputSize; // 记录实际有效长度
+    collectedCount_ = 0;
+    // 本地数据处理：先把自己的一份拷到 Output 对应位置
+    u8* selfDstPtr = outputDataPtr_ + (userRank_ * inputSize);
+    CHK_SAFETY_FUNC_RET(memcpy_s(selfDstPtr, inputSize, inputPtr, inputSize));
+    collectedCount_++;
+
+    Packet dataPkt;
+    dataPkt.type = MsgType::MSG_TYPE_DATA;
+    dataPkt.rankId = userRank_;
+    CHK_SAFETY_FUNC_RET(memset_s(dataPkt.data, PACKET_DATA_MAX_LEN, 0, PACKET_DATA_MAX_LEN));
+    CHK_SAFETY_FUNC_RET(memcpy_s(dataPkt.data, PACKET_DATA_MAX_LEN, inputPtr, inputSize));
+    {
+        std::lock_guard<std::mutex> lock(queueMutex_);
+        requestQueue_.push(dataPkt);
+    }
+    isProcessingTask_ = true;
+
+    CHK_RET(WaitForCollectionComplete());
+    HCCL_INFO("[SymmetricMemoryAgent] ExchangeInfo end");
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemoryAgent::WaitForCollectionComplete()
+{
+    std::unique_lock<std::mutex> lock(completionMutex_);
+    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
+    auto status = completionCv_.wait_for(lock, timeout);
+    if (status == std::cv_status::timeout) {
+        HCCL_ERROR("[SymmetricMemoryAgent] ExchangeInfo Timeout! Collected: %u/%u",
+            collectedCount_.load(), rankSize_);
+        return HCCL_E_TCP_TRANSFER;
+    }
+    return HCCL_SUCCESS;
+}
+
+void SymmetricMemoryAgent::DealWithRequest()
+{
+    if (hrtSetDevice(deviceLogicId_) != HCCL_SUCCESS) {
+        return;
+    }
+
+    std::vector<u8> leftRecvBuf(PACKET_TOTAL_LEN, 0);
+    u32 leftRecvLen = 0;
+
+    while (threadRun_) {
+        if (isProcessingTask_) {
+            if (collectedCount_ < rankSize_) {
+                u64 received = 0;
+                std::unique_lock<std::mutex> lock(socketMutex_);
+                HcclResult ret = mapRankIdconnectedSockets_[leftRank_]->IRecv(
+                    leftRecvBuf.data() + leftRecvLen, PACKET_TOTAL_LEN - leftRecvLen, received);
+                
+                CHK_PRT_CONT((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN),
+                    HCCL_ERROR("[SymmetricMemoryAgent][DealWithRequest] IRecv failed, ret[%d] remoteRank[%u] receivedSize[%llu]",
+                    ret, leftRank_, leftRecvLen));
+
+                leftRecvLen += received;
+                if (leftRecvLen == PACKET_TOTAL_LEN) {
+                    Packet* pkt = reinterpret_cast<Packet*>(leftRecvBuf.data());
+                    ProcessReceivedPacket(*pkt);
+                    leftRecvLen = 0;
+                }
+            }
+            std::lock_guard<std::mutex> lock(queueMutex_);
+            if (!requestQueue_.empty()) {
+                Packet pkt = requestQueue_.front();
+                std::unique_lock<std::mutex> sockLock(socketMutex_);
+                HcclResult ret = mapRankIdconnectedSockets_[rightRank_]->Send((u8*)&pkt, PACKET_TOTAL_LEN);
+                if (ret == HCCL_SUCCESS) {
+                    requestQueue_.pop();
+                }else {
+                    HCCL_ERROR("[SymmetricMemoryAgent][DealWithRequest] Data(from rank[%u]) Send to rank[%u] failed.", pkt.rankId, rightRank_);
+                }
+            }
+            // 检查是否完全结束, 退出条件: 数据全齐 && 队列空闲
+            if (requestQueue_.empty() && collectedCount_ == rankSize_) {
+                std::unique_lock<std::mutex> lock(completionMutex_);
+                HCCL_INFO("[SymmetricMemoryAgent] ExchangeInfo Complete.");
+                isProcessingTask_ = false;
+                completionCv_.notify_all();
+            }
+        }
+        SaluSleep(USLEEP_ONE_THOUSAND);
+    }
+    
+    hrtResetDevice(deviceLogicId_);
+}
+
+HcclResult SymmetricMemoryAgent::ProcessReceivedPacket(Packet& pkt) {
+    if (pkt.rankId < rankSize_ && pkt.rankId != userRank_) {
+        u8* dest = outputDataPtr_ + (pkt.rankId * currentInputSize_);
+        CHK_SAFETY_FUNC_RET(memcpy_s(dest, currentInputSize_, pkt.data, currentInputSize_));
+        collectedCount_++;
+    }
+    HCCL_INFO("[SymmetricMemoryAgent][ProcessReceivedPacket] Data Recv from rank[%u]. Collected[%u / %u].",
+        pkt.rankId, collectedCount_.load(), rankSize_);
+    // Ring 转发逻辑：如果数据不是自己的，也不是右边Rank发出的(转了一圈)，则转发给右边
+    if (pkt.rankId != userRank_ && pkt.rankId != rightRank_) {
+        std::lock_guard<std::mutex> lock(queueMutex_);
+        requestQueue_.push(pkt);
+    }
+    return HCCL_SUCCESS;
+}
+} // namespace hccl
\ No newline at end of file

```

### src/framework/communicator/impl/symmetric_memory/symmetric_memory_agent.h
```diff
@@ -0,0 +1,101 @@
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
+#ifndef SYMMETRIC_MEMORY_AGENT_H
+#define SYMMETRIC_MEMORY_AGENT_H
+
+#include <vector>
+#include <map>
+#include <mutex>
+#include <thread>
+#include <atomic>
+#include <queue>
+#include <condition_variable>
+#include <cstring>
+#include <memory>
+
+#include "hccl/hccl_types.h"
+#include "hccl/base.h"
+#include "hccl_socket_manager.h" 
+#include "adapter_hccp_common.h"
+
+namespace hccl {
+
+// 协议常量
+constexpr u32 PACKET_DATA_MAX_LEN = 144;
+constexpr u32 PACKET_TOTAL_LEN = 152;     // 4(Type) + 4(Rank) + 144(Data)
+
+// 消息类型
+enum class MsgType : u32 {
+    MSG_TYPE_DATA = 0,
+};
+
+// 协议包结构
+struct Packet {
+    MsgType type;
+    u32 rankId;
+    u8 data[PACKET_DATA_MAX_LEN];
+};
+
+class SymmetricMemoryAgent {
+public:
+    SymmetricMemoryAgent(const std::unique_ptr<HcclSocketManager> &socketManager, u32 devicePhyId,
+        s32 deviceLogicId, const HcclIpAddress &localVnicIp, const std::vector<RankInfo> &rankInfoList, u32 userRank,
+        bool useSuperPodMode, const std::string &identifier);
+    virtual ~SymmetricMemoryAgent();
+
+    HcclResult Init();
+    HcclResult ExchangeInfo(void *inputPtr, void *outputPtr, u64 inputSize);
+
+private:
+    HcclResult InitRecvThread();
+    HcclResult EstablishSockets();
+    std::string GenerateSocketTag(u32 localRank, u32 remoteRank);
+
+    void DealWithRequest();
+    HcclResult WaitForCollectionComplete();
+    HcclResult ProcessReceivedPacket(Packet& pkt);
+
+    HcclNetDevCtx vnicPortCtx_{nullptr};
+    const std::unique_ptr<HcclSocketManager> &socketManager_;
+    u32 devicePhyId_;
+    s32 deviceLogicId_;
+    HcclIpAddress localVnicIp_;
+    const std::vector<RankInfo> &rankInfoList_;
+    u32 userRank_;
+    u32 leftRank_;
+    u32 rightRank_;
+    u32 rankSize_;
+    bool useSuperPodMode_;
+    std::string identifier_{};
+
+    std::unique_ptr<std::thread> recvThread_;
+    std::atomic<bool> threadRun_{false};
+    
+    std::mutex socketMutex_;
+    std::unordered_map<u32, std::shared_ptr<HcclSocket>> mapRankIdconnectedSockets_;
+    std::unordered_map<u32, u32> mapRankId2DevPhyId_;
+
+    u8* outputDataPtr_{nullptr}; 
+    u64 currentInputSize_{0};      // 记录当前交换数据的实际有效长度
+    std::atomic<u32> collectedCount_{0}; 
+
+    std::queue<Packet> requestQueue_;
+    std::mutex queueMutex_;
+
+    // 完成通知 (用于 WaitForCollectionComplete)
+    std::mutex completionMutex_;
+    std::condition_variable completionCv_;
+
+    std::atomic<bool> isProcessingTask_{false};
+};
+
+} // namespace hccl
+#endif // SYMMETRIC_MEMORY_AGENT_H
\ No newline at end of file

```

### src/framework/device/framework/aicpu_communicator.cc
(diff 过长，已截断)

> 注: 以下非 C/C++ 文件未纳入审查: src/framework/common/src/CMakeLists.txt, src/framework/communicator/impl/CMakeLists.txt, src/framework/communicator/impl/symmetric_memory/CMakeLists.txt, src/framework/device/framework/CMakeLists.txt, test/ut/framework/communicator/impl/CMakeLists.txt, test/ut/framework/communicator/impl/symmetric_memory/CMakeLists.txt
