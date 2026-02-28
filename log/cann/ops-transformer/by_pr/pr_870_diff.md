# PR #870: Support symmetric memory for aicpu unflod mode

- 作者: linzhenkang
- 分支: symmetric_temp -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/870
- 描述: Support symmetric memory for aicpu unflod mode

## 变更文件 (40 个, 其中 C/C++ 文件 34 个)

- [modified] include/hccl/hccl_comm.h (+41, -0) *
- [modified] include/hccl/hccl_types.h (+6, -1) *
- [modified] src/algorithm/impl/operator/all_gather_operator.cc (+2, -2) *
- [modified] src/algorithm/impl/operator/all_reduce_operator.cc (+2, -2) *
- [modified] src/algorithm/impl/operator/broadcast_operator.cc (+2, -2) *
- [modified] src/algorithm/impl/operator/reduce_scatter_operator.cc (+2, -2) *
- [modified] src/algorithm/pub_inc/coll_alg_param.h (+5, -0) *
- [modified] src/framework/CMakeLists.txt (+3, -0)
- [modified] src/framework/cluster_maintenance/health/heartbeat/heartbeat.cc (+6, -0) *
- [modified] src/framework/common/src/CMakeLists.txt (+1, -0)
- [added] src/framework/common/src/hccl_mem_alloc.cc (+87, -0) *
- [added] src/framework/common/src/hccl_mem_alloc.h (+23, -0) *
- [modified] src/framework/common/src/launch_aicpu.cc (+1, -0) *
- [modified] src/framework/communicator/hccl_comm.cc (+13, -0) *
- [modified] src/framework/communicator/impl/CMakeLists.txt (+2, -1)
- [modified] src/framework/communicator/impl/aclgraph/zero_copy_acl_graph.cc (+1, -1) *
- [modified] src/framework/communicator/impl/hccl_communicator.cc (+4, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator.h (+6, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator_device.cc (+16, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator_host.cc (+73, -0) *
- [added] src/framework/communicator/impl/symmetric_memory/CMakeLists.txt (+8, -0)
- [added] src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc (+622, -0) *
- [added] src/framework/communicator/impl/symmetric_memory/symmetric_memory.h (+116, -0) *
- [modified] src/framework/device/framework/CMakeLists.txt (+1, -0)
- [modified] src/framework/device/framework/aicpu_communicator.cc (+90, -8) *
- [modified] src/framework/device/framework/aicpu_communicator.h (+5, -0) *
- [modified] src/framework/device/framework/aicpu_hccl_process.cc (+5, -0) *
- [added] src/framework/device/framework/aicpu_symmetric_memory.cc (+33, -0) *
- [added] src/framework/device/framework/aicpu_symmetric_memory.h (+19, -0) *
- [modified] src/framework/device/hccl_aicpu_interface.cc (+3, -1) *
- [modified] src/framework/inc/hccl_comm_pub.h (+3, -0) *
- [modified] src/framework/op_base/src/op_base.cc (+41, -0) *
- [modified] src/platform/common/externalinput.cc (+42, -0) *
- [modified] src/platform/common/externalinput.h (+6, -0) *
- [modified] src/pub_inc/aicpu_operator_pub.h (+5, -0) *
- [modified] test/llt/aicpu_kfc/stub/llt_aicpu_kfc_stub.cc (+26, -0) *
- [modified] test/llt/ut/single_test/impl/CMakeLists.txt (+1, -0)
- [added] test/llt/ut/single_test/impl/ut_hccl_mem_alloc.cc (+195, -0) *
- [modified] test/stub/framework_stub/llt_hccl_stub.cc (+26, -0) *
- [modified] test/ut/stub/llt_hccl_stub.cc (+26, -0) *

## Diff 内容

### include/hccl/hccl_comm.h
```diff
@@ -299,6 +299,47 @@ extern HcclResult HcclSetCommConfig(HcclComm comm, HcclConfig config, HcclConfig
 extern HcclResult HcclGetCommConfig(HcclComm comm, HcclConfig config, HcclConfigValue *configValue);
 #endif
 
+/**
+ * @brief Register a memory window for HCCL communication.
+ *
+ * @param comm A pointer identifying the communication resource based on.
+ * @param ptr A pointer identifying the user memory address.
+ * @param size A size_t identifying the size of memory window.
+ * @param winHandle A pointer identifying the registered memory window handle.
+ * @param flags The flag of this memory window, now only support 0
+ * @return HcclResult
+ */
+extern HcclResult HcclCommWindowRegister(HcclComm comm, void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags);
+
+/**
+ * @brief Deregister a memory window for HCCL communication.
+ *
+ * @param comm A pointer identifying the communication resource based on.
+ * @param winHandle A pointer identifying the registered memory window handle.
+ * @return HcclResult
+ */
+extern HcclResult HcclCommWindowDeRegister(HcclComm comm, HcclWindow winHandle);
+
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
+
 #ifdef __cplusplus
 }
 #endif // __cplusplus

```

### include/hccl/hccl_types.h
```diff
@@ -61,7 +61,12 @@ typedef void *HcclComm;
 typedef void *HcclConn;
 
 /**
- * @brief HCCL Reduction operation
+ * @brief handle to HCCL Window
+ */
+typedef void *HcclWindow;
+
+/**
+ * @brief HCCL Reduction opperation
  */
 typedef enum {
     HCCL_REDUCE_SUM = 0,    /**< sum */

```

### src/algorithm/impl/operator/all_gather_operator.cc
```diff
@@ -352,8 +352,8 @@ HcclResult AllGatherOperator::SelectAlgfor91093(const OpParam& param, std::strin
         algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
     } else if (smallCountOptimMultiServer || smallCountOptimSingleServer) {
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
@@ -678,8 +678,8 @@ HcclResult AllReduceOperator::SelectAlgfor91093(const OpParam& param, std::strin
         algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
     } else if (smallCountOptimSingleServer) {
         algName = "AllReduceMeshSmallCountExecutor";
-    } else if (param.supportZeroCopy &&
-        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize > HCCL_MID_COUNT_16_MB * serverNum_)) {
+    } else if (param.supportSymmetricMemory || (param.supportZeroCopy &&
+        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize > HCCL_MID_COUNT_16_MB * serverNum_))) {
         algName = "AllReduceRingZerocopyExecutor";
     } else {
         if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {

```

### src/algorithm/impl/operator/broadcast_operator.cc
```diff
@@ -233,8 +233,8 @@ HcclResult BroadCastOperator::SelectAlgfor91093(const OpParam& param, std::strin
         algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
     } else if (smallCountOptimSingleServer) {
         algName = "BroadCastSmallCountExecutor";
-    } else if (param.supportZeroCopy &&
-        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize > HCCL_MID_COUNT_16_MB * serverNum_)) {
+    } else if (param.supportSymmetricMemory || (param.supportZeroCopy &&
+        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize > HCCL_MID_COUNT_16_MB * serverNum_))) {
         algName = "BroadCastRingZerocopyExecutor";
     } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
         algName = "BroadCastRingFor91093Executor";

```

### src/algorithm/impl/operator/reduce_scatter_operator.cc
```diff
@@ -478,8 +478,8 @@ HcclResult ReduceScatterOperator::SelectAlgfor91093(const OpParam& param, std::s
         (smallCountOptimMultiServer && isPowOfTwo &&
         (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] * serverNum_ <= smallCountMultiServerThreshold))) {
         algName = "ReduceScatterDeterExecutor";
-    } else if (param.supportZeroCopy && isSupportInlineReduce &&    // 不申请scratch ==> 不支持非InlineReduce
-        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB)) {
+    } else if (isSupportInlineReduce && param.supportSymmetricMemory || (param.supportZeroCopy &&    // isSupportInlineReduce：不申请scratch ==> 不支持非InlineReduce
+        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB))) {
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
+    void* inputWindow = nullptr;
+    u64 inputOffset = 0;
+    void* outputWindow = nullptr;
+    u64 outputOffset = 0;
 
     inline HcclDataType GetDataType() const
     {

```

### src/framework/cluster_maintenance/health/heartbeat/heartbeat.cc
```diff
@@ -575,6 +575,10 @@ HcclResult Heartbeat::UnRegisterRanks(const std::string &group)
         HCCL_INFO("[%s]group[%s] UnregisterRanks Completed.", __func__, group.c_str());
     }
 
+    for (const auto it : groupMap_) {
+        HCCL_INFO("[Heartbeat][UnRegisterToHeartBeat] group[%s]", it.first.c_str());
+    }
+    HCCL_INFO("[Heartbeat][UnRegisterToHeartBeat] groupMap_.size[%d]", groupMap_.size());
     if (groupMap_.size() == 0) {
         HCCL_RUN_INFO("[%s]Entry HeartBeat DeInit.", __func__);
         CHK_RET(DeInit());
@@ -584,6 +588,7 @@ HcclResult Heartbeat::UnRegisterRanks(const std::string &group)
 
 void Heartbeat::UnRegisterToHeartBeat(DevType devType, const std::string &commIdentifier)
 {
+    HCCL_INFO("[Heartbeat][UnRegisterToHeartBeat] commIdentifier[%s]", commIdentifier.c_str());
     if (Is310PDevice() || devType == DevType::DEV_TYPE_310P3) {
         return;
     }
@@ -595,6 +600,7 @@ void Heartbeat::UnRegisterToHeartBeat(DevType devType, const std::string &commId
 }
 void Heartbeat::UnRegisterToHeartBeat(DevType devType, const std::string &commIdentifier, const std::string &tag)
 {
+    HCCL_INFO("[Heartbeat][UnRegisterToHeartBeat] commIdentifier[%s], tag[%s]", commIdentifier.c_str(), tag.c_str());
     if (Is310PDevice() || devType == DevType::DEV_TYPE_310P3) {
         return;
     }

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
@@ -0,0 +1,23 @@
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
+#include "hccl_comm_pub.h"
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

### src/framework/communicator/hccl_comm.cc
```diff
@@ -1493,5 +1493,18 @@ HcclResult hcclComm::GetKFCWorkSpace(void **addr, uint64_t *size)
     return HCCL_SUCCESS;
 }
 
+HcclResult hcclComm::RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags)
+{
+    CHK_SMART_PTR_NULL(communicator_);
+    CHK_RET(communicator_->RegisterWindow(ptr, size, winHandle, flags));
+    return HCCL_SUCCESS;
+}
+
+HcclResult hcclComm::DeregisterWindow(HcclWindow winHandle)
+{
+    CHK_SMART_PTR_NULL(communicator_);
+    CHK_RET(communicator_->DeregisterWindow(winHandle));
+    return HCCL_SUCCESS;
+}
 
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
+        opTilingData->inputWindow = reinterpret_cast<u64>(opParam.inputWindow);
+        opTilingData->inputOffset = opParam.inputOffset;
+        opTilingData->outputWindow = reinterpret_cast<u64>(opParam.outputWindow);
+        opTilingData->outputOffset = opParam.outputOffset;
         return HCCL_SUCCESS;
     }
 

```

### src/framework/communicator/impl/hccl_communicator.h
```diff
@@ -55,6 +55,7 @@
 #include "independent_op.h"
 #include "comm_config_pub.h"
 #include "new/hccl_dispatcher_ctx.h"
+#include "symmetric_memory/symmetric_memory.h"
 
 namespace hccl {
 using ServRankInfo_t = std::map<std::string, std::vector<RankInfo_t> >;
@@ -474,6 +475,9 @@ public:
     HcclResult GroupSyncMainstream(std::unordered_map<u32, std::vector<u64>> &sendIdx2Byte, std::unordered_map<u32, std::vector<u64>> &recvIdx2Byte);
     HcclResult GroupSubstreamsSync();
     void SetReleaseChannel(std::function<HcclResult()> releaseChannel);
+    HcclResult RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags);
+    HcclResult DeregisterWindow(HcclWindow winHandle);
+    HcclResult InitSymmetricMemory();
 private:
 
     bool IsEnableRoce();
@@ -773,6 +777,7 @@ private:
     bool GetSupportHDCommunicate();
     HcclResult InitOpRetry();
     HcclResult InitOpResPara();
+    bool IsSupportSymmetricMemory(OpParam &opParam);
     bool IsSupportZeroCopy(const OpParam &opParam);
     HcclResult PrepareZeroCopy(const std::string &algName, const AlgDesc &algDesc, OpParam &opParam);
     HcclResult UpdateZeroCopy(const OpParam &opParam, const AlgResourceResponse &algResource);
@@ -1110,6 +1115,7 @@ private:
     std::function<bool()> getAicpuCommState_; // 获取自定义算子aicpu通信域是否初始化
     bool isInvalidComm_ { false };
     std::function<HcclResult()> releaseChannel_ = nullptr;
+    std::unique_ptr<SymmetricMemory> symmetricMemory_;
 };
 }  // end namespace hccl
 #endif  // HCCL_IMPL_BASE_H

```

### src/framework/communicator/impl/hccl_communicator_device.cc
```diff
@@ -1590,4 +1590,20 @@ namespace hccl
     {
         return cclBufferManager_;
     }
+
+    HcclResult HcclCommunicator::InitSymmetricMemory()
+    {
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags)
+    {
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult HcclCommunicator::DeregisterWindow(HcclWindow winHandle)
+    {
+        return HCCL_SUCCESS;
+    }
+
 }

```

### src/framework/communicator/impl/hccl_communicator_host.cc
```diff
@@ -50,6 +50,7 @@
 #include "hccl_group_utils.h"
 #include "snapshot_control.h"
 #include "comm_topo_desc.h"
+#include "externalinput.h"
 
 using namespace std;
 constexpr u32 MODULE_NUM_FOUR = 4;
@@ -168,6 +169,10 @@ namespace hccl
     {
         HCCL_DEBUG("Enter ~HcclCommunicator.");
 
+        if(symmetricMemory_ != nullptr){
+            symmetricMemory_ = nullptr;
+        }
+
         DeinitZeroCopyMemoryAgent(true);
         if (!isInvalidComm_) {
             (void)DestroyAicpuComm();
@@ -359,6 +364,7 @@ namespace hccl
         CHK_RET(rankGraph_.Init(rankTable, topoAttr));
         CHK_RET(SaveTopoDesc(params.identifier));
         CHK_RET(RegisterToSnapshot());
+        CHK_RET(InitSymmetricMemory());
         return HCCL_SUCCESS;
     }
 
@@ -387,6 +393,7 @@ namespace hccl
         attrCollector_.GetTopoAttr(topoAttr);
         CHK_RET(rankGraph_.Init(topoAttr));
         CHK_RET(SaveTopoDesc(params.identifier));
+        CHK_RET(InitSymmetricMemory());
         return HCCL_SUCCESS;
     }
 
@@ -569,6 +576,42 @@ namespace hccl
         return HCCL_SUCCESS;
     }
 
+    bool HcclCommunicator::IsSupportSymmetricMemory(OpParam &opParam)
+    {
+        HCCL_INFO("[%s] aicpuUnfold[%d], workflowMode[%d], deviceType[%d], "
+            "deviceNumPerAggregation_[%d], multiModuleDiffDeviceNumMode_[%d], tag[%s].",
+            __func__, opParam.aicpuUnfoldMode, GetWorkflowMode(), deviceType_,
+            deviceNumPerAggregation_, multiModuleDiffDeviceNumMode_, opParam.tag.c_str());
+
+        // 只支持aicpu展开、非重执行、单算子模式、910_93芯片
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
+        HcclResult ret = symmetricMemory_->FindSymmetricWindow(opParam.inputPtr, opParam.inputSize, &opParam.inputWindow, opParam.inputOffset);
+        CHK_PRT_RET(ret != HCCL_SUCCESS || opParam.inputWindow == nullptr,
+                    HCCL_INFO("[%s] input[%p] size[%llu] is not support symmetric memory", __func__, opParam.inputPtr, opParam.inputSize), false);
+        ret = symmetricMemory_->FindSymmetricWindow(opParam.outputPtr, opParam.outputSize, &opParam.outputWindow, opParam.outputOffset);
+        CHK_PRT_RET(ret != HCCL_SUCCESS || opParam.outputWindow == nullptr,
+                    HCCL_INFO("[%s] output[%p] size[%llu] is not support symmetric memory", __func__, opParam.outputPtr, opParam.outputSize), false);
+        
+        HCCL_INFO("[HcclCommunicator][IsSupportSymmetricMemory] opParam.inputPtr[%p], inputOffset[%llu], inputWindow[%p]", opParam.inputPtr, opParam.inputOffset, opParam.inputWindow);
+        HCCL_INFO("[HcclCommunicator][IsSupportSymmetricMemory] opParam.outputPtr[%p], outputOffset[%llu], outputWindow[%p]", opParam.outputPtr, opParam.outputOffset, opParam.outputWindow);
+
+        return true;
+    }
+
     bool HcclCommunicator::IsSupportZeroCopy(const OpParam &opParam)
     {
         HCCL_INFO("[%s] aicpuUnfold[%d], workflowMode[%d], deviceType[%d], "
@@ -603,9 +646,16 @@ namespace hccl
     HcclResult HcclCommunicator::PrepareZeroCopy(const std::string &algName, const AlgDesc &algDesc, OpParam &opParam)
     {
         if (!algDesc.isZeroCopy) {
+            opParam.supportSymmetricMemory = false;
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
@@ -4257,6 +4307,7 @@ namespace hccl
         }
 
         ForceProf(opParam.isCapture);
+        opParam.supportSymmetricMemory = IsSupportSymmetricMemory(opParam);
         opParam.supportZeroCopy = !commConfig_.GetConfigDeterministic() && IsSupportZeroCopy(opParam);
         opParam.aclGraphZeroCopyEnable = GetConfigAclGraphZeroCopyEnable();
         bool isInGraphCaptureZeroCopy = false;
@@ -6952,6 +7003,7 @@ namespace hccl
         opTilingData->isZeroCopy = opParam.isZeroCopy;
         opTilingData->isCapture = opParam.isCapture;
         opTilingData->orderLaunchMode = GetOrderLaunchMode(opParam.isCapture);
+        opTilingData->isSymmetricMemory = opParam.supportSymmetricMemory;
         // 有没有存在对应的Notify
         CHK_RET(InitAndCheckAicpuOrderNotify(opTilingData->orderLaunchMode));
         CHK_RET(BuildHierarchicalAlgOption(opTilingData->ahcConfInfo));
@@ -8767,4 +8819,25 @@ namespace hccl
     {
         return cclBufferManager_;
     }
+
+    HcclResult HcclCommunicator::InitSymmetricMemory()
+    {
+        HCCL_RUN_INFO("InitSymmetricMemory, commHandle_[%p], comm identifier[%s], userRank[%u], userRankSize[%u], stride[%llu], devicePhyId[%u].",
+                      commHandle_, identifier_.c_str(), userRank_, userRankSize_, GetExternalInputSymmetricMemoryStride(), devicePhyId_);
+        symmetricMemory_ = std::make_unique<SymmetricMemory>(commHandle_, userRank_, userRankSize_,
+            GetExternalInputSymmetricMemoryStride() * 1024 * 1024 * 1024, devicePhyId_);
+        CHK_SMART_PTR_NULL(symmetricMemory_);
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags)
+    {
+        return symmetricMemory_->RegisterSymmetricMem(ptr, size, winHandle);
+    }
+
+    HcclResult HcclCommunicator::DeregisterWindow(HcclWindow winHandle)
+    {
+        return symmetricMemory_->DeregisterSymmetricMem(winHandle);
+    }
+
 }

```

### src/framework/communicator/impl/symmetric_memory/symmetric_memory.cc
```diff
@@ -0,0 +1,622 @@
+/*
+ * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd. All Rights Reserved.
+ * This file is a part of the CANN Open Software.
+ * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
+ * Please refer to the License for details. You may not use this file except in compliance with the License.
+ * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
+ * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
+ * See LICENSE in the root of the software repository for the full text of the License.
+ */
+
+#include "symmetric_memory.h"
+#include <algorithm> // for std::max
+#include <cstddef>
+#include <list>      // for SimpleVaAllocator
+#include "hccl_comm.h"
+
+namespace hccl 
+{
+constexpr uint32_t ROOTINFO_INDENTIFIER_MAX_LENGTH = 128;
+
+/**
+ * @brief (内部) 简单的VA空间分配器
+ * (... 此处省略 SimpleVaAllocator 的实现，与上一版相同 ...)
+ */
+class SymmetricMemory::SimpleVaAllocator {
+    struct FreeBlock {
+        size_t offset;
+        size_t size;
+    };
+    std::list<FreeBlock> freeList_; // 按offset排序的空闲块
+    std::mutex mutex_;
+    size_t totalSize_;
+
+public:
+    SimpleVaAllocator() : totalSize_(0) {}
+    ~SimpleVaAllocator() { Destroy(); }
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
+        if (size == 0) return HCCL_E_PARA;
+        totalSize_ = size;
+        freeList_.push_back({0, (size_t)size});
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
+                    // std::list::insert inserts BEFORE the iterator. 
+                    // std::next(to_erase) points to the element AFTER current. 
+                    // So inserting before next element puts it after current. Correct.
+                    freeList_.insert(std::next(to_erase), 
+                                    {alignedOffset + (size_t)size, backPad});
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
+                // 增加调试：为什么这个块不满足？
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
+        // ... Release 代码保持不变，也可以加上日志 ...
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
+        // ... (省略中间重叠检查代码，与原版一致) ...
+        
+        // 插入新释放的块
+        auto newIt = freeList_.insert(it, {offset, size});
+        HCCL_INFO("[VAAllocator] Releasing [0x%zx, size %zu]", offset, size);
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
+SymmetricMemory::SymmetricMemory(HcclComm &comm, u32 rank, u32 rankSize, size_t stride, u32 devicePhyId)
+    : commHandle_(comm), 
+      rank_(rank),
+      rankSize_(rankSize),
+      devicePhyId_(devicePhyId),
+      stride_(stride), 
+      heapBase_(nullptr), 
+      granularity_(0),
+      vaAllocator_(new (std::nothrow) SimpleVaAllocator()),
+      initResult_(HCCL_E_INTERNAL) {}
+
+SymmetricMemory::~SymmetricMemory() 
+{
+    HCCL_INFO("[SymmetricMemory][~SymmetricMemory] begin");
+    for (auto& pair : windowMap_) {
+        // pair.first is void* devWin
+        if (pair.first) hrtFree(pair.first);
+    }
+    windowMap_.clear();
+    sortedWindows_.clear();
+
+    if (vaAllocator_) {
+        vaAllocator_->Destroy();
+    }
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
+HcclResult SymmetricMemory::SymmetricMemory::Init() 
+{
+    // 0. 检查Pimpl是否构造成功
+    CHK_SMART_PTR_NULL(vaAllocator_);
+
+    // 1. 获取内存映射的粒度
+    aclrtPhysicalMemProp prop = {
+        ACL_MEM_HANDLE_TYPE_NONE,
+        ACL_MEM_ALLOCATION_TYPE_PINNED,
+        ACL_HBM_MEM_HUGE,
+        {0, ACL_MEM_LOCATION_TYPE_DEVICE},
+        0
+    };
+
+    if (aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity_) != ACL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][Init] aclrtMemGetAllocationGranularity failed.");
+        return HCCL_E_INTERNAL;
+    }
+    if (granularity_ == 0) {
+        HCCL_ERROR("[SymmetricMemory][Init] Invalid memory granularity: 0");
+        return HCCL_E_INTERNAL;
+    }
+    
+    // 2. 预留总的VA空间
+    // 每个rank都预留一个总大小为 totalHeapSize 的VA空间。
+    // heapBase_ 在不同rank上可能是不同的，这是预期行为。
+    size_t totalHeapSize = (size_t)stride_ * rankSize_;
+    if (stride_ % granularity_ != 0) {
+        HCCL_ERROR("[SymmetricMemory][Init] Stride %u is not a multiple of granularity %zu.", stride_, granularity_);
+        return HCCL_E_PARA;
+    }
+    
+    // 默认大页对齐
+    if (aclrtReserveMemAddress(&heapBase_, totalHeapSize, 0, nullptr, 1) != ACL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][Init] aclrtReserveMemAddress failed to reserve %zu bytes. stride: %u, rankSize: %u.",
+                   totalHeapSize, stride_, rankSize_);
+        return HCCL_E_INTERNAL;
+    }
+
+    // 5. 初始化VA分配器 (管理本地rank的stride_大小空间，即管理偏移量)
+    // 这是一个集合调用，所有rank上的vaAllocator_状态将保持一致（前提是 SimpleVaAllocator 是确定性的）
+    CHK_RET(vaAllocator_->Init(stride_));
+
+    CHK_RET(CreateSubComm());
+
+    HCCL_INFO("[SymmetricMemory][Init] SymmetricMemory initialized. Rank[%u], Local Heap Base: %p, Stride: %u, Ranks: %u.",
+               rank_, heapBase_, stride_, rankSize_);
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
+        return NULL;
+    }
+
+    ret = RegisterSymmetricMem(ptr, size, &devWin);
+    if (ret != HCCL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][AllocSymmetricMem] RegisterSymmetricMem failed for ptr[%p], size[%u].", ptr, size);
+        (void)HcclMemFree(ptr);
+        return NULL;
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
+    CHK_RET(HcclMemFree(userPtr));
+    CHK_RET(DeregisterSymmetricMem(devWin));
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
+HcclResult SymmetricMemory::RegisterSymmetricMem(void* ptr, size_t size, void** devWin)
+{
+    CHK_RET(EnsureInit());
+    if (ptr == nullptr || size == 0) {
+        HCCL_ERROR("[SymmetricMemory][RegisterSymmetricMem] Invalid parameters.");
+        return HCCL_E_PARA;
+    }
+
+    // 打印当前注册请求的关键信息
+    HCCL_INFO("[SymmetricMemory][RegisterSymmetricMem] Request: ptr=%p, size=%zu, granularity=%zu", 
+        ptr, size, granularity_);
+
+    void* baseUserVa = ptr;
+    size_t baseVaSize = size;
+    // if(aclrtMemGetAddressRange(ptr, baseUserVa, &baseVaSize) != 0) {
+    //     HCCL_ERROR("[SymmetricMemory][RegisterSymmetricMem] aclrtMemGetAddressRange failed for ptr[%p], size[%zu]. ", ptr, size);
+    //     return HCCL_E_PARA;
+    // }
+    CHK_PTR_NULL(baseUserVa);
+    aclrtDrvMemHandle paHandle;
+    if (aclrtMemRetainAllocationHandle(baseUserVa, &paHandle) != 0) {
+        HCCL_ERROR("[SymmetricMemory][RegisterSymmetricMem] MemRetainAllocationHandle failed for ptr[%p], size[%zu]. ", ptr, size);
+        return HCCL_E_PARA;
+    }
+    size_t alignedBaseSize = (baseVaSize + granularity_ - 1) & ~(granularity_ - 1);
+
+    if (reinterpret_cast<uintptr_t>(ptr) + size > reinterpret_cast<uintptr_t>(baseUserVa) + alignedBaseSize) {
+        HCCL_ERROR("[SymmetricMemory][RegisterSymmetricMem] ptr=%p size=%zu exceeds  block [baseUserVa=%p, size=%zu]", 
+           ptr, size, baseUserVa, alignedBaseSize);
+        return HCCL_E_PARA;
+    }
+
+    HCCL_INFO("[SymmetricMemory][RegisterSymmetricMem] Retained paHandle[%p] for baseUserVa[%p], alignedBaseSize[%zu]. Total Stride: %zu",
+        paHandle, baseUserVa, alignedBaseSize, stride_);
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
+        if (vaAllocator_->Reserve(alignedBaseSize, granularity_, offset) != HCCL_SUCCESS) {
+            HCCL_ERROR("[SymmetricMemory][RegisterSymmetricMem] Failed to reserve VA space. "
+                "Req alignedSize: %zu (0x%zx), Align: %zu. Total Stride: %zu. "
+                "Is fragmentation too high or stride too small?", 
+                alignedBaseSize, alignedBaseSize, granularity_, stride_);
+            return HCCL_E_MEMORY;
+        }
+
+        HcclResult ret = RegisterInternal(paHandle, offset, alignedBaseSize);
+        if (ret != HCCL_SUCCESS) {
+            HCCL_ERROR("[SymmetricMemory] RegisterInternal Failed! Releasing offset 0x%zx", offset);
+            (void)vaAllocator_->Release(offset, alignedBaseSize);
+            return ret;
+        }
+
+        paMapInfo = std::make_shared<PaMappingInfo>();
+        paMapInfo->paHandle = paHandle;
+        paMapInfo->origAllocBaseVa = baseUserVa;
+        paMapInfo->origAllocSize = baseVaSize;
+        paMapInfo->heapBaseOffset = offset;
+        paMapInfo->refCount = 1;
+        paMappingMap_.emplace(paHandle, paMapInfo);
+    }
+    
+    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
+    pWin->userVa = baseUserVa;
+    pWin->userSize = baseVaSize;
+    pWin->baseVa = static_cast<uint8_t*>(heapBase_) + paMapInfo->heapBaseOffset;
+    pWin->alignedHeapOffset = paMapInfo->heapBaseOffset;
+    pWin->alignedSize = alignedBaseSize;
+    pWin->localRank = rank_;
+    pWin->rankSize = rankSize_;
+    pWin->stride = stride_;
+    pWin->paHandle = paHandle;
+
+    HcclResult ret = AddSymmetricWindow(pWin);
+    if (ret != HCCL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory] AddSymmetricWindow Failed!");
+        if (paMapInfo->refCount == 1) {
+            HCCL_ERROR("[SymmetricMemory] Releasing offset 0x%zx", paMapInfo->heapBaseOffset);
+            (void)vaAllocator_->Release(paMapInfo->heapBaseOffset, alignedBaseSize);
+            paMappingMap_.erase(paHandle);
+        } else {
+            paMapInfo->refCount--;
+        }
+        return ret;
+    }
+    *devWin = pWin->devWin;
+    return HCCL_SUCCESS;
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
+        break;
+    }
+
+    return ret;
+}
+
+HcclResult SymmetricMemory::FindSymmetricWindow(void* ptr, size_t size, void** win, u64 &offset)
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
+            offset = userVaStart - winStart;
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
+    int32_t localPid;
+    std::vector<aclrtMemFabricHandle> remoteShareableHandles(rankSize_);
+    std::vector<int32_t> remoteShareablePids(rankSize_);
+
+    if (aclrtDeviceGetBareTgid(&localPid) != ACL_SUCCESS) {
+        HCCL_ERROR("[SymmetricMemory][RegisterInternal] Failed to get pid");
+        return HCCL_E_DRV;
+    }
+    HCCL_INFO("[SymmetricMemory][RegisterInternal] Local pid: %d, offset: %zu, size: %zu", localPid, offset, mapSize);
+
+    if (ExchangePhyAddrHandle((void*)&localPid, (void*)remoteShareablePids.data(), 1,
+        HCCL_DATA_TYPE_INT32) != HCCL_SUCCESS) {
+        return HCCL_E_INTERNAL;
+    }
+    HCCL_INFO("[SymmetricMemory][RegisterInternal] remote pid: %d, %d", remoteShareablePids[0], remoteShareablePids[1]);
+
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
+    if (ExchangePhyAddrHandle((void*)&shareableHandle, (void*)remoteShareableHandles.data(), sizeof(aclrtMemFabricHandle),
+        HCCL_DATA_TYPE_UINT8) != HCCL_SUCCESS) {
+        return HCCL_E_INTERNAL;
+    }
+
+    u32 i = 0;
+    aclrtDrvMemHandle importedHandle;
+    for (; i < rankSize_; i++) {
+        void* targetVa = static_cast<uint8_t*>(heapBase_) + (stride_ * i) + offset;
+        if (i == rank_) {
+            importedHandle = paHandle;
+        } else if (aclrtMemImportFromShareableHandleV2((void*)&remoteShareableHandles[i], ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0,
+            &importedHandle) != ACL_SUCCESS) {
+            HCCL_ERROR("[SymmetricMemory][RegisterInternal] Failed to import handle from rank %u.", i);
+            goto MAP_ERROR;
+        }
+
+        if (aclrtMapMem(targetVa, mapSize, 0, importedHandle, 0) != ACL_SUCCESS) {
+            HCCL_ERROR("[SymmetricMemory][RegisterInternal] Failed to map mem for rank %u at va %p.", i, targetVa);
+            goto MAP_ERROR;
+        }
+        HCCL_INFO("[SymmetricMemory][RegisterInternal] success to Mapmem for rank %u at va %p to handle[%p].", i, targetVa, importedHandle);
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
+HcclResult SymmetricMemory::CreateSubComm()
+{
+    // aclrtStream stream;
+    HcclCommConfig config;
+    HcclCommConfigInit(&config);
+    stream_ = std::make_unique<Stream>(StreamType::STREAM_TYPE_ONLINE);
+
+    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
+    CHK_RET(HcclGetCommName(commHandle_, commName));
+    std::string identifier = std::string(commName) + "::symmetric_memory_sub_comm";
+
+    config.hcclBufferSize = 1;  // 1MByte
+    strcpy_s(config.hcclCommName, sizeof(config.hcclCommName), identifier.c_str());
+    std::vector<u32> rankIds(rankSize_);
+    for (u32 i = 0; i < rankSize_; i++) {
+        rankIds[i] = i;
+    }
+    CHK_RET(HcclCreateSubCommConfig(&commHandle_, rankSize_, rankIds.data(), 1, rank_, &config, &subCommHandle_));
+
+    return HCCL_SUCCESS;
+}
+
+HcclResult SymmetricMemory::ExchangePhyAddrHandle(void* inputBuff, void *outputBuff, u64 inputCount, HcclDataType dataType)
+{
+    u32 unitSize = SIZE_TABLE[dataType];
+    u64 inputSize = inputCount * unitSize;
+    u64 outputSize = inputSize * rankSize_;
+    void* sendBuff = nullptr;
+    void* recvBuff = nullptr;
+
+    CHK_RET(hrtMalloc((void**)&sendBuff, inputSize));
+    CHK_RET(hrtMalloc((void**)&recvBuff, outputSize));
+    CHK_RET(hrtMemSyncCopy((void*)sendBuff, inputSize, inputBuff, inputSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
+
+    CHK_RET(HcclAllGatherInner(sendBuff, recvBuff, inputCount, dataType, subCommHandle_, stream_->ptr()));
+    CHK_RET(hcclStreamSynchronize(stream_->ptr()));
+    CHK_RET(hrtMemSyncCopy((void*)outputBuff, outputSize, (void*)recvBuff, outputSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
+
+    HCCL_INFO("[SymmetricMemory][ExchangePhyAddrHandle] ExchangePhyAddrHandle success. inputCount: %llu, dataType size: %u, outputBuff: %p, outputSize: %llu", 
+        inputCount, unitSize, outputBuff, outputSize);
+
+    CHK_RET(hrtFree(sendBuff));
+    CHK_RET(hrtFree(recvBuff));
+
+    return HCCL_SUCCESS;
+}
+
+} // namespace hccl

```

### src/framework/communicator/impl/symmetric_memory/symmetric_memory.h
```diff
@@ -0,0 +1,116 @@
+/*
+ * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd. All Rights Reserved.
+ * This file is a part of the CANN Open Software.
+ * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
+ * Please refer to the License for details. You may not use this file except in compliance with the License.
+ * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
+ * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
+ * See LICENSE in the root of the software repository for the full text of the License.
+ */
+#ifndef SYMMETRIC_MEMORY_H_
+#define SYMMETRIC_MEMORY_H_
+
+#include "log.h"
+#include <vector>
+#include <unordered_map>
+#include <map>
+#include <mutex>
+#include <functional>
+#include <memory> // For std::unique_ptr
+
+// HCCL
+#include "hccl/base.h"
+#include "hccl_comm.h"
+#include "adapter_rts_common.h"
+#include "hccl_inner.h"
+#include "stream_pub.h"
+
+// NPU VMM API
+#include "acl/acl_rt.h"
+
+namespace hccl {
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
+
+    SymmetricMemory(HcclComm &comm, u32 rank, u32 rankSize, size_t stride, u32 devicePhyId);
+    ~SymmetricMemory();
+
+    // 禁止拷贝和赋值
+    SymmetricMemory(const SymmetricMemory&) = delete;
+    SymmetricMemory& operator=(const SymmetricMemory&) = delete;
+    HcclResult EnsureInit();
+    void* AllocSymmetricMem(size_t size);
+    HcclResult FreeSymmetricMem(void* devWin);
+    HcclResult RegisterSymmetricMem(void* ptr, size_t size, void** devWin);
+    HcclResult DeregisterSymmetricMem(void* devWin);
+    HcclResult FindSymmetricWindow(void* ptr, size_t size, void** win, u64 &offset);
+
+private:
+    HcclResult Init();
+    HcclResult RegisterInternal(aclrtDrvMemHandle &paHandle, size_t offset, size_t mapSize);
+    HcclResult AddSymmetricWindow(std::shared_ptr<SymmetricWindow> &win);
+    HcclResult DeleteSymmetricWindow(std::shared_ptr<SymmetricWindow> &win);
+    HcclResult DeleteSymmetricWindow(void* devWin);
+    HcclResult CreateSubComm();
+    HcclResult ExchangePhyAddrHandle(void* inputBuff, void* outputBuff, u64 inputCount, HcclDataType dataType);
+private:
+    // VA空间分配器 (Pimpl)
+    HcclComm commHandle_{nullptr};
+    HcclComm subCommHandle_{nullptr};
+    std::once_flag init_flag_;
+    u32 rank_{0};
+    u32 rankSize_{0};
+    u32 devicePhyId_{0};
+    size_t stride_{0};      // 每个Rank的VA空间大小
+    void* heapBase_{nullptr};  // 对称VA空间的总基地址 (所有rank相同)
+    size_t granularity_{0};
+    class SimpleVaAllocator;
+    std::unique_ptr<Stream> stream_{nullptr};
+    std::unique_ptr<SimpleVaAllocator> vaAllocator_;
+    HcclResult initResult_{HCCL_E_INTERNAL}; // 存储Init()的结果
+    std::vector<std::shared_ptr<SymmetricWindow>> sortedWindows_;
+    std::map<void*, std::shared_ptr<SymmetricWindow>> windowMap_; // device指针到host SymmetricWindow 的映射
+    std::unordered_map<void*, SymmetricWindow*> exactMatchMap_; // 用于提升查找性能
+    std::unordered_map<aclrtDrvMemHandle, std::shared_ptr<PaMappingInfo>> paMappingMap_;
+};
+
+} // namespace hccl
+
+#endif // SYMMETRIC_MEMORY_H_
\ No newline at end of file

```

### src/framework/device/framework/aicpu_communicator.cc
```diff
@@ -35,6 +35,7 @@
 #include "dlprof_function.h"
 #include "profiling_command_handle.h"
 #include "dispatcher_ctx.h"
+#include "aicpu_symmetric_memory.h"
 
 namespace hccl {
 constexpr u32 IPC_SIGNAL_MODULUS = 2;
@@ -320,8 +321,46 @@ HcclResult HcclCommAicpu::PrepareUserMemRanges(const OpParam &param, const AlgRe
     curUserOutputMemRange.baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
     curUserOutputMemRange.memSize = outputSize; // NOTE: 不应该使用param.outputSize (alltoall类始终为0)
 
-    // 针对zero copy, 设置remote rank的input/output usermem addr
-    if (param.isZeroCopy) {
+    //针对SymmetricMemory/zero copy, 设置remote rank的input/output usermem addr
+    if (param.supportSymmetricMemory) {
+        HCCL_INFO("[HcclCommAicpu][PrepareUserMemRanges] symmetric memory is enabled, prepare remote user memory ranges accordingly");
+        const std::unordered_set<LinkType> supportedLinkTypes = {LinkType::LINK_HCCS, LinkType::LINK_SIO, LinkType::LINK_HCCS_SW};
+        for (auto &singleSubCommTransport : algResource.opTransportResponse[COMM_LEVEL0]) {
+            for (u64 i = 0; i < singleSubCommTransport.links.size(); ++i) {
+                LINK link = singleSubCommTransport.links[i];
+                if (link == nullptr || !singleSubCommTransport.transportRequests[i].isValid || supportedLinkTypes.count(link->GetLinkType()) == 0) {
+                    // 无效或者不支持的链路
+                    continue;
+                }
+                // 对端在通信域内的rank id
+                const uint32_t remoteRank = link->GetRemoteRank();
+                CHK_PRT_RET(remoteRank >= rankSize, HCCL_ERROR("[HcclCommAicpu][PrepareUserMemRanges] remoteRank %u >= rankSize %u", remoteRank, rankSize), HCCL_E_INTERNAL);
+                HCCL_INFO("[HcclCommAicpu][PrepareUserMemRanges] prepare memory range of remote rank %u", remoteRank);
+
+                // 获取remote user input memory addr
+                void *remoteUserInputBaseAddr = nullptr;
+                CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remoteUserInputBaseAddr));
+                CHK_PTR_NULL(remoteUserInputBaseAddr);
+
+                // 更新remote user input memory range
+                OpUnfoldMemRange& remoteInputMemRange = userInputMemRanges[remoteRank];
+                remoteInputMemRange.isValid = true;
+                remoteInputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserInputBaseAddr);
+                remoteInputMemRange.memSize = inputSize;
+
+                // 获取remote user output memory addr
+                void *remoteUserOutputBaseAddr = nullptr;
+                CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &remoteUserOutputBaseAddr));
+                CHK_PTR_NULL(remoteUserOutputBaseAddr);
+
+                // 更新remote user output memory range
+                OpUnfoldMemRange& remoteOutputMemRange = userOutputMemRanges[remoteRank];
+                remoteOutputMemRange.isValid = true;
+                remoteOutputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserOutputBaseAddr);
+                remoteOutputMemRange.memSize = outputSize;
+            }
+        }
+    } else if (param.isZeroCopy) {
         // 注意: 只有非V类算子可能使用zero copy (因此假设remote ranks' input/output size与local rank相同)
         // 注意: 而V类算子一定是buffer copy (否则PrepareRemoteUserMemRanges需要额外的输入作为remote ranks' input/output size)
         const HcclCMDType opType = param.opType;
@@ -641,6 +680,12 @@ void HcclCommAicpu::SetZeroCopyEnable(bool enable)
     isZeroCopy_ = enable;
 }
 
+void HcclCommAicpu::SetSymmetricMemoryEnable(bool enable)
+{
+    HCCL_INFO("[HcclCommAicpu::SetSymmetricMemoryEnable] enable[%d]", enable);
+    isSymmetricMemory_ = enable;
+}
+
 HcclResult HcclCommAicpu::PrepareZeroCopyExchanger(const std::string &newTag, OpParam &opParam,
     AlgResourceResponse *algResResponse)
 {
@@ -2200,18 +2245,55 @@ u32 HcclCommAicpu::CalculateOpExecIndex(const OpParam &opParam, u32 userRank)
     return opIndex;
 }
 
+HcclResult HcclCommAicpu::PrepareSymmetricMemory(const OpParam &param, OpCommTransport &opTransportResponse)
+{
+    CHK_PRT_RET(opTransportResponse.size() == 0,
+        HCCL_ERROR("[HcclCommAicpu][PrepareSymmetricMemory] opTransportResponse size is 0"),
+        HCCL_E_PARA);
+    
+    const std::unordered_set<LinkType> supportedLinkTypes = {LinkType::LINK_HCCS, LinkType::LINK_SIO, LinkType::LINK_HCCS_SW};
+    for (auto &singleSubCommTransport : opTransportResponse[COMM_LEVEL0]) {
+        for (u64 i = 0; i < singleSubCommTransport.links.size(); ++i) {
+            LINK &link = singleSubCommTransport.links[i];
+            if (link == nullptr || !singleSubCommTransport.transportRequests[i].isValid || supportedLinkTypes.count(link->GetLinkType()) == 0) {
+                // 无效或者不支持的链路
+                continue;
+            }
+            u32 peerRank = link->GetRemoteRank();
+            void *remoteIn = HcclGetSymPtr(param.inputWindow, peerRank, param.inputOffset);
+            void *remoteOut = HcclGetSymPtr(param.outputWindow, peerRank, param.outputOffset);
+
+            CHK_PRT_RET(remoteIn == nullptr || remoteOut == nullptr,
+                HCCL_ERROR("[HcclCommAicpu][PrepareSymmetricMemory] remoteRank[%d] in[%p] out[%p] is invalid", peerRank, remoteIn, remoteOut),
+                HCCL_E_INTERNAL);
+            HCCL_INFO("[HcclCommAicpu][PrepareSymmetricMemory] remoteRank[%d] in[%p] out[%p]", peerRank, remoteIn, remoteOut);
+            CHK_RET(link->UpdateRemoteAddr(remoteIn, remoteOut));
+        }
+    }
+    return HCCL_SUCCESS;
+}
+
 HcclResult HcclCommAicpu::ExecOp(const std::string &newTag, const std::string &algName,
                                             OpParam &opParam, const HcclOpResParam *commParam)
 {
     std::unique_ptr<CollExecutorBase> executor;
     hccl::AlgResourceResponse *algResResponse;
     CHK_RET(GetAlgResponseRes(newTag, algName, opParam, commParam, executor, algResResponse));
-    if (isZeroCopy_) {
-        HcclResult ret = PrepareZeroCopyExchanger(newTag, opParam, algResResponse);
-        if(ret != HCCL_SUCCESS) {
-            HCCL_ERROR("[HcclCommAicpu][ExecOp] newTag[%s], localRankId[%u]",
-            newTag.c_str(), commParam->localUsrRankId);
-            return ret;
+
+    if (isZeroCopy_ || isSymmetricMemory_) {
+        // 对称内存场景使用内部虚拟地址，而非用户传入的地址
+        if (isSymmetricMemory_) {
+            HCCL_INFO("[HcclCommAicpu][ExecOp] opParam.inputPtr[%p], inputOffset[%llu], inputWindow[%p]", opParam.inputPtr, opParam.inputOffset, opParam.inputWindow);
+            HCCL_INFO("[HcclCommAicpu][ExecOp] opParam.outputPtr[%p], outputOffset[%llu], outputWindow[%p]", opParam.outputPtr, opParam.outputOffset, opParam.outputWindow);
+            opParam.inputPtr = HcclGetSymPtr(opParam.inputWindow, commParam->localUsrRankId, opParam.inputOffset);
+            opParam.outputPtr = HcclGetSymPtr(opParam.outputWindow, commParam->localUsrRankId, opParam.outputOffset);
+            CHK_PTR_NULL(opParam.inputPtr);
+            CHK_PTR_NULL(opParam.outputPtr);
+            CHK_RET(PrepareSymmetricMemory(opParam, algResResponse->opTransportResponse));
+        } else {
+            HcclResult ret = PrepareZeroCopyExchanger(newTag, opParam, algResResponse);
+            CHK_PRT_RET(ret != HCCL_SUCCESS, 
+                HCCL_ERROR("[HcclCommAicpu][ExecOp] newTag[%s], localRankId[%u]", newTag.c_str(), commParam->localUsrRankId), ret);            
         }
 
         // 零拷贝场景scratchMem的大小会与用户的输入大小不同，会导致后续算法展开模块计算出错

```

### src/framework/device/framework/aicpu_communicator.h
```diff
@@ -152,6 +152,7 @@ public:
     HcclResult PrintTaskExceptionAllThreads();
     bool GetOpRetryEnable();
     void SetZeroCopyEnable(bool enable);
+    void SetSymmetricMemoryEnable(bool enable);
     bool IsTaskExceptionForHccs();
     static u32 HcclGetWaitStopExecCmdTimeout();
     u32 HcclGetWaitRetryCmdTimeout(uint32_t retryCnt);
@@ -406,6 +407,9 @@ private:
     HcclResult IsInplace(const OpParam &param, bool& isInplace);
     HcclResult ParseOpParamForCache(const OpParam &param, HcclDataType& sendType, HcclDataType& recvType, uint64_t& inputSize, uint64_t& outputSize);
 
+    //对称内存
+    HcclResult PrepareSymmetricMemory(const OpParam &param, OpCommTransport &opTransportResponse);
+
     std::unordered_map<s32, u32> opExecIndexMap_;
 
     // 管理aicpu和custom进程共享的数据
@@ -529,6 +533,7 @@ private:
     std::map<u32, u32> bsrRecvIndexMap_;
 
     bool isZeroCopy_{false};
+    bool isSymmetricMemory_{false};
     hccl::AlgOpContext algOpContext_;
     std::unique_ptr<HcclTraceInfo> UtraceInfo_;
     // taskException

```

### src/framework/device/framework/aicpu_hccl_process.cc
```diff
@@ -402,6 +402,10 @@ HcclResult AicpuHcclProcess::AicpuRunRpcServerV2(
     opParam.reduceType = static_cast<HcclReduceOp>(tilingData->reduceType);
     opParam.stream = hcclCommAicpu->GetMainStream();
     opParam.syncMode = static_cast<SyncMode>(tilingData->syncMode);
+    opParam.inputWindow = reinterpret_cast<void *>(tilingData->inputWindow);
+    opParam.inputOffset = tilingData->inputOffset;
+    opParam.outputWindow = reinterpret_cast<void *>(tilingData->outputWindow);
+    opParam.outputOffset = tilingData->outputOffset;
 
     hcclCommAicpu->UpdateNotifyWaitTimeOut(opParam.syncMode, commParam->config.notifyWaitTime);
 
@@ -411,6 +415,7 @@ HcclResult AicpuHcclProcess::AicpuRunRpcServerV2(
     opParam.srcRank = tilingData->srcRank;
     opParam.opType = static_cast<HcclCMDType>(tilingData->opType);
     opParam.isZeroCopy = tilingData->isZeroCopy;
+    opParam.supportSymmetricMemory = tilingData->isSymmetricMemory;
     opParam.index = tilingData->index;
     opParam.isCapture = tilingData->isCapture;
     opParam.aicpuCacheEnable = tilingData->aicpuCacheEnable;

```

### src/framework/device/framework/aicpu_symmetric_memory.cc
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
+#include "aicpu_symmetric_memory.h"
+
+namespace hccl {
+
+class SymmetricMemory::SimpleVaAllocator {
+    public:
+        // 不需要任何成员，只要让编译器觉得它是个完整的类就行
+        SimpleVaAllocator() {} 
+        ~SimpleVaAllocator() {}
+};
+
+SymmetricMemory::~SymmetricMemory() {}
+
+void *HcclGetSymPtr(HcclWindow winHandle, int32_t peerRank, size_t offset)
+{
+    SymmetricWindow *symWin = reinterpret_cast<SymmetricWindow *>(winHandle);
+    size_t peerOffset = peerRank * symWin->stride + offset;
+    void *peerVa = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(symWin->baseVa) + peerOffset);
+    HCCL_INFO("[HcclGetSymPtr] Get Ptr[%p] from winHandle[%p], rank[%d], peerOffset[%llu]", peerVa, winHandle, peerRank, peerOffset);
+    return peerVa;
+}
+
+}
\ No newline at end of file

```

### src/framework/device/framework/aicpu_symmetric_memory.h
```diff
@@ -0,0 +1,19 @@
+/**
+ * Copyright (c) 2025 Huawei Technologies Co., Ltd.
+ * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
+ * CANN Open Software License Agreement Version 2.0 (the "License").
+ * Please refer to the License for details. You may not use this file except in compliance with the License.
+ * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
+ * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
+ * See LICENSE in the root of the software repository for the full text of the License.
+ */
+#ifndef AICPU_SYMMETRIC_MEMORY_H_
+#define AICPU_SYMMETRIC_MEMORY_H_
+
+#include "symmetric_memory/symmetric_memory.h"
+
+namespace hccl {
+    void *HcclGetSymPtr(HcclWindow winHandle, int32_t peerRank, size_t offset);
+}
+
+#endif // AICPU_SYMMETRIC_MEMORY_H_
\ No newline at end of file

```

### src/framework/device/hccl_aicpu_interface.cc
```diff
@@ -71,8 +71,10 @@ __attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunchV2(void *arg
         HCCL_ERROR("RunAicpuRpcSrvLaunchV2 get Hcclcomm error group[%s], tag[%s]", commParam->hcomId, tilingData->tag);
         return HCCL_E_INTERNAL;
     }
-    HCCL_INFO("[RunAicpuRpcSrvLaunchV2] isZeroCopy [%d], workflowMode[%d]", tilingData->isZeroCopy, tilingData->workflowMode);
+    HCCL_INFO("[RunAicpuRpcSrvLaunchV2] isZeroCopy [%d], isSymmetricMemory [%d], workflowMode[%d]",
+        tilingData->isZeroCopy, tilingData->isSymmetricMemory, tilingData->workflowMode);
     hcclCommAicpu->SetZeroCopyEnable(tilingData->isZeroCopy);
+    hcclCommAicpu->SetSymmetricMemoryEnable(tilingData->isSymmetricMemory);
     DfxExtendInfo* dfxInfo = hcclCommAicpu->GetDfxExtendInfo();
     if ((dfxInfo->cqeStatus != dfx::CqeStatus::kDefault) ||
         (dfxInfo->pollStatus == PollStatus::kStopAsException)) {

```

### src/framework/inc/hccl_comm_pub.h
```diff
@@ -390,6 +390,9 @@ public:
     HcclResult GroupPrepareStreamAndNotify(HcclRtStream sendRecvMainStream);
     HcclResult GroupSyncMainstream(std::unordered_map<u32, std::vector<u64>> &sendIdx2Byte, std::unordered_map<u32, std::vector<u64>> &recvIdx2Byte);
     HcclResult GroupSubstreamsSync();
+    HcclResult RegisterWindow(void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags);
+    HcclResult DeregisterWindow(HcclWindow winHandle);
+
 protected:
     /* * 禁止用户对API类的实体做拷贝构造或拷贝赋值的操作，内部有指针成员变量 */
     hcclComm(const hcclComm &) = delete;

```

### src/framework/op_base/src/op_base.cc
```diff
@@ -1360,6 +1360,7 @@ HcclResult InitCommRootInfo(const u32 nRanks, const u32 rank, const HcclRootHand
             HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init other Info", HCCL_ERROR_CODE(ret)),
             errorFlag = true);
 
+        params.commHandle = pComm.get();
         HCCL_INFO("rootInfo[%s], params.logiceDevice[%d]", params.id.internal, params.logicDevId);
         ret = pComm->init(params, commConfig, rankTable);
         CHK_PRT_BREAK(ret != HCCL_SUCCESS,
@@ -2953,17 +2954,29 @@ HcclResult HcclCommDestroy(HcclComm comm)
     }
 
     HcomUnSetGroupTopoInfo(group.c_str());
+    HCCL_INFO("[HcclCommDestroy][HcomUnSetGroupTopoInfo] end, group[%s]", group.c_str());
 
     std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
+    HCCL_INFO("[HcclCommDestroy][opBaseHcom.opGroupMapMutex] begin, group[%s]", group.c_str());
     auto iter = opBaseHcom.opGroup2CommMap.find(group);
     if (iter != opBaseHcom.opGroup2CommMap.end()) {
+        string subGroup = group + "::symmetric_memory_sub_comm";
+        auto subIter = opBaseHcom.opGroup2CommMap.find(subGroup);
+        if (subIter != opBaseHcom.opGroup2CommMap.end()) {
+            EXECEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(subGroup), return HCCL_E_MEMORY);
+            HCCL_INFO("[HcclCommDestroy][opBaseHcom.opGroup2CommMap.erase(group)] end, group[%s]", subGroup.c_str());
+            HcclCloseCommConnections(subGroup);
+        }
         EXECEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
+        HCCL_INFO("[HcclCommDestroy][opBaseHcom.opGroup2CommMap.erase(group)] end, group[%s]", group.c_str());
         HcclCloseCommConnections(group);
     } else {
         HCCL_ERROR("[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(), deviceLogicId);
         return HCCL_E_PARA;
     }
 
+    HCCL_INFO("[HcclCommDestroy][HcclCloseCommConnections] end, group[%s]", group.c_str());
+
     if (ProfilingManagerPub::GetAllState()) {
         ProfilingManagerPub::ClearStoragedProfilingInfo();
     }
@@ -4455,6 +4468,34 @@ HcclResult CommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufSize)
     HCCL_RUN_INFO("CommGetCCLBufSizeCfg success, comm[%s], size[%u]", hcclComm->GetIdentifier().c_str(), buffSize);
     return HCCL_SUCCESS;
 }
+
+HcclResult HcclCommWindowRegister(HcclComm comm, void* ptr, size_t size, HcclWindow *winHandle, uint64_t flags)
+{
+    // 入参校验
+    CHK_PTR_NULL(comm);
+    CHK_PTR_NULL(ptr);
+    CHK_PTR_NULL(winHandle);
+    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] size is 0, please check size value", __func__), HCCL_E_PARA);
+
+    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
+    CHK_RET(hcclComm->RegisterWindow(ptr, size, winHandle, flags));
+    HCCL_RUN_INFO("[%s]WindowRegister mem success, group[%s], handle ptr[%p], size[%llu]", __func__,
+        hcclComm->GetIdentifier().c_str(), *winHandle, size);
+    return HCCL_SUCCESS;
+}
+
+HcclResult HcclCommWindowDeRegister(HcclComm comm, HcclWindow winHandle)
+{
+    // 入参校验
+    CHK_PTR_NULL(comm);
+    CHK_PTR_NULL(winHandle);
+    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
+    CHK_RET(hcclComm->DeregisterWindow(winHandle));
+    HCCL_RUN_INFO("[%s]WindowDeregister mem success, group[%s], handle ptr[%p]", __func__,
+        hcclComm->GetIdentifier().c_str(), winHandle);
+    return HCCL_SUCCESS;
+}
+
 #ifdef __cplusplus
 }
 #endif // __cplusplus

```

### src/platform/common/externalinput.cc
```diff
@@ -514,6 +514,20 @@ HcclResult InitEnvVarParam()
             ret),
         ret);
 
+    ret = ParseSymmetricMemoryStride();
+    RPT_ENV_ERR(ret != HCCL_SUCCESS,
+        "EI0001",
+        std::vector<std::string>({"env", "tips"}),
+        std::vector<std::string>({"HCCL_SYMMETRIC_MEMORY_STRIDE", "Value should be a power of 2."}));
+    CHK_PRT_RET(ret != HCCL_SUCCESS,
+        HCCL_ERROR("[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_SYMMETRIC_MEMORY_STRIDE failed. "
+                   "errorno[%d]",
+            LOG_KEYWORDS_INIT_GROUP.c_str(),
+            LOG_KEYWORDS_ENV_CONFIG.c_str(),
+            HCCL_ERROR_CODE(ret),
+            ret),
+        ret);
+
     return HCCL_SUCCESS;
 }
 
@@ -1307,6 +1321,29 @@ HcclResult ParseRDMARetryCnt()
     return HCCL_SUCCESS;
 }
 
+HcclResult ParseSymmetricMemoryStride()
+{
+    // std::string symmetricMemoryStrideEnv = GET_ENV(MM_ENV_HCCL_SYMMETRIC_MEM_STRIDE);
+    // if (symmetricMemoryStrideEnv == "EmptyString") {
+    //     g_externalInput.symmetricMemoryStride = SYMMETRIC_MEM_STRIDE_DEFAULT;
+    //     HCCL_RUN_INFO("HCCL_SYMMETRIC_MEM_STRIDE set by default to [%u]GB", g_externalInput.symmetricMemoryStride);
+    //     return HCCL_SUCCESS;
+    // }
+
+    // // 校验环境变量长度
+    // bool isEnvLenValid = CheckEnvLen(symmetricMemoryStrideEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);
+    // CHK_PRT_RET(!isEnvLenValid,
+    //     HCCL_ERROR("[Parse][SymmetricMemoryStride]errNo[0x%016llx] Invalid SymmetricMemoryStride env len,"
+    //     "len is bigger than [%u]. errorno[%d]",
+    //     HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
+    //     HCCL_E_PARA);
+
+    g_externalInput.symmetricMemoryStride = SYMMETRIC_MEM_STRIDE_DEFAULT;
+    // CHK_RET(IsAllDigit(symmetricMemoryStrideEnv.c_str()));
+    // HcclResult ret = SalStrToULong(symmetricMemoryStrideEnv, HCCL_BASE_DECIMAL, symmetricMemoryStride);
+    return HCCL_SUCCESS;
+}
+
 HcclResult ParseCannVersion()
 {
 #if !defined(CCL_KERNEL_AICPU) && !defined(HCCD)
@@ -2078,3 +2115,8 @@ void SetExternalInputDebugConfig(u64 value)
 {
     g_externalInput.debugConfig = value;
 }
+
+const u64& GetExternalInputSymmetricMemoryStride()
+{
+    return g_externalInput.symmetricMemoryStride;
+}
\ No newline at end of file

```

### src/platform/common/externalinput.h
```diff
@@ -60,6 +60,8 @@ constexpr u32 HCCL_RETRY_HLOD_TIME_MAX = 60000; // 首次重执行等待时间
 constexpr u32 HCCL_RETRY_INTERVAL_MAX = 60000;  // 重执行间隔，最大值60s, 单位ms
 
 constexpr u32 MAX_LEN_OF_LOGIC_SUPER_ID = 128; // 逻辑超节点最大长度
+constexpr u64 SYMMETRIC_MEM_STRIDE_DEFAULT = 16; // 默认对称内存的stride大小，GB
+
 //  外部输入参数
 struct ExternalInput {
     //  初始化判断
@@ -80,6 +82,7 @@ struct ExternalInput {
     u32 retryMaxCnt;          // 重执行最大尝试次数，配置范围[0,60000]，默认值为3
     u32 retryIntervalTime;    // 重执行间隔时间，配置范围[0,3600000]，默认值1000
     u32 retryHoldTime;        // 首次重执行等待时间，配置范围[0,3600000]，默认值1000
+    u64 symmetricMemoryStride; // 对称内存stride大小
 
     // cann版本信息
     std::string cannVersion;
@@ -219,6 +222,8 @@ HcclResult ParseRDMATimeOut(std::pair<u32, u32> &rdmaTimeOutRange);
 
 HcclResult ParseRDMARetryCnt();
 
+HcclResult ParseSymmetricMemoryStride();
+
 HcclResult ParseRdmaQpsPerConnection();
 
 HcclResult ParseMultiQpThreshold();
@@ -284,4 +289,5 @@ const s32& GetIncreSaveExecTimeOut();
 const u64& GetProfConfig();
 
 const u64& GetExternalInputDebugConfig();
+const u64& GetExternalInputSymmetricMemoryStride();
 #endif  //  EXTERNALINPUT_H

```

### src/pub_inc/aicpu_operator_pub.h
```diff
@@ -705,6 +705,11 @@ struct OpTilingData {
     s32 userStreamId;
     u32 ahcConfInfo[TOP_HIERARCHICAL_CONF_SIZE] = {0};
     uint8_t aicpuCacheEnable = 0; // 是否开启aicpu cache
+    u8 isSymmetricMemory = 0;
+    u64 inputWindow;
+    u64 inputOffset = 0;
+    u64 outputWindow;
+    u64 outputOffset = 0;
 
     /******************可变长度数据区，如需新增字段请在这之前增加*******************/
     u64 length;   // 可变长度数据区长度

```

### test/llt/aicpu_kfc/stub/llt_aicpu_kfc_stub.cc
```diff
@@ -2119,6 +2119,32 @@ aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
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
@@ -1947,6 +1947,32 @@ aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
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
(diff 过长，已截断)

> 注: 以下非 C/C++ 文件未纳入审查: src/framework/CMakeLists.txt, src/framework/common/src/CMakeLists.txt, src/framework/communicator/impl/CMakeLists.txt, src/framework/communicator/impl/symmetric_memory/CMakeLists.txt, src/framework/device/framework/CMakeLists.txt, test/llt/ut/single_test/impl/CMakeLists.txt
