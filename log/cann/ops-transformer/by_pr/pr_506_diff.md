# PR #506: Support Symmetric Memory

- 作者: linzhenkang
- 分支: temp1 -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/506
- 描述: Support Symmetric Memory

## 变更文件 (9 个, 其中 C/C++ 文件 9 个)

- [modified] src/algorithm/pub_inc/coll_alg_param.h (+5, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator.cc (+4, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator.h (+1, -0) *
- [modified] src/framework/communicator/impl/hccl_communicator_host.cc (+36, -1) *
- [modified] src/framework/device/framework/aicpu_communicator.cc (+44, -6) *
- [modified] src/framework/device/framework/aicpu_communicator.h (+2, -0) *
- [modified] src/framework/device/framework/aicpu_hccl_process.cc (+4, -0) *
- [modified] src/framework/device/hccl_aicpu_interface.cc (+3, -1) *
- [modified] src/pub_inc/aicpu_operator_pub.h (+5, -0) *

## Diff 内容

### src/algorithm/pub_inc/coll_alg_param.h
```diff
@@ -205,6 +205,11 @@ struct OpParam {
     u8 deterministic = 0;
     u32 srTag = 0;
     u32 localGroupRank = 0;
+    bool supportSymmetricMemory = false;
+    void* inputWindow = nullptr;
+    u64 inputOffset = 0;
+    void* outputWindow = nullptr;
+    u64 outputOffset = 0;
 
     inline HcclDataType GetDataType() const
     {

```

### src/framework/communicator/impl/hccl_communicator.cc
```diff
@@ -1686,6 +1686,10 @@ namespace hccl
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
@@ -743,6 +743,7 @@ private:
     bool GetSupportHDCommunicate();
     HcclResult InitOpRetry();
     HcclResult InitOpResPara();
+    bool IsSupportSymmetricMemory(OpParam &opParam);
     bool IsSupportZeroCopy(const OpParam &opParam);
     HcclResult PrepareZeroCopy(const std::string &algName, const AlgDesc &algDesc, OpParam &opParam);
     HcclResult UpdateZeroCopy(const OpParam &opParam, const AlgResourceResponse &algResource);

```

### src/framework/communicator/impl/hccl_communicator_host.cc
```diff
@@ -540,6 +540,39 @@ namespace hccl
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
+        HcclResult ret = SymmetricMemory::FindSymmetricWindow(opParam.inputPtr, opParam.inputSize, opParam.inputWindow, opParam.inputOffset);
+        CHK_PRT_RET(ret != HCCL_SUCCESS || opParam.inputWindow == nullptr,
+                    HCCL_INFO("[%s] input[%p] size[%llu] is not support symmetric memory", __func__, opParam.inputPtr, opParam.inputSize), false);
+        HcclResult ret = SymmetricMemory::FindSymmetricWindow(opParam.outputPtr, opParam.outputSize, opParam.outputWindow, opParam.outOffset);
+        CHK_PRT_RET(ret != HCCL_SUCCESS || opParam.outputWindow == nullptr,
+                    HCCL_INFO("[%s] output[%p] size[%llu] is not support symmetric memory", __func__, opParam.outputPtr, opParam.outputSize), false);
+
+        return true;
+    }
+
     bool HcclCommunicator::IsSupportZeroCopy(const OpParam &opParam)
     {
         HCCL_INFO("[%s] aicpuUnfold[%d], workflowMode[%d], deviceType[%d], "
@@ -3953,7 +3986,8 @@ namespace hccl
         }
 
         ForceProf(opParam.isCapture);
-        opParam.supportZeroCopy = !commConfig_.GetConfigDeterministic() && IsSupportZeroCopy(opParam);
+        opParam.supportSymmetricMemory = IsSupportSymmetricMemory(opParam);
+        opParam.supportZeroCopy = opParam.supportSymmetricMemory || (!commConfig_.GetConfigDeterministic() && IsSupportZeroCopy(opParam));
         opParam.aclGraphZeroCopyEnable = GetConfigAclGraphZeroCopyEnable();
         bool isInGraphCaptureZeroCopy = false;
         zeroCopyAclGraph_->SetRetryEnable(retryEnable_);
@@ -6564,6 +6598,7 @@ namespace hccl
         opTilingData->isZeroCopy = opParam.isZeroCopy;
         opTilingData->isCapture = opParam.isCapture;
         opTilingData->orderLaunchMode = GetOrderLaunchMode(opParam.isCapture);
+        opTilingData->isSymmetricMemory = opParam.supportSymmetricMemory;
         // 有没有存在对应的Notify
         CHK_RET(InitAndCheckAicpuOrderNotify(opTilingData->orderLaunchMode));
         CHK_RET(BuildHierarchicalAlgOption(opTilingData->ahcConfInfo));

```

### src/framework/device/framework/aicpu_communicator.cc
```diff
@@ -322,7 +322,35 @@ HcclResult HcclCommAicpu::PrepareUserMemRanges(const OpParam &param, const AlgRe
 
         // 直接传入local rank's input/output size用于remote ranks' memory ranges
         HCCL_INFO("[HcclCommAicpu][PrepareUserMemRanges] prepare user memory ranges of other remote ranks");
-        CHK_RET(ZeroCopyExchanger_->PrepareRemoteUserMemRanges(inputSize, outputSize, userInputMemRanges, userOutputMemRanges));
+        if(!isSymmetricMemory_) {
+            CHK_RET(ZeroCopyExchanger_->PrepareRemoteUserMemRanges(inputSize, outputSize, userInputMemRanges, userOutputMemRanges));
+        }else {
+            for(size_t peerRank = 0; peerRank < rankSize; ++peerRank) {
+                if(peerRank != curRank) {
+                    // 获取remote user input memory addr
+                    void *remoteUserInputBaseAddr = nullptr;
+                    remoteUserInputBaseAddr = HcclGetSymPtr(opParam.inputWindow, peerRank, opParam.inputOffset);
+                    CHK_PTR_NULL(remoteUserInputBaseAddr);
+
+                    // 更新remote user input memory range
+                    OpUnfoldMemRange& remoteInputMemRange = userInputMemRanges[peerRank];
+                    remoteInputMemRange.isValid = true;
+                    remoteInputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserInputBaseAddr);
+                    remoteInputMemRange.memSize = inputSize;
+
+                    // 获取remote user output memory addr
+                    void *remoteUserOutputBaseAddr = nullptr;
+                    remoteUserOutputBaseAddr = HcclGetSymPtr(opParam.outputWindow, peerRank, opParam.outputOffset);
+                    CHK_PTR_NULL(remoteUserOutputBaseAddr);
+
+                    // 更新remote user output memory range
+                    OpUnfoldMemRange& remoteOutputMemRange = userOutputMemRanges[peerRank];
+                    remoteOutputMemRange.isValid = true;
+                    remoteOutputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserOutputBaseAddr);
+                    remoteOutputMemRange.memSize = outputSize;
+                }
+            }
+        }
     } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
         HCCL_INFO("[HcclCommAicpu][PrepareUserMemRanges] check transport resource for potential user memory of remote ranks");
 
@@ -633,6 +661,11 @@ void HcclCommAicpu::SetZeroCopyEnable(bool enable)
     isZeroCopy_ = enable;
 }
 
+void HcclCommAicpu::SetSymmetricMemoryEnable(bool enable)
+{
+    isSymmetricMemory_ = enable;
+}
+
 HcclResult HcclCommAicpu::PrepareZeroCopyExchanger(const std::string &newTag, OpParam &opParam,
     AlgResourceResponse *algResResponse)
 {
@@ -2192,11 +2225,16 @@ HcclResult HcclCommAicpu::ExecOp(const std::string &newTag, const std::string &a
     hccl::AlgResourceResponse *algResResponse;
     CHK_RET(GetAlgResponseRes(newTag, algName, opParam, commParam, executor, algResResponse));
     if (isZeroCopy_) {
-        HcclResult ret = PrepareZeroCopyExchanger(newTag, opParam, algResResponse);
-        if(ret != HCCL_SUCCESS) {
-            HCCL_ERROR("[HcclCommAicpu][ExecOp] newTag[%s], localRankId[%u]",
-            newTag.c_str(), commParam->localUsrRankId);
-            return ret;
+        // 对称内存场景使用内部虚拟地址，而非用户传入的地址
+        if (isSymmetricMemory_) {
+            opParam.inputPtr = HcclGetSymPtr(opParam.inputWindow, commParam->localUsrRankId, opParam.inputOffset);
+            opParam.outputPtr = HcclGetSymPtr(opParam.outputWindow, commParam->localUsrRankId, opParam.outputOffset);
+            CHK_PTR_NULL(opParam.inputPtr);
+            CHK_PTR_NULL(opParam.outputPtr);
+        }else {
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
@@ -529,6 +530,7 @@ private:
     std::map<u32, u32> bsrRecvIndexMap_;
 
     bool isZeroCopy_{false};
+    bool isSymmetricMemory_{false};
     hccl::AlgOpContext algOpContext_;
     std::unique_ptr<HcclTraceInfo> UtraceInfo_;
     // taskException

```

### src/framework/device/framework/aicpu_hccl_process.cc
```diff
@@ -383,6 +383,10 @@ HcclResult AicpuHcclProcess::AicpuRunRpcServerV2(
     opParam.reduceType = static_cast<HcclReduceOp>(tilingData->reduceType);
     opParam.stream = hcclCommAicpu->GetMainStream();
     opParam.syncMode = static_cast<SyncMode>(tilingData->syncMode);
+    opParam.inputWindow = reinterpret_cast<void *>(tilingData->inputWindow);
+    opParam.inputOffset = tilingData->inputOffset;
+    opParam.outputWindow = reinterpret_cast<void *>(tilingData->outputWindow);
+    opParam.outputOffset = tilingData->outputOffset;
 
     hcclCommAicpu->UpdateNotifyWaitTimeOut(opParam.syncMode, commParam->config.notifyWaitTime);
 

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

### src/pub_inc/aicpu_operator_pub.h
```diff
@@ -701,6 +701,11 @@ struct OpTilingData {
     u64 version = 0;
     s32 userStreamId;
     u32 ahcConfInfo[TOP_HIERARCHICAL_CONF_SIZE] = {0};
+    u8 isSymmetricMemory = 0;
+    void* inputWindow = nullptr;
+    u64 inputOffset = 0;
+    void* outputWindow = nullptr;
+    u64 outputOffset = 0;
 
     /******************可变长度数据区，如需新增字段请在这之前增加*******************/
     u64 length;   // 可变长度数据区长度

```
