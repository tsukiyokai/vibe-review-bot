# PR #1144: fix bug of ccl addr update in graph mode

- 作者: gcw_TwqkoH55
- 分支: graphccl_c25 -> r1.25.0
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/1144
- 描述: fix bug of ccl addr update in graph mode

## 变更文件 (2 个, 其中 C/C++ 文件 2 个)

- [modified] src/framework/communicator/impl/hccl_communicator_host.cc (+8449, -8427) *
- [modified] src/framework/device/framework/aicpu_communicator.cc (+25, -5) *

## Diff 内容

### src/framework/communicator/impl/hccl_communicator_host.cc
```diff
The content of this file is too large to show differences: max size 204800 bytes
```

### src/framework/device/framework/aicpu_communicator.cc
```diff
@@ -321,6 +321,7 @@ HcclResult HcclCommAicpu::PrepareUserMemRanges(const OpParam &param, const AlgRe
     curUserOutputMemRange.memSize = outputSize; // NOTE: 不应该使用param.outputSize (alltoall类始终为0)
 
     // 针对zero copy, 设置remote rank的input/output usermem addr
+    constexpr uint8_t FORCE_OP_BASE_DELTA = 10;
     if (param.isZeroCopy) {
         // 注意: 只有非V类算子可能使用zero copy (因此假设remote ranks' input/output size与local rank相同)
         // 注意: 而V类算子一定是buffer copy (否则PrepareRemoteUserMemRanges需要额外的输入作为remote ranks' input/output size)
@@ -331,7 +332,8 @@ HcclResult HcclCommAicpu::PrepareUserMemRanges(const OpParam &param, const AlgRe
         // 直接传入local rank's input/output size用于remote ranks' memory ranges
         HCCL_INFO("[HcclCommAicpu][PrepareUserMemRanges] prepare user memory ranges of other remote ranks");
         CHK_RET(ZeroCopyExchanger_->PrepareRemoteUserMemRanges(inputSize, outputSize, userInputMemRanges, userOutputMemRanges));
-    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
+    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB ||
+        param.aicpuCacheEnable > FORCE_OP_BASE_DELTA) { // 图模式 或者 存在强制单算子模式转换 (图模式建链 + 单算子模式展开)
         HCCL_INFO("[HcclCommAicpu][PrepareUserMemRanges] check transport resource for potential user memory of remote ranks");
 
         // 遍历所有transport信息, 更新remote ranks' user input/output memory ranges
@@ -365,7 +367,8 @@ HcclResult HcclCommAicpu::PrepareUserMemRanges(const OpParam &param, const AlgRe
                         CHK_PTR_NULL(curLink);
 
                         // 获取user input memory range if any
-                        if (curReq.inputMemType == TransportMemType::PARAM_INPUT) {
+                        if (curReq.inputMemType == TransportMemType::PARAM_INPUT ||
+                            curReq.inputMemType == TransportMemType::CCL_INPUT) {
                             // 获取remoteRank的user input memory baseaddr
                             void *remoteUserInputBaseAddr = nullptr;
                             CHK_RET(curLink->GetRemoteMem(UserMemType::INPUT_MEM, &remoteUserInputBaseAddr));
@@ -377,11 +380,20 @@ HcclResult HcclCommAicpu::PrepareUserMemRanges(const OpParam &param, const AlgRe
                             OpUnfoldMemRange& remoteUserInputMemRange = userInputMemRanges[curReq.remoteUserRank];
                             remoteUserInputMemRange.isValid = true;
                             remoteUserInputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserInputBaseAddr);
-                            remoteUserInputMemRange.memSize = inputSize;
+                            if (curReq.inputMemType == TransportMemType::PARAM_INPUT) { // user input
+                                remoteUserInputMemRange.memSize = inputSize;
+                            } else if (curReq.inputMemType == TransportMemType::CCL_INPUT) { // hccl input
+                                remoteUserInputMemRange.memSize = algResource.cclInputMem.size();
+                            } else {
+                                HCCL_ERROR("[HcclCommAicpu][PrepareUserMemRanges] invalid curReq.inputMemType[%u]",
+                                    curReq.inputMemType);
+                                return HCCL_E_INTERNAL;
+                            }
                         }
 
                         // 获取user output memory range if any
-                        if (curReq.outputMemType == TransportMemType::PARAM_OUTPUT) {
+                        if (curReq.outputMemType == TransportMemType::PARAM_OUTPUT ||
+                            curReq.outputMemType == TransportMemType::CCL_OUTPUT) {
                             // 获取remoteRank的user output memory baseaddr
                             void *remoteUserOutputBaseAddr = nullptr;
                             CHK_RET(curLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &remoteUserOutputBaseAddr));
@@ -393,7 +405,15 @@ HcclResult HcclCommAicpu::PrepareUserMemRanges(const OpParam &param, const AlgRe
                             OpUnfoldMemRange& remoteUserOutputMemRange = userOutputMemRanges[curReq.remoteUserRank];
                             remoteUserOutputMemRange.isValid = true;
                             remoteUserOutputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserOutputBaseAddr);
-                            remoteUserOutputMemRange.memSize = outputSize;
+                            if (curReq.outputMemType == TransportMemType::PARAM_OUTPUT) { // user output
+                                remoteUserOutputMemRange.memSize = outputSize;
+                            } else if (curReq.outputMemType == TransportMemType::CCL_OUTPUT) { // hccl output
+                                remoteUserOutputMemRange.memSize = algResource.cclOutputMem.size();
+                            } else {
+                                HCCL_ERROR("[HcclCommAicpu][PrepareUserMemRanges] invalid curReq.outputMemType[%u]",
+                                    curReq.outputMemType);
+                                return HCCL_E_INTERNAL;
+                            }
                         }
                     } // curReq.isValid
                 } // Each TransportRequest

```
