# PR #1143: dfx plus

- 作者: linyixin4
- 分支: master -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/1143
- 描述: dfx plus

## 变更文件 (15 个, 其中 C/C++ 文件 15 个)

- [modified] src/legacy/framework/communicator/aicpu/communicator_impl_lite.cc (+8, -1) *
- [modified] src/legacy/framework/communicator/aicpu/communicator_impl_lite.h (+1, -0) *
- [modified] src/legacy/framework/communicator/aicpu/kernel_entrance.cc (+2, -2) *
- [modified] src/legacy/framework/communicator/aicpu/kernel_param_lite.h (+1, -0) *
- [modified] src/legacy/framework/communicator/communicator_impl.cc (+14, -1) *
- [modified] src/legacy/framework/communicator/communicator_impl.h (+3, -0) *
- [modified] src/legacy/framework/communicator/hccl_communicator.cc (+5, -0) *
- [modified] src/legacy/framework/dfx/common/task_info.cc (+6, -3) *
- [modified] src/legacy/framework/dfx/common/task_info.h (+7, -3) *
- [modified] src/legacy/framework/entrance/hcom/hcom_v2.cc (+17, -0) *
- [modified] src/legacy/framework/entrance/op_base/op_base_v2.cc (+16, -16) *
- [modified] src/legacy/framework/service/coll_service_base.cc (+6, -1) *
- [modified] src/legacy/include/hccl_communicator.h (+1, -0) *
- [modified] test/legacy/ut/framework/dfx/aicpu/profiling/ut_profiling_handler_lite.cc (+15, -3) *
- [modified] test/legacy/ut/framework/dfx/common/ut_task_info.cc (+6, -3) *

## Diff 内容

### src/legacy/framework/communicator/aicpu/communicator_impl_lite.cc
```diff
@@ -31,6 +31,7 @@ int CommunicatorImplLite::LoadWithOpBasedMode(HcclKernelParamLite *kernelParam)
         // 设定devType，初始化能力，算法及其他模块通过Get获取能力
         DevCapability::GetInstance().Init(kernelParam->comm.devType);
         UnfoldOp(kernelParam);
+        opIndex++;//算子计数
     } catch (HcclException &e) {
         HCCL_ERROR("Hccl exception %s was caught.", e.what());
         return KERNEL_ERROR_CODE;
@@ -189,6 +190,7 @@ void CommunicatorImplLite::UpdateCommParam(HcclKernelParamLite *kernelParam)
     devPhyId      = kernelParam->comm.devPhyId;
     devType       = kernelParam->comm.devType;
     opCounterAddr = kernelParam->comm.opCounterAddr;
+    opIndex       = kernelParam->comm.opIndex;
     hcclExecTimeout = kernelParam->envConfig.hcclExecTimeout;
     if (rmtDataBufferMgr == nullptr) {
         collAlgInfo   = std::make_unique<CollAlgInfo>(kernelParam->op.algOperator.opMode, kernelParam->opTag);
@@ -448,12 +450,17 @@ void CommunicatorImplLite::InitCurrentOp(HcclKernelParamLite *kernelParam)
 
 void CommunicatorImplLite::SetDfxOpInfo(uint64_t beginTime)
 {
+    u64 size = 4;
     auto dfxopInfo           = std::make_shared<DfxOpInfo>();
     dfxopInfo->op_           = currentOp;
     dfxopInfo->algType_      = AlgType::MESH; // 暂时
-    dfxopInfo->index_        = idIndex_;
+    dfxopInfo->commIndex_    = idIndex_;
     dfxopInfo->beginTime_    = beginTime;
     dfxopInfo->comm_         = this;
+    dfxopInfo->commId_       = commId;
+    dfxopInfo->opIndex_      = opIndex;
+    dfxopInfo->headOpCounter_ = *(reinterpret_cast<u32 *>(opCounterAddr + size));
+    dfxopInfo->tailOpCounter_ = *(reinterpret_cast<u32 *>(opCounterAddr + size * 2));
     CHECK_NULLPTR(streamLiteMgr->GetMaster(), "[SetDfxOpInfo]master stream is nullptr!");
     dfxopInfo->mainStreamId_ = streamLiteMgr->GetMaster()->GetId();
     mirrorTaskMgr->SetCurrDfxOpInfo(dfxopInfo);

```

### src/legacy/framework/communicator/aicpu/communicator_impl_lite.h
```diff
@@ -237,6 +237,7 @@ private:
     u64 scratchSize{0};
     u64 locBuffer[BufferType::__COUNT__]{};
     u64 opCounterAddr{0};
+    u32 opIndex{0};
     std::string commId;
     bool isUpdateComm {false};
     CollOperator currentOp;

```

### src/legacy/framework/communicator/aicpu/kernel_entrance.cc
```diff
@@ -33,9 +33,9 @@ uint32_t HcclKernelEntrance(void *args)
     NsRecoveryHandlerFunc::GetInstance();
 
     u32 commIdIndex = kernelParam->comm.idIndex;
-    HCCL_RUN_INFO("HcclKernelEntrance begin, OpType[%s] algName[%s] commIdIndex[%u] commId[%s] opTag[%s], devPhyId[%u] myRank[%u] rankSie[%u]",
+    HCCL_RUN_INFO("HcclKernelEntrance begin, OpType[%s] algName[%s] commIdIndex[%u] commId[%s] opTag[%s], devPhyId[%u] myRank[%u] rankSie[%u] opIndex[%u]",
         kernelParam->op.algOperator.opType.Describe().c_str(), kernelParam->algName, commIdIndex, kernelParam->comm.commId,
-        kernelParam->opTag, kernelParam->comm.devPhyId, kernelParam->comm.myRank, kernelParam->comm.rankSie);
+        kernelParam->opTag, kernelParam->comm.devPhyId, kernelParam->comm.myRank, kernelParam->comm.rankSie, kernelParam->comm.opIndex);
     Hccl::CommunicatorImplLite *communicatorImplLite = CommunicatorImplLiteMgr::GetInstance().Get(commIdIndex);
     if (communicatorImplLite == nullptr) {
         HCCL_ERROR("HcclKernelEntrance communicatorImplLite is null.");

```

### src/legacy/framework/communicator/aicpu/kernel_param_lite.h
```diff
@@ -38,6 +38,7 @@ struct HcclAicpuCommunicatorLite {
     HcclAicpuLocBufLite opBaseScratch;
     uint64_t            opCounterAddr;
     char                commId[COMM_NAME_MAX_LENGTH]{0};
+    u32                 opIndex{0};
 };
 
 struct HcclAicpuOpLite {

```

### src/legacy/framework/communicator/communicator_impl.cc
```diff
@@ -442,10 +442,14 @@ bool CommunicatorImpl::TryFastCcuLaunch(const CollOpParams &opParams, aclrtStrea
         dfxOpInfo->op_           = *GetCurrentCollOperator();
         dfxOpInfo->tag_          = OpTypeToString(dfxOpInfo->op_.opType);
         dfxOpInfo->algType_      = AlgType::MESH;
-        dfxOpInfo->index_        = GetIdIndex();
+        dfxOpInfo->commIndex_    = GetIdIndex();
         dfxOpInfo->comm_         = this;
         dfxOpInfo->mainStreamId_ = HrtGetStreamId(stream);
         dfxOpInfo->beginTime_    = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
+        dfxOpInfo->commId_       = id;
+        dfxOpInfo->opIndex_      = opIndex;
+        dfxOpInfo->headOpCounter_ = 0x7fffffff;
+        dfxOpInfo->tailOpCounter_ = 0x7fffffff;
         GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
         ExecuteFastCcuLaunch(opParams, stream, params);
         ReportProfInfo(beginTime, opParams.staticShape, true);
@@ -535,6 +539,7 @@ void CommunicatorImpl::ExecuteFastCcuLaunch(const CollOpParams &opParams, aclrtS
     collOpIndex++;
     submittedOpCnt = collOpIndex;
     opBaseOpIndex++;
+    opIndex++;
     status = CommStatus::COMM_READY;
 }
 
@@ -614,6 +619,7 @@ HcclResult CommunicatorImpl::LoadOpbasedCollOp(const CollOpParams &opParams, voi
         TraceEndInfo(startut, endut, opParams);
         RefreshSubmittedOpcnt();
         opBaseOpIndex++;
+        opIndex++;
         status = CommStatus::COMM_READY;
     } catch (HcclException &e) {
         status = CommStatus::COMM_READY;
@@ -871,6 +877,7 @@ HcclResult CommunicatorImpl::LoadOffloadCollOp(std::string &opTag, const CollOpP
         ReportProfInfo(beginTime, opParams.staticShape, false);
         HcclUs endut = std::chrono::steady_clock::now();
         TraceEndInfo(startut, endut, opParams);
+        opIndex++;
     } catch (HcclException &e) {
         status = CommStatus::COMM_READY;
         HCCL_ERROR(e.what());
@@ -1398,11 +1405,17 @@ u32 CommunicatorImpl::GetSubmittedOpCnt() const
 {
     return submittedOpCnt;
 }
+
 u32 CommunicatorImpl::GetOpBaseOpIndex() const
 {
     return opBaseOpIndex;
 }
 
+u32 CommunicatorImpl::GetOpIndex() const
+{
+    return opIndex;
+}
+
 bool CommunicatorImpl::GetOpAiCpuTSFeatureFlag() const
 {
     return opExecuteConfig.accState == AcceleratorState::AICPU_TS;

```

### src/legacy/framework/communicator/communicator_impl.h
```diff
@@ -162,6 +162,8 @@ public:
 
     virtual u32 GetOpBaseOpIndex() const;
 
+    virtual u32 GetOpIndex() const;
+
     u32 GetSubmittedOpCnt() const;
 
     HDCommunicate &GetKfcControlTransferH2D() const;
@@ -421,6 +423,7 @@ private:
     u32 step           = 0; // 全局device信息的step
     u32 opBaseOpIndex  = 0; // 单算子次数
     u32 collOpIndex    = 0; // 集合通信算子次数
+    u32 opIndex        = 0; // 算子总计数(单算子/图模式/CCU快速下发)
     u32 sendRecvIndex  = 0; // send/recv 算子次数
     u32 submittedOpCnt = 0;
     u32 aivCoreLimit = 0;

```

### src/legacy/framework/communicator/hccl_communicator.cc
```diff
@@ -516,4 +516,9 @@ HcclResult HcclCommunicator::GetEndpointInfo(uint32_t rankId, const EndpointDesc
     return pimpl->GetEndpointInfo(rankId, endpointDesc, endpointAttr, infoLen, info);
 }
 
+u32 HcclCommunicator::GetOpIndex() const
+{
+    return pimpl->GetOpIndex();
+}
+
 } // namespace Hccl

```

### src/legacy/framework/dfx/common/task_info.cc
```diff
@@ -122,12 +122,15 @@ string TaskInfo::GetOpInfo() const
             static_cast<u64>(opInfo->op_.inputMem->GetAddr()),
             static_cast<u64>(opInfo->op_.outputMem->GetAddr()));
     }
-    return StringFormat("index[%u], count[%llu], reduceType[%s], %sdataType[%s]",
-        opInfo->index_,
+    return StringFormat("commIndex[%u], count[%llu], reduceType[%s], %sdataType[%s], opIndex[%u], headOpCounter[%u], tailOpCounter[%u]",
+        opInfo->commIndex_,
         opInfo->op_.dataCount,
         opInfo->op_.reduceOp.Describe().c_str(),
         addr.c_str(),
-        opInfo->op_.dataType.Describe().c_str());
+        opInfo->op_.dataType.Describe().c_str(),
+        opInfo->opIndex_,
+        opInfo->headOpCounter_,
+        opInfo->tailOpCounter_);
 }
 
 string TaskInfo::GetRemoteRankInfo(bool needConcise) const

```

### src/legacy/framework/dfx/common/task_info.h
```diff
@@ -23,18 +23,22 @@ public:
     CollOperator op_;
     std::string  tag_;
     AlgType      algType_;
-    u32          index_;
+    u32          commIndex_;
     u64          beginTime_;
     u64          endTime_;
     void        *comm_;
     u32          mainStreamId_;
+    std::string  commId_;
+    u32          opIndex_;
+    u32          headOpCounter_;
+    u32          tailOpCounter_;
 
 public:
     std::string Describe() const
     {
         return StringFormat(
-            "DfxOpInfo: [collOperator:[%s], tag:[%s], algType:[%u], index:[%u], beginTime:[%llu], endTime:[%llu]",
-            CollOpToString(op_).c_str(), tag_.c_str(), algType_, index_, beginTime_, endTime_);
+            "DfxOpInfo: [collOperator:[%s], tag:[%s], algType:[%u], commIndex:[%u], commId[%s], beginTime:[%llu], endTime:[%llu], opIndex[%u], headOpCounter[%u], tailOpCounter[%u]",
+            CollOpToString(op_).c_str(), tag_.c_str(), algType_, commIndex_, commId_.c_str(), beginTime_, endTime_, opIndex_, headOpCounter_, tailOpCounter_);
     }
 };
 

```

### src/legacy/framework/entrance/hcom/hcom_v2.cc
```diff
@@ -84,6 +84,11 @@ inline Hccl::CollOpParams GetHcclOpParams(void *inputPtr, void *outputPtr, u64 c
     return opParams;
 }
 
+static void PrintOpTagAndComm(std::string tag, u32 opIndex)
+{
+    HCCL_RUN_INFO("Entry-[%s] V950 OpIndex[%u]", tag.c_str(), opIndex);
+}
+
 HcclResult HcomAllGatherV2(const char *tag, void *inputPtr, void *outputPtr, u64 inputCount, HcclDataType dataType,
     const char *group, rtStream_t stream)
 {
@@ -99,6 +104,7 @@ HcclResult HcomAllGatherV2(const char *tag, void *inputPtr, void *outputPtr, u64
     /* 入参的正确性由HCCL确保 */
     Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, inputCount, dataType, Hccl::OpType::ALLGATHER);
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
 
     /* 关键状态记录 */
@@ -150,6 +156,7 @@ HcclResult HcomAllGatherVV2(const char *tag, void *sendBuf, u64 sendCount, void
     opParams.vDataDes.counts = recvCounts;
     opParams.vDataDes.displs = rdispls;
     opParams.vDataDes.dataType = HcclDataTypeToDataType(dataType);
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
     /* 关键状态记录 */
     HCCL_RUN_INFO("hcom allgatherv success,take time [%lld]us, tag[%s], sendBuf[%p], sendCount[%llu], "\
@@ -177,6 +184,7 @@ HcclResult HcomAllReduceV2(const char *tag, void *inputPtr, void *outputPtr, u64
     /* 入参的正确性由HCCL确保 */
     Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::ALLREDUCE, op);
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
 
     /* 关键状态记录 */
@@ -205,6 +213,7 @@ HcclResult HcomReduceScatterV2(const char *tag, void *inputPtr, void *outputPtr,
     /* 入参的正确性由HCCL确保 */
     Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::REDUCESCATTER, op);
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
     /* 关键状态记录 */
     HCCL_RUN_INFO("hcom reducescatter success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "\
@@ -258,6 +267,7 @@ HcclResult HcomReduceScatterVV2(const char *tag, void *sendBuf, void *sendCounts
     opParams.vDataDes.counts = sendCounts;
     opParams.vDataDes.displs = sdispls;
     opParams.vDataDes.dataType = HcclDataTypeToDataType(dataType);
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
     /* 关键状态记录 */
     HCCL_RUN_INFO("hcom reducescatterv success,take time [%lld]us, tag[%s], sendBuf[%p], sendCounts[%p], "\
@@ -283,6 +293,7 @@ HcclResult HcomSendV2(const char *tag, void *inputPtr, u64 count, HcclDataType d
     Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, nullptr, count, dataType, Hccl::OpType::SEND);
     opParams.dstRank = destRank;
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
     /* 关键状态记录 */
     HCCL_RUN_INFO("hcom send success,time[%lld]us,tag[%s],inputPtr[%p],count[%llu],dataType[%s],destRank[%u],"
@@ -308,6 +319,7 @@ HcclResult HcomReceiveV2(const char *tag, void *outputPtr, u64 count, HcclDataTy
     Hccl::CollOpParams opParams = GetHcclOpParams(nullptr, outputPtr, count, dataType, Hccl::OpType::RECV);
     opParams.dstRank = srcRank;
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
     /* 关键状态记录 */
     HCCL_RUN_INFO("hcom receive success,time[%lld]us,tag[%s],outputPtr[%p],count[%llu],dataType[%s],srcRank[%u],"
@@ -438,6 +450,7 @@ HcclResult HcomAlltoAllVV2(const void *sendBuf, const void *sendCounts, const vo
     opParams.all2AllVDataDes.rdispls = const_cast<void*>(rdispls);
     opParams.dataType = HcclDataTypeToDataType(sendType);
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
     
     /* 关键状态记录 */
@@ -476,6 +489,7 @@ HcclResult HcomAlltoAllVCV2(const void *sendBuf, const void *sendCountMatrix, Hc
     opParams.all2AllVCDataDes.sendCountMatrix = const_cast<void*>(sendCountMatrix);
     opParams.dataType = HcclDataTypeToDataType(sendType);
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
 
     /* 关键状态记录 */
@@ -509,6 +523,7 @@ HcclResult HcomAlltoAllV2(const void *sendBuf, u64 sendCount, HcclDataType sendT
     opParams.all2AllDataDes.recvType = HcclDataTypeToDataType(recvType);
     opParams.dataType = HcclDataTypeToDataType(sendType);
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
 
     HCCL_RUN_INFO("HcomAlltoAll success,take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], "\
@@ -579,6 +594,7 @@ HcclResult HcomBroadcastV2(const char *tag, void *ptr, u64 count, HcclDataType d
     Hccl::CollOpParams opParams = GetHcclOpParams(ptr, ptr, count, dataType, Hccl::OpType::BROADCAST);
     opParams.root = root;
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
 
     /* 关键状态记录 */
@@ -609,6 +625,7 @@ HcclResult HcomReduceV2(const char *tag, void *inputPtr, void *outputPtr, u64 co
     Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::REDUCE, op);
     opParams.root = root;
     std::string opTag = tag;
+    PrintOpTagAndComm(opTag, hcclComm->GetOpIndex());
     CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
 
     /* 关键状态记录 */

```

### src/legacy/framework/entrance/op_base/op_base_v2.cc
```diff
@@ -610,9 +610,9 @@ HcclResult HcclCommDestroyV2(HcclComm comm)
     return HCCL_SUCCESS;
 }
 
-static void PrintOpTagAndComm(std::string tag)
+static void PrintOpTagAndComm(std::string tag, u32 opIndex)
 {
-    HCCL_RUN_INFO("Entry-[%s] V910_95", tag.c_str());
+    HCCL_RUN_INFO("Entry-[%s] V950 OpIndex[%u]", tag.c_str());
 }
 
 
@@ -621,7 +621,7 @@ HcclResult HcclAlltoAllV2(const void *sendBuf, uint64_t sendCount, HcclDataType
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "ALLTOALL_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET(HcomCheckOpParamV2(tag.c_str(), 0, sendType, stream));
     CHK_RET(HcomCheckDataTypeV2(recvType));
     static thread_local Hccl::CollOpParams opParams;
@@ -644,7 +644,7 @@ HcclResult HcclAlltoAllVV2(const void *sendBuf, const void *sendCounts, const vo
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "HCCL_ALLTOALLV_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), 0, sendType, stream), tag.c_str());
     CHK_RET_AND_PRINT_IDE(HcomCheckDataTypeV2(recvType), tag.c_str());
     CHK_RET(HcomCheckDataTypeV2(sendType));
@@ -849,7 +849,7 @@ HcclResult HcclAlltoAllVCV2(const void *sendBuf, const void *sendCountMatrix, Hc
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "HCCL_ALLTOALLVC_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), 0, sendType, stream), tag.c_str());
     CHK_RET_AND_PRINT_IDE(HcomCheckDataTypeV2(recvType), tag.c_str());
     u32 rankSize = 0;
@@ -874,7 +874,7 @@ HcclResult HcclReduceV2(void *sendBuf, void *recvBuf, uint64_t count, HcclDataTy
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "Reduce_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());
     CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), tag.c_str());
     CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataTypeV2(dataType, op), tag.c_str());
@@ -900,7 +900,7 @@ HcclResult HcclAllReduceV2(void *sendBuf, void *recvBuf, uint64_t count, HcclDat
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "AllReduce_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());
 
     CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), tag.c_str());
@@ -923,7 +923,7 @@ HcclResult HcclBroadcastV2(void *buf, uint64_t count, HcclDataType dataType, uin
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "Broadcast_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());
     u32 rankSize = INVALID_VALUE_RANKSIZE;
     CHK_RET_AND_PRINT_IDE(communicator->GetRankSize(&rankSize), tag.c_str());
@@ -1353,7 +1353,7 @@ HcclResult HcclScatterV2(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclD
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "Scatter_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), recvCount, dataType, stream), tag.c_str());
     u32 rankSize = INVALID_VALUE_RANKSIZE;
     CHK_RET_AND_PRINT_IDE(communicator->GetRankSize(&rankSize), tag.c_str());
@@ -1381,7 +1381,7 @@ HcclResult HcclAllGatherV2(void *sendBuf, void *recvBuf, uint64_t sendCount, Hcc
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "AllGather_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), sendCount, dataType, stream), tag.c_str());
     static thread_local Hccl::CollOpParams opParams;
     opParams.opType = Hccl::OpType::ALLGATHER;
@@ -1400,7 +1400,7 @@ HcclResult HcclAllGatherVV2(void *sendBuf, uint64_t sendCount, void *recvBuf, vo
     // 获取通信域
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "AllGatherV_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     // 获取rank信息
     uint32_t rankId;
     CHK_RET(communicator->GetRankId(rankId));
@@ -1442,7 +1442,7 @@ HcclResult HcclSendV2(
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "Send_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET(HcomCheckDataTypeV2(dataType));
     static thread_local Hccl::CollOpParams opParams{};
     opParams.opType = Hccl::OpType::SEND;
@@ -1462,7 +1462,7 @@ HcclResult HcclRecvV2(
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "Recv_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET(HcomCheckDataTypeV2(dataType));
     static thread_local Hccl::CollOpParams opParams{};
     opParams.opType = Hccl::OpType::RECV;
@@ -1482,7 +1482,7 @@ HcclResult HcclReduceScatterV2(void *sendBuf, void *recvBuf, uint64_t recvCount,
 {
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "ReduceScatter_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), recvCount, dataType, stream), tag.c_str());
     CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), tag.c_str());
     CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataTypeV2(dataType, op), tag.c_str());
@@ -1505,7 +1505,7 @@ HcclResult HcclReduceScatterVV2(void *sendBuf, void *sendCounts, void *sendDispl
     // 获取通信域
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "ReduceScatterV_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     // 获取rank信息
     uint32_t rankId;
     CHK_RET(communicator->GetRankId(rankId));
@@ -1554,7 +1554,7 @@ HcclResult HcclBatchSendRecvV2(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum,
     CHK_PTR_NULL(comm);
     Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
     const std::string tag = "HcclBatchSendRecvV2_" + communicator->GetId();
-    PrintOpTagAndComm(tag);
+    PrintOpTagAndComm(tag, communicator->GetOpIndex());
     CHK_PTR_NULL(stream);
     CHK_PTR_NULL(sendRecvInfo);
     CHK_PRT_RET(itemNum == 0, HCCL_WARNING("[BatchSendRecv] taskList itemNum is zero."), HCCL_SUCCESS);

```

### src/legacy/framework/service/coll_service_base.cc
```diff
@@ -357,14 +357,19 @@ void CollServiceBase::SaveMirrorDfxOpInfo()
 {
     auto dfxOpInfo = std::make_shared<DfxOpInfo>();
     CHECK_NULLPTR(comm, "[CollServiceBase::SaveMirrorDfxOpInfo] comm is nullptr!");
+    std::pair<u32,u32> counter = GetOpCount();
 
     dfxOpInfo->op_ = *comm->GetCurrentCollOperator();
     dfxOpInfo->tag_ = OpTypeToString(dfxOpInfo->op_.opType);
     dfxOpInfo->algType_ = AlgType::MESH;
-    dfxOpInfo->index_ = comm->GetIdIndex();
+    dfxOpInfo->commIndex_ = comm->GetIdIndex();
     dfxOpInfo->comm_ = comm;
     dfxOpInfo->mainStreamId_ = comm->GetStreamManager().GetMaster()->GetId();
     dfxOpInfo->beginTime_ = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
+    dfxOpInfo->commId_ = comm->GetId();
+    dfxOpInfo->opIndex_ = comm->GetOpIndex();
+    dfxOpInfo->headOpCounter_ = counter.first;
+    dfxOpInfo->tailOpCounter_ = counter.second;
 
     comm->GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
 }

```

### src/legacy/include/hccl_communicator.h
```diff
@@ -129,6 +129,7 @@ public:
                        void* info);
  
     u32 GetDeviceLogicId() const;
+    u32 GetOpIndex() const;
  
 private:
     CommParams                        commParams;

```

### test/legacy/ut/framework/dfx/aicpu/profiling/ut_profiling_handler_lite.cc
```diff
@@ -103,10 +103,14 @@ TEST_F(ProfilingHandlerLiteTest, ReportHcclOpInfo_test)
     op.staticAddr = false;
     dfxOpInfo->op_ = op;
     dfxOpInfo->tag_ = "testTag";
-    dfxOpInfo->index_ = 0;
+    dfxOpInfo->commIndex_ = 0;
     dfxOpInfo->beginTime_ = 0;
     dfxOpInfo->endTime_ = 1;
     dfxOpInfo->comm_ = &comm;
+    dfxOpInfo->commId_ = "testTag";
+    dfxOpInfo->opIndex_ = 0;
+    dfxOpInfo->headOpCounter_ = 0;
+    dfxOpInfo->tailOpCounter_ = 0;
     mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
     ableNum = 0;
     handler.enableHcclL0_ = true;
@@ -125,10 +129,14 @@ TEST_F(ProfilingHandlerLiteTest, ReportHcclOpInfo1_test)
     op.staticAddr = false;
     dfxOpInfo->op_ = op;
     dfxOpInfo->tag_ = "testTag";
-    dfxOpInfo->index_ = 0;
+    dfxOpInfo->commIndex_ = 0;
     dfxOpInfo->beginTime_ = 0;
     dfxOpInfo->endTime_ = 1;
     dfxOpInfo->comm_ = &comm;
+    dfxOpInfo->commId_ = "testTag";
+    dfxOpInfo->opIndex_ = 0;
+    dfxOpInfo->headOpCounter_ = 0;
+    dfxOpInfo->tailOpCounter_ = 0;
     mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
     ableNum = 1;
     handler.enableHcclL0_ = true;
@@ -146,10 +154,14 @@ TEST_F(ProfilingHandlerLiteTest, ReportHcclOpInfo2_test)
     op.staticAddr = false;
     dfxOpInfo->op_ = op;
     dfxOpInfo->tag_ = "testTag";
-    dfxOpInfo->index_ = 0;
+    dfxOpInfo->commIndex_ = 0;
     dfxOpInfo->beginTime_ = 0;
     dfxOpInfo->endTime_ = 1;
     dfxOpInfo->comm_ = &comm;
+    dfxOpInfo->commId_ = "testTag";
+    dfxOpInfo->opIndex_ = 0;
+    dfxOpInfo->headOpCounter_ = 0;
+    dfxOpInfo->tailOpCounter_ = 0;
     mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
     ableNum = 2;
     handler.enableHcclL0_ = true;

```

### test/legacy/ut/framework/dfx/common/ut_task_info.cc
```diff
@@ -212,15 +212,18 @@ TEST_F(TaskInfoTest, test_get_op_info)
 {
     TaskInfo taskInfo = InitTaskInfo();
 
-    taskInfo.dfxOpInfo_->index_ = 3;
+    taskInfo.dfxOpInfo_->opIndex_ = 0;
+    taskInfo.dfxOpInfo_->commIndex_ = 3;
+    taskInfo.dfxOpInfo_->headOpCounter_ = 0;
+    taskInfo.dfxOpInfo_->tailOpCounter_ = 0;
     taskInfo.dfxOpInfo_->op_.dataCount = 0xaaaabbbbcccc;
     taskInfo.dfxOpInfo_->op_.reduceOp = ReduceOp::SUM;
     taskInfo.dfxOpInfo_->op_.dataType = DataType::UINT64;
-    EXPECT_EQ(taskInfo.GetOpInfo(), "index[3], count[187650270809292], reduceType[ReduceOp::SUM], dataType[DataType::UINT64]");
+    EXPECT_EQ(taskInfo.GetOpInfo(), "commIndex[3], count[187650270809292], reduceType[ReduceOp::SUM], dataType[DataType::UINT64], opIndex[0], headOpCounter[0], tailOpCounter[0]");
 
     taskInfo.dfxOpInfo_->op_.inputMem = make_shared<Buffer>(0x111122223333, 0);
     taskInfo.dfxOpInfo_->op_.outputMem = make_shared<Buffer>(0xaaaabbbbcccc, 0);
-    EXPECT_EQ(taskInfo.GetOpInfo(), "index[3], count[187650270809292], reduceType[ReduceOp::SUM], src:[0x111122223333], dst:[0xaaaabbbbcccc], dataType[DataType::UINT64]");
+    EXPECT_EQ(taskInfo.GetOpInfo(), "commIndex[3], count[187650270809292], reduceType[ReduceOp::SUM], src:[0x111122223333], dst:[0xaaaabbbbcccc], dataType[DataType::UINT64], opIndex[0], headOpCounter[0], tailOpCounter[0]");
 
     taskInfo.dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
     EXPECT_EQ(taskInfo.GetOpInfo(), "");

```
