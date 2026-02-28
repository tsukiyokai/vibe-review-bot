# PR #1102: CpuRoceEndpoint and HostCpuRoceChannel

- 作者: zwlStateflow
- 分支: hostdpu4 -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/1102
- 描述: CpuRoceEndpoint and HostCpuRoceChannel
        modified:   src/framework/next/coll_comms/api_c_adpt/coll_comm_res_c_adpt.cc
        modified:   src/framework/next/comms/endpoint_pairs/channels/channel.cc
        modified:   src/framework/next/comms/endpoint_pairs/channels/host/host_cpu_roce_channel.cc
        modified:   src/framework/next/comms/endpoint_pairs/channels/host/host_cpu_roce_channel.h
        modified:   src/framework/next/comms/endpoint_pairs/channels/host/host_rdma_connection.cc
 

## 变更文件 (22 个, 其中 C/C++ 文件 22 个)

- [modified] src/framework/next/coll_comms/api_c_adpt/coll_comm_res_c_adpt.cc (+1, -1) *
- [modified] src/framework/next/comms/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc (+1, -1) *
- [modified] src/framework/next/comms/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.h (+1, -1) *
- [modified] src/framework/next/comms/endpoint_pairs/channels/channel.cc (+2, -3) *
- [modified] src/framework/next/comms/endpoint_pairs/channels/channel.h (+1, -1) *
- [modified] src/framework/next/comms/endpoint_pairs/channels/host/host_cpu_roce_channel.cc (+87, -162) *
- [modified] src/framework/next/comms/endpoint_pairs/channels/host/host_cpu_roce_channel.h (+26, -29) *
- [modified] src/framework/next/comms/endpoint_pairs/channels/host/host_rdma_connection.cc (+6, -7) *
- [modified] src/framework/next/comms/endpoints/reged_mems/roce_mem.cc (+3, -0) *
- [modified] src/framework/op_base/src/op_base.cc (+2, -1) *
- [modified] src/framework/op_base/src/op_base_mc2.cc (+6, -1) *
- [modified] src/legacy/framework/communicator/communicator_impl.cc (+4, -5) *
- [modified] src/legacy/framework/communicator/communicator_impl.h (+2, -2) *
- [modified] src/legacy/framework/communicator/hccl_communicator.cc (+0, -5) *
- [modified] src/legacy/framework/entrance/op_base/op_base_v2.cc (+0, -10) *
- [modified] src/legacy/framework/entrance/op_base/op_base_v2.h (+0, -1) *
- [modified] src/legacy/include/hccl_communicator.h (+0, -2) *
- [modified] src/legacy/unified_platform/resource/buffer/exchange_rdma_buffer_dto.h (+7, -5) *
- [modified] src/legacy/unified_platform/resource/buffer/local_rdma_rma_buffer.cc (+1, -1) *
- [modified] src/legacy/unified_platform/resource/buffer/remote_rma_buffer.cc (+2, -2) *
- [modified] src/platform/common/adapter/adapter_rts.cc (+1, -1) *
- [modified] test/ut/stub/llt_next_orion_stub.cc (+1, -1) *

## Diff 内容

### src/framework/next/coll_comms/api_c_adpt/coll_comm_res_c_adpt.cc
```diff
@@ -69,7 +69,7 @@ HcclResult ProcessHcclResPackReq(const HcclChannelDesc &channelDesc, HcclChannel
                 channelDescFinal.roceAttr.retryInterval = (channelDesc.roceAttr.retryInterval == INVALID_UINT) ? EnvConfig::GetExternalInputRdmaTimeOut() : channelDesc.roceAttr.retryInterval;
                 channelDescFinal.roceAttr.tc = (channelDesc.roceAttr.tc == 0xFF) ? EnvConfig::GetExternalInputRdmaTrafficClass() : channelDesc.roceAttr.tc;
                 channelDescFinal.roceAttr.sl = (channelDesc.roceAttr.sl == 0xFF) ? EnvConfig::GetExternalInputRdmaServerLevel() : channelDesc.roceAttr.sl;
-                HCCL_INFO("[%s]queueNum[%u], retryCnt[%u], retryInterval[%u], tc[%u], sl[%u]",
+                HCCL_INFO("[%s]queueNum[%u], retryCnt[%u], retryInterval[%u], tc[%u], sl[%u]", __func__,
                     channelDescFinal.roceAttr.queueNum, channelDescFinal.roceAttr.retryCnt, channelDescFinal.roceAttr.retryInterval,
                     channelDescFinal.roceAttr.tc, channelDescFinal.roceAttr.sl);
                 break;

```

### src/framework/next/comms/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.cc
```diff
@@ -176,7 +176,7 @@ HcclResult AicpuTsUrmaChannel::GetNotifyNum(uint32_t *notifyNum) const
     return HCCL_SUCCESS;
 }
 
-HcclResult AicpuTsUrmaChannel::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags) const
+HcclResult AicpuTsUrmaChannel::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags)
 {
     return memTransport_->GetRemoteMem(remoteMem, memNum, memTags);
 }

```

### src/framework/next/comms/endpoint_pairs/channels/aicpu/aicpu_ts_urma_channel.h
```diff
@@ -28,7 +28,7 @@ public:
 
     HcclResult Init() override;
     HcclResult GetNotifyNum(uint32_t *notifyNum) const override;
-    HcclResult GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags) const override;
+    HcclResult GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags) override;
     ChannelStatus GetStatus() override;
 
     HcclResult H2DResPack(std::vector<char>& buffer);

```

### src/framework/next/comms/endpoint_pairs/channels/channel.cc
```diff
@@ -28,9 +28,8 @@ HcclResult Channel::CreateChannel(
         case COMM_ENGINE_CPU:
             // TODO: if 判断 EndpointDesc 里面的协议
             if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE) {
-                channelPtr.reset(new (std::nothrow) HostCpuRoceChannel(
-                    endpointHandle, channelDesc
-                ));
+                EXECEPTION_CATCH(channelPtr = std::make_unique<HostCpuRoceChannel>(endpointHandle, channelDesc),
+                    return HCCL_E_PARA);
                 break;
             }
             HCCL_ERROR("[Channel][%s] CommEngine[COMM_ENGINE_CPU] not support", __func__);

```

### src/framework/next/comms/endpoint_pairs/channels/channel.h
```diff
@@ -50,7 +50,7 @@ public:
     // ------------------ 控制面接口 ------------------
     virtual HcclResult Init() = 0;
     virtual HcclResult GetNotifyNum(uint32_t *notifyNum) const = 0;
-    virtual HcclResult GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags) const = 0;
+    virtual HcclResult GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags) = 0;
     virtual ChannelStatus GetStatus() = 0;
 
     // ------------------ 数据面接口 ------------------

```

### src/framework/next/comms/endpoint_pairs/channels/host/host_cpu_roce_channel.cc
```diff
@@ -21,6 +21,7 @@
 
 namespace hcomm {
 constexpr u32 FENCE_TIMEOUT_MS = 30 * 1000; // 定义最大等待30秒
+constexpr u32 MEM_BLOCK_SIZE = 128;
 
 HostCpuRoceChannel::HostCpuRoceChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc)
     : endpointHandle_(endpointHandle), channelDesc_(channelDesc) {}
@@ -35,7 +36,7 @@ HostCpuRoceChannel::~HostCpuRoceChannel() {
 HcclResult HostCpuRoceChannel::ParseInputParam()
 {
     // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
-    // TODO: 待 endpoint 实现
+    CHK_PTR_NULL(endpointHandle_);
     Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
     localEp_ = localEpPtr->GetEndpointDesc();
     rdmaHandle_ = localEpPtr->GetRdmaHandle();
@@ -43,17 +44,16 @@ HcclResult HostCpuRoceChannel::ParseInputParam()
 
     // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
     remoteEp_ = channelDesc_.remoteEndpoint;
-    socket_.reset(reinterpret_cast<Hccl::Socket*>(channelDesc_.socket));
+    socket_ = reinterpret_cast<Hccl::Socket*>(channelDesc_.socket);
     CHK_PTR_NULL(socket_);
     notifyNum_ = channelDesc_.notifyNum;
 
     // 3. 从 channelDesc 的 memHandle，获得 bufs_
-    // TODO: 待 memHandle 实现
-    // memHandles 对应是一个裸指针还是一个解引用的智能指针？
+    CHK_PTR_NULL(channelDesc_.memHandles);
     for (uint32_t i = 0; i < channelDesc_.memHandleNum; ++i) {
-        std::unique_ptr<Hccl::LocalRdmaRmaBuffer> localRdmaBuffer =
-            std::unique_ptr<Hccl::LocalRdmaRmaBuffer>(static_cast<Hccl::LocalRdmaRmaBuffer *>(channelDesc_.memHandles[i]));
-        localRmaBuffers_.emplace_back(std::move(localRdmaBuffer));
+        CHK_PTR_NULL(channelDesc_.memHandles[i]);
+        Hccl::LocalRdmaRmaBuffer* localRdmaBuffer = static_cast<Hccl::LocalRdmaRmaBuffer *>(channelDesc_.memHandles[i]);
+        localRmaBuffers_.emplace_back(localRdmaBuffer);
     }
 
     return HCCL_SUCCESS;
@@ -71,7 +71,7 @@ HcclResult HostCpuRoceChannel::BuildConnection()
 {
     std::unique_ptr<HostRdmaConnection> conn;
     EXECEPTION_CATCH(
-        conn = std::make_unique<HostRdmaConnection>(socket_.get(), rdmaHandle_),
+        conn = std::make_unique<HostRdmaConnection>(socket_, rdmaHandle_),
         return HCCL_E_INTERNAL);
     CHK_PTR_NULL(conn);
     CHK_RET(conn->Init());
@@ -86,14 +86,10 @@ HcclResult HostCpuRoceChannel::BuildNotify()
     return HCCL_SUCCESS;
 }
 
-// TODO: 内存注册改到 endpoint 上
+// NotifyRecord使用的内存
 HcclResult HostCpuRoceChannel::BuildBuffer()
 {
-    // for (uint32_t i = 0; i < bufs_.size(); ++i) {
-    //     localRmaBuffers_.emplace_back(std::move(
-    //         std::make_unique<Hccl::LocalRdmaRmaBuffer>(bufs_[i], rdmaHandle_)
-    //     ));
-    // }
+    // TODO: 追加构造NotifyRecord使用的LocalRdmaRmaBuffer，使用malloc创建Host侧内存
     bufferNum_ = localRmaBuffers_.size();
     return HCCL_SUCCESS;
 }
@@ -108,12 +104,9 @@ HcclResult HostCpuRoceChannel::Init()
     return HCCL_SUCCESS;
 }
 
-// 当前AICPU和框架没有改为返回错误码形式，所有暂时使用改方法转换
+// 当前AICPU和框架没有改为返回错误码形式，所有暂时使用该方法转换
 ChannelStatus HostCpuRoceChannel::GetStatus()
 {
-    // EXCEPTION_HANDLE_BEGIN
-    // EXCEPTION_HANDLE_END
-
     ChannelStatus status;
     HcclResult ret = GetStatus(status);
     if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
@@ -137,11 +130,13 @@ HcclResult HostCpuRoceChannel::GetStatus(ChannelStatus &status) {
         case RdmaStatus::QP_CREATED:
             // 发送交换数据
             CHK_RET(ExchangeData());
-            rdmaStatus_ = RdmaStatus::DATE_EXCHANG;
+            rdmaStatus_ = RdmaStatus::DATA_EXCHANGE;
             break;
-        case RdmaStatus::DATE_EXCHANG:
+        case RdmaStatus::DATA_EXCHANGE:
             CHK_RET(ModifyQp());
             rdmaStatus_ = RdmaStatus::QP_MODIFIED;
+        case RdmaStatus::QP_MODIFIED:
+            // TODO: Prepare Rqe
         default:
             rdmaStatus_ = RdmaStatus::CONN_OK;
             channelStatus_ = ChannelStatus::READY;
@@ -196,7 +191,7 @@ HcclResult HostCpuRoceChannel::ExchangeData()
 
     // 同步数据打包
     Hccl::BinaryStream binaryStream;
-    // HandshakeMsgPack(binaryStream);
+    // HandshakeMsgPack(binaryStream); // attr的数据看上去没有起到作用，先注释
     NotifyVecPack(binaryStream);
     CHK_RET(BufferVecPack(binaryStream));
     CHK_RET(ConnVecPack(binaryStream));
@@ -372,12 +367,11 @@ HcclResult HostCpuRoceChannel::ModifyQp() {
     return HCCL_SUCCESS;
 }
 
-HcclResult HostCpuRoceChannel::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char** memTags) const
+HcclResult HostCpuRoceChannel::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char** memTags)
 {
-    CHK_PTR_NULL(remoteMem);
-    CHK_PTR_NULL(memNum);
-
-    *remoteMem = nullptr;
+    CHK_PRT_RET(remoteMem == nullptr, HCCL_ERROR("[GetRemoteMem] remoteMem is nullptr"), HCCL_E_PTR);
+    CHK_PRT_RET(memNum == nullptr, HCCL_ERROR("[GetRemoteMem] memNum is nullptr"), HCCL_E_PTR);
+    HCCL_RUN_INFO("GetRemoteMem begin");
     *memNum = 0;
 
     uint32_t totalCount = rmtRmaBuffers_.size();
@@ -386,20 +380,23 @@ HcclResult HostCpuRoceChannel::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNu
         return HCCL_SUCCESS;
     }
 
-    // 释放之前的内存
-    auto remoteMemsPtr_ = std::make_unique<HcclMem[]>(totalCount);
-    CHK_PTR_NULL(remoteMemsPtr_);
-
     for (uint32_t i = 0; i < totalCount; i++) {
         auto& rmtRmaBuffer = rmtRmaBuffers_[i];
-        remoteMemsPtr_[i].type = rmtRmaBuffer->GetMemType();
-        remoteMemsPtr_[i].addr = reinterpret_cast<void *>(rmtRmaBuffer->GetAddr());
-        remoteMemsPtr_[i].size = rmtRmaBuffer->GetSize();
+        std::unique_ptr<HcclMem> hcclMem{};
+        EXECEPTION_CATCH(hcclMem = std::make_unique<HcclMem>(), return HCCL_E_PARA);
+        
+        hcclMem->type = rmtRmaBuffer->GetMemType();
+        hcclMem->addr = reinterpret_cast<void *>(rmtRmaBuffer->GetAddr());
+        hcclMem->size = rmtRmaBuffer->GetSize();
         memTags[i] = const_cast<char*>(rmtRmaBuffer->GetMemTag());
+        remoteMem[i] = hcclMem.get();
+        HCCL_INFO("[HostCpuRoceChannel::%s] rmtBuf[addr[%p], size[%lu]]", 
+            __func__, remoteMem[i]->addr, remoteMem[i]->size);
+        remoteMems.emplace_back(std::move(hcclMem));
     }
 
     *memNum = totalCount;
-    *remoteMem = remoteMemsPtr_.get();
+    HCCL_RUN_INFO("GetRemoteMem end");
     return HCCL_SUCCESS;
 }
 
@@ -433,13 +430,21 @@ std::string HostCpuRoceChannel::Describe() const
     return msg;
 }
 
-HcclResult HostCpuRoceChannel::IbvPostRecv(ibv_qp *const qp, const uint64_t len) const {
+// TODO: 可能需要错开地址
+HcclResult HostCpuRoceChannel::IbvPostRecv() const {
+    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
+    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
+    CHK_PRT_RET(localRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] localRmaBuffer is Empty", __func__),
+                HCCL_E_ROCE_CONNECT);
+    CHK_PRT_RET(rmtRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] rmtRmaBuffers is Empty", __func__),
+                HCCL_E_ROCE_CONNECT);
+
+    // 准备wr
     ibv_recv_wr recvWr {};
     ibv_recv_wr *recvbadWr = nullptr;
-
     ibv_sge recvsgList {};
     recvsgList.addr   = localRmaBuffers_[0]->GetBufferInfo().first; // 本端起始地址
-    recvsgList.length = len;
+    recvsgList.length = MEM_BLOCK_SIZE;
     recvsgList.lkey   = localRmaBuffers_[0]->GetLkey();             // 本端的访问秘钥
     recvWr.wr_id      = 0;
     recvWr.sg_list    = &recvsgList;
@@ -447,8 +452,8 @@ HcclResult HostCpuRoceChannel::IbvPostRecv(ibv_qp *const qp, const uint64_t len)
     recvWr.num_sge    = 1;
 
     HCCL_INFO("[HostCpuRoceChannel::%s] call ibv_post_recv", __func__);
-    HCCL_INFO("qp_state = [%u]", qp->state);
-    int32_t ret = ibv_post_recv(qp, &recvWr, &recvbadWr);
+    HCCL_INFO("qp_state = [%u]", qpInfo[0].qp->state);
+    int32_t ret = ibv_post_recv(qpInfo[0].qp, &recvWr, &recvbadWr);
     CHK_PRT_RET(ret == ENOMEM,
                 HCCL_WARNING("[HostCpuRoceChannel][%s] post recv wqe overflow. ret:%d, "
                              "badWr->wr_id[%llu], badWr->sg_list->addr[%llu]",
@@ -465,7 +470,7 @@ HcclResult HostCpuRoceChannel::IbvPostRecv(ibv_qp *const qp, const uint64_t len)
 }
 
 HcclResult HostCpuRoceChannel::PrepareNotifyWrResource(
-    const uint64_t len, const uint32_t remoteNotifyIdx, ibv_send_wr &notifyRecordWr) const
+    const uint64_t len, const uint32_t remoteNotifyIdx, struct ibv_send_wr &notifyRecordWr) const
 {
     if (remoteNotifyIdx >= remoteDpuNotifyIds_.size()) {
         HCCL_ERROR("[HostCpuRoceChannel::%s] remoteNotifyIdx[%u] out of the range of remoteDpuNotifyIds_[%u].",
@@ -480,11 +485,9 @@ HcclResult HostCpuRoceChannel::PrepareNotifyWrResource(
                 HCCL_E_ROCE_CONNECT);
 
     // 构造send_WR
-    struct ibv_sge sgList {};
-    sgList.addr                 = localRmaBuffers_[0]->GetBufferInfo().first + len; // 本端起始地址
-    sgList.length               = len / 2;                                          // 取的本端长度
-    sgList.lkey                 = localRmaBuffers_[0]->GetLkey();                               // 本端的访问秘钥
-    notifyRecordWr.sg_list      = &sgList;
+    notifyRecordWr.sg_list->addr                 = localRmaBuffers_[0]->GetBufferInfo().first + len; // 本端起始地址
+    notifyRecordWr.sg_list->length               = len / 2;                                          // 取的本端长度
+    notifyRecordWr.sg_list->lkey                 = localRmaBuffers_[0]->GetLkey();                               // 本端的访问秘钥
     notifyRecordWr.opcode       = IBV_WR_RDMA_WRITE_WITH_IMM;
     notifyRecordWr.send_flags   = IBV_SEND_SIGNALED;
     notifyRecordWr.imm_data     = dpuNotifyId;
@@ -498,19 +501,22 @@ HcclResult HostCpuRoceChannel::PrepareNotifyWrResource(
 
 HcclResult HostCpuRoceChannel::NotifyRecord(const uint32_t remoteNotifyIdx) const
 {
-    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
-    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
-    CHK_PRT_RET(localRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] localRmaBuffers_ is Empty", __func__), HCCL_E_ROCE_CONNECT);
     uint64_t bufferBlockSize = localRmaBuffers_[0]->GetBufferInfo().second / 2;
 
     // 补充rq中消耗的rqe
-    CHK_RET(IbvPostRecv(qpInfo[0].qp, bufferBlockSize / 2));
+    // 1. 准备recv_WR
+    CHK_RET(IbvPostRecv());
 
     // 1.构造send_WR
     struct ibv_send_wr  notifyRecordWr {};
     struct ibv_send_wr *sendbadWr = nullptr;
+    struct ibv_sge sgList {};
+    notifyRecordWr.sg_list      = &sgList;
     CHK_RET(PrepareNotifyWrResource(bufferBlockSize, remoteNotifyIdx, notifyRecordWr));
 
+    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
+    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
+
     // 3.调用ibv_post_send
     HCCL_INFO("[HostCpuRoceChannel::%s] call ibv_post_send, qp_state = [%u]", __func__, qpInfo[0].qp->state);
     int32_t ret = ibv_post_send(qpInfo[0].qp, &notifyRecordWr, &sendbadWr);
@@ -521,23 +527,13 @@ HcclResult HostCpuRoceChannel::NotifyRecord(const uint32_t remoteNotifyIdx) cons
     CHK_PRT_RET(ret == ENOMEM,
         HCCL_WARNING("[HostCpuRoceChannel][%s] post send wqe overflow. ret:%d, badWr->wr_id[%llu], "
                      "badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
-            __func__,
-            ret,
-            sendbadWr->wr_id,
-            sendbadWr->sg_list->addr,
-            sendbadWr->wr.rdma.remote_addr,
-            sendbadWr->wr.ud.remote_qpn),
+            __func__, ret, sendbadWr->wr_id, sendbadWr->sg_list->addr, sendbadWr->wr.rdma.remote_addr, sendbadWr->wr.ud.remote_qpn),
         HCCL_E_AGAIN);
 
     CHK_PRT_RET(ret != 0,
         HCCL_ERROR("[HostCpuRoceChannel][%s] ibv_post_send failed. ret:%d, badWr->wr_id[%llu], "
                    "badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
-            __func__,
-            ret,
-            sendbadWr->wr_id,
-            sendbadWr->sg_list->addr,
-            sendbadWr->wr.rdma.remote_addr,
-            sendbadWr->wr.ud.remote_qpn),
+            __func__, ret, sendbadWr->wr_id, sendbadWr->sg_list->addr, sendbadWr->wr.rdma.remote_addr, sendbadWr->wr.ud.remote_qpn),
         HCCL_E_NETWORK);
     HCCL_INFO("[HostCpuRoceChannel::NotifyRecord] NotifyRecord end");
     return HCCL_SUCCESS;
@@ -562,6 +558,14 @@ HcclResult HostCpuRoceChannel::NotifyWait(const uint32_t localNotifyIdx, const u
         HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__);
         return HCCL_E_ROCE_CONNECT;
     }
+    if (qpInfo[0].recvCq == nullptr) {
+        HCCL_INFO("[HostCpuRoceChannel::NotifyWait] recvCq is null");
+        return HCCL_E_INTERNAL;
+    }
+    if (qpInfo[0].recvCq->context == nullptr) {
+        HCCL_INFO("[HostCpuRoceChannel::NotifyWait] recvCq->context is null");
+        return HCCL_E_INTERNAL;
+    }
 
     HCCL_INFO("[HostCpuRoceChannel::NotifyWait] poll recvCq = %p, localNotifyIdx = %u, notifyId = %u.",
         qpInfo[0].recvCq, localNotifyIdx, dpuNotifyId);
@@ -569,20 +573,11 @@ HcclResult HostCpuRoceChannel::NotifyWait(const uint32_t localNotifyIdx, const u
     // 2.轮询rq_cq
     auto startTime = std::chrono::steady_clock::now();
     auto waitTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(timeout));
-
     while (true) {
         HCCL_INFO("[HostCpuRoceChannel::NotifyWait] start to poll cq");
-        if (qpInfo[0].recvCq == nullptr) {
-            HCCL_INFO("[HostCpuRoceChannel::NotifyWait] recvCq is null");
-            return HCCL_E_INTERNAL;
-        }
-        if (qpInfo[0].recvCq->context == nullptr) {
-            HCCL_INFO("[HostCpuRoceChannel::NotifyWait] recvCq->context is null");
-            return HCCL_E_INTERNAL;
-        }
+        
         HCCL_INFO("qp_state = [%u]", qpInfo[0].qp->state);
         auto actualNum = ibv_poll_cq(qpInfo[0].recvCq, 1, &wc);
-        HCCL_INFO("qp_state = [%u]", qpInfo[0].qp->state);
         CHK_PRT_RET(wc.status != IBV_WC_SUCCESS,
             HCCL_ERROR("[HostCpuRoceChannel][%s] ibv_poll_cq return wc.status is [%d].",
             __func__, wc.status), HCCL_E_NETWORK);
@@ -594,8 +589,6 @@ HcclResult HostCpuRoceChannel::NotifyWait(const uint32_t localNotifyIdx, const u
             break;
         }
 
-        SaluSleep(18000);
-
         if ((std::chrono::steady_clock::now() - startTime) >= waitTime) {
             HCCL_ERROR("[HostCpuRoceChannel][%s] call ibv_poll_cq timeout.", __func__);
             return HCCL_E_TIMEOUT;
@@ -606,7 +599,7 @@ HcclResult HostCpuRoceChannel::NotifyWait(const uint32_t localNotifyIdx, const u
 }
 
 HcclResult HostCpuRoceChannel::PrepareWriteWrResource(const void *dst, const void *src, const uint64_t len,
-    const uint32_t remoteNotifyIdx, ibv_send_wr &writeWithNotifyWr) const
+    const uint32_t remoteNotifyIdx, struct ibv_send_wr &writeWithNotifyWr) const
 {
     if (remoteNotifyIdx >= remoteDpuNotifyIds_.size()) {
         HCCL_ERROR("[HostCpuRoceChannel::%s] remoteNotifyIdx[%u] out of the range of remoteDpuNotifyIds_[%u].",
@@ -621,20 +614,18 @@ HcclResult HostCpuRoceChannel::PrepareWriteWrResource(const void *dst, const voi
                 HCCL_E_ROCE_CONNECT);
 
     // 1. 构造WR
-    struct ibv_sge sgList{};
-    sgList.addr = reinterpret_cast<uint64_t>(src); // 本端起始地址
     CHK_PRT_RET(len > UINT32_MAX, HCCL_ERROR("[HostCpuRoceChannel][%s] the len[%llu] exceeds the size of u32.",
         __func__, len), HCCL_E_PARA);
-    sgList.length = len;
-    sgList.lkey = localRmaBuffers_[0]->GetLkey(); // 本端的访问秘钥
 
-    writeWithNotifyWr.sg_list = &sgList;
+    writeWithNotifyWr.sg_list->addr = reinterpret_cast<uint64_t>(src); // 本端起始地址
+    writeWithNotifyWr.sg_list->length = len;
+    writeWithNotifyWr.sg_list->lkey = localRmaBuffers_[0]->GetLkey(); // 本端的访问秘钥
 
     writeWithNotifyWr.opcode              = IBV_WR_RDMA_WRITE_WITH_IMM;
     writeWithNotifyWr.send_flags          = IBV_SEND_SIGNALED;
     writeWithNotifyWr.next                = nullptr;
     writeWithNotifyWr.num_sge             = 1;
-    writeWithNotifyWr.wr_id               = 1;
+    writeWithNotifyWr.wr_id               = 0;
     writeWithNotifyWr.imm_data            = dpuNotifyId;
     writeWithNotifyWr.wr.rdma.rkey        = rmtRmaBuffers_[0]->GetRkey();
     writeWithNotifyWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dst);
@@ -647,39 +638,40 @@ HcclResult HostCpuRoceChannel::WriteWithNotify(
 {
     HCCL_INFO("[HostCpuRoceChannel::WriteWithNotify] WriteWithNotify start");
 
+    CHK_PRT_RET(localRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] localRmaBuffer is Empty", __func__),
+                HCCL_E_ROCE_CONNECT);
+    CHK_PRT_RET(rmtRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] rmtRmaBuffers is Empty", __func__),
+                HCCL_E_ROCE_CONNECT);
+
     std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
     CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
 
     // 补充rq中消耗的rqe
-    CHK_RET(IbvPostRecv(qpInfo[0].qp, len));
+    CHK_RET(IbvPostRecv());
 
     // 1. 构造WR
     struct ibv_send_wr writeWithNotifyWr{};
     struct ibv_send_wr *badWr = nullptr;
+    struct ibv_sge sgList{};
+    writeWithNotifyWr.sg_list = &sgList;
     CHK_RET(PrepareWriteWrResource(dst, src, len, remoteNotifyIdx, writeWithNotifyWr));
 
     // 2. 调用ibv_post_send
     int32_t ret = ibv_post_send(qpInfo[0].qp, &writeWithNotifyWr, &badWr);
+    if (ret != 0 && badWr == nullptr) {
+        HCCL_ERROR("[HostCpuRoceChannel::%s] ibv_post_send failed while badWr is nullptr", __func__);
+        return HCCL_E_INTERNAL;
+    }
     CHK_PRT_RET(ret == ENOMEM,
         HCCL_WARNING("[HostCpuRoceChannel][%s] post send wqe overflow. ret:%d, badWr->wr_id[%llu], "
                      "badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
-            __func__,
-            ret,
-            badWr->wr_id,
-            badWr->sg_list->addr,
-            badWr->wr.rdma.remote_addr,
-            badWr->wr.ud.remote_qpn),
+            __func__, ret, badWr->wr_id, badWr->sg_list->addr, badWr->wr.rdma.remote_addr, badWr->wr.ud.remote_qpn),
         HCCL_E_AGAIN);
 
     CHK_PRT_RET(ret != 0,
         HCCL_ERROR("[HostCpuRoceChannel][%s] ibv_post_send failed. ret:%d, badWr->wr_id[%llu], "
                    "badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
-            __func__,
-            ret,
-            badWr->wr_id,
-            badWr->sg_list->addr,
-            badWr->wr.rdma.remote_addr,
-            badWr->wr.ud.remote_qpn),
+            __func__, ret, badWr->wr_id, badWr->sg_list->addr, badWr->wr.rdma.remote_addr, badWr->wr.ud.remote_qpn),
         HCCL_E_NETWORK);
     HCCL_INFO("[HostCpuRoceChannel::WriteWithNotify] WriteWithNotify end");
     return HCCL_SUCCESS;
@@ -687,79 +679,12 @@ HcclResult HostCpuRoceChannel::WriteWithNotify(
 
 HcclResult HostCpuRoceChannel::Write(void *dst, const void *src, const uint64_t len) const
 {
-    HCCL_INFO("[HostCpuRoceChannelImpl::%s] START. dst[%p], src[%p], len[%llu].", __func__, dst, src, len);
-
-    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
-    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
-
-    CHK_RET(IbvPostRecv(qpInfo[0].qp, len));
-
-    // 1. 构造 WR
-    struct ibv_send_wr writeWr{};
-    struct ibv_send_wr *badWr = nullptr;
-    writeWr.sg_list->addr       = reinterpret_cast<uint64_t>(src); // 源地址
-    writeWr.sg_list->length     = len;
-    writeWr.sg_list->lkey       = localRmaBuffers_[0]->GetLkey(); // LKey
-
-    writeWr.opcode              = IBV_WR_RDMA_WRITE;
-    writeWr.next                = nullptr;
-    writeWr.num_sge             = 1;
-    writeWr.wr_id               = 0;
-    writeWr.wr.rdma.rkey        = rmtRmaBuffers_[0]->GetRkey(); // 远端 RKey
-    writeWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dst); // 远端地址
-
-    // 2. 调用 ibv_post_send
-    s32 ret = ibv_post_send(qpInfo[0].qp, &writeWr, &badWr);
-    CHK_PRT_CONT(ret == ENOMEM,
-        HCCL_WARNING("[CpuRoceChannelImpl::%s] post send wqe overflow. ret:%d, "
-        "badWr->wr_id[%llu], badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
-        __func__, ret, badWr->wr_id, badWr->sg_list->addr, badWr->wr.rdma.remote_addr, badWr->wr.ud.remote_qpn));
-
-    CHK_PRT_CONT(ret != 0,
-        HCCL_ERROR("[CpuRoceChannelImpl::%s] ibv_post_send failed. ret:%d, "
-        "badWr->wr_id[%llu], badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
-        __func__, ret, badWr->wr_id, badWr->sg_list->addr, badWr->wr.rdma.remote_addr, badWr->wr.ud.remote_qpn));
-
-    return HCCL_SUCCESS;
+    return HCCL_E_NOT_SUPPORT;
 }
 
 HcclResult HostCpuRoceChannel::Read(void *dst, const void *src, const uint64_t len) const 
 {
-    HCCL_INFO("[HostCpuRoceChannelImpl::%s] START. dst[%p], src[%p], len[%llu].", __func__, dst, src, len);
-
-    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
-    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
-
-    // 补充rq中消耗的rqe
-    CHK_RET(IbvPostRecv(qpInfo[0].qp, len));
-
-    // 1. 构造 WR
-    struct ibv_send_wr readWr{};
-    struct ibv_send_wr *badWr = nullptr;
-    readWr.sg_list->addr       = reinterpret_cast<uint64_t>(dst);
-    readWr.sg_list->length     = len;
-    readWr.sg_list->lkey       = localRmaBuffers_[0]->GetLkey(); // LKey
-
-    readWr.opcode              = IBV_WR_RDMA_READ;
-    readWr.next                = nullptr;
-    readWr.num_sge             = 1;
-    readWr.wr_id               = 0;
-    readWr.wr.rdma.rkey        = rmtRmaBuffers_[0]->GetRkey(); // 远端 RKey
-    readWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(src); // 远端地址
-
-    // 2. 调用 ibv_post_send
-    s32 ret = ibv_post_send(qpInfo[0].qp, &readWr, &badWr);
-    CHK_PRT_CONT(ret == ENOMEM,
-        HCCL_WARNING("[CpuRoceChannelImpl][%s] post send wqe overflow. ret:%d, "
-        "badWr->wr_id[%llu], badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
-        __func__, ret, badWr->wr_id, badWr->sg_list->addr, badWr->wr.rdma.remote_addr, badWr->wr.ud.remote_qpn));
-
-    CHK_PRT_CONT(ret != 0,
-        HCCL_ERROR("[CpuRoceChannelImpl][%s] ibv_post_send failed. ret:%d, "
-        "badWr->wr_id[%llu], badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
-        __func__, ret, badWr->wr_id, badWr->sg_list->addr, badWr->wr.rdma.remote_addr, badWr->wr.ud.remote_qpn));
-
-    return HCCL_SUCCESS;
+    return HCCL_E_NOT_SUPPORT;
 }
 
 HcclResult HostCpuRoceChannel::ChannelFence() const

```

### src/framework/next/comms/endpoint_pairs/channels/host/host_cpu_roce_channel.h
```diff
@@ -25,14 +25,14 @@ namespace hcomm {
 
 class HostCpuRoceChannel final : public Channel {
 public:
-    MAKE_ENUM(RdmaStatus, INIT, SOCKET_OK, QP_CREATED,  DATE_EXCHANG, QP_MODIFIED, CONN_OK)
+    MAKE_ENUM(RdmaStatus, INIT, SOCKET_OK, QP_CREATED,  DATA_EXCHANGE, QP_MODIFIED, CONN_OK)
 
     HostCpuRoceChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc);
     ~HostCpuRoceChannel();
 
     HcclResult Init() override;
     HcclResult GetNotifyNum(uint32_t *notifyNum) const override;
-    HcclResult GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char** memTags) const override;
+    HcclResult GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char** memTags) override;
     ChannelStatus GetStatus() override;
     HcclResult GetStatus(ChannelStatus &status);
 
@@ -72,37 +72,34 @@ private:
 
     std::vector<Hccl::QpInfo> GetQpInfos() const; // in Connection
 
-    HcclResult IbvPostRecv(ibv_qp *const qp, const uint64_t len) const;
-    HcclResult PrepareNotifyWrResource(const uint64_t len, const uint32_t remoteNotifyIdx, ibv_send_wr &notifyRecordWr) const;
+    HcclResult IbvPostRecv() const;
+    HcclResult PrepareNotifyWrResource(const uint64_t len, const uint32_t remoteNotifyIdx, struct ibv_send_wr &notifyRecordWr) const;
     HcclResult PrepareWriteWrResource(const void *dst, const void *src, const uint64_t len, const uint32_t remoteNotifyIdx,
-                                      ibv_send_wr &writeWithNotifyWr) const;
+                                      struct ibv_send_wr &writeWithNotifyWr) const;
 
-    // --------------------- 入参 ---------------------
-    EndpointHandle                                              endpointHandle_;
-    HcommChannelDesc                                            channelDesc_;
+    // 入参
+    EndpointHandle endpointHandle_;
+    HcommChannelDesc channelDesc_;
 
-    // --------------------- 转换参数 ---------------------
-    EndpointDesc                                                localEp_;
-    EndpointDesc                                                remoteEp_;
-    uint32_t                                                    notifyNum_{0};
-    std::vector<std::shared_ptr<Hccl::Buffer>>                  bufs_;
+    // 转换参数
+    EndpointDesc localEp_;
+    EndpointDesc remoteEp_;
+    uint32_t notifyNum_{0};
+    Hccl::Socket *socket_{nullptr};
+    RdmaHandle rdmaHandle_{nullptr};
 
-    std::unique_ptr<Hccl::Socket>                               socket_{nullptr};
-    RdmaHandle                                                  rdmaHandle_{nullptr};
-    std::vector<std::unique_ptr<HostRdmaConnection>>            connections_{};
-    std::vector<std::unique_ptr<Hccl::LocalRdmaRmaBuffer>>      localRmaBuffers_{};
-    std::vector<uint32_t>                                       localDpuNotifyIds_{};
-    
-    uint32_t                                                    bufferNum_{0};
-    uint32_t                                                    connNum_{0};
-
-    // Hccl::BaseMemTransport::Attribution                         attr_;
-    ChannelStatus                                               channelStatus_{ChannelStatus::INIT};
-    RdmaStatus                                                  rdmaStatus_{RdmaStatus::INIT}; 
-                                                   
-    std::vector<uint32_t>                                       remoteDpuNotifyIds_;
-    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>>     rmtRmaBuffers_{};
-    ExchangeRdmaConnDto                                         rmtConnDto_;
+    std::vector<std::unique_ptr<HostRdmaConnection>> connections_{};
+    std::vector<Hccl::LocalRdmaRmaBuffer *> localRmaBuffers_{};
+    std::vector<uint32_t> localDpuNotifyIds_{};
+    uint32_t bufferNum_{0};
+    uint32_t connNum_{0};
+    // Hccl::BaseMemTransport::Attribution attr_;
+    ChannelStatus channelStatus_{ChannelStatus::INIT};
+    RdmaStatus rdmaStatus_{RdmaStatus::INIT};
+    std::vector<uint32_t> remoteDpuNotifyIds_;
+    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> rmtRmaBuffers_{};
+    ExchangeRdmaConnDto rmtConnDto_;
+    std::vector<std::unique_ptr<HcclMem>> remoteMems{};
 
     std::mutex cq_mutex;
 };

```

### src/framework/next/comms/endpoint_pairs/channels/host/host_rdma_connection.cc
```diff
@@ -9,12 +9,7 @@
  */
 #include "host_rdma_connection.h"
 #include "dtype_common.h"
-
-// #include "invalid_params_exception.h"
 #include "orion_adapter_rts.h"
-
-
-
 #include "exchange_rdma_conn_dto.h"
 #include "hccp.h"
 #include "sal.h"
@@ -57,7 +52,7 @@ HcclResult HostRdmaConnection::Init()
 
 HostRdmaConnection::~HostRdmaConnection()
 {
-    if (rdmaConnStatus_ == RdmaConnStatus::CLOSED) {
+    if (rdmaConnStatus_ == RdmaConnStatus::CLOSED || rdmaConnStatus_ == RdmaConnStatus::INIT) {
         return;
     }
     HcclResult ret = DestroyQp();
@@ -105,8 +100,11 @@ HcclResult HostRdmaConnection::CreateQp()
 
 HcclResult HostRdmaConnection::DestroyQp()
 {
+    if (rdmaConnStatus_ == RdmaConnStatus::CLOSED || rdmaConnStatus_ == RdmaConnStatus::INIT) {
+        return HCCL_SUCCESS;
+    }
+
     CHK_RET(Hccl::HrtRaDestroyQpWithCq(qpInfo_, isHdcMode_));
-    qpInfo_ = Hccl::QpInfo();
 
     s32 ret = RaDestroyCompChannel(qpInfo_.rdmaHandle, sendCompChannel_);
     CHK_PRT_RET(ret != 0,
@@ -121,6 +119,7 @@ HcclResult HostRdmaConnection::DestroyQp()
                            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.rdmaHandle, &recvCompChannel_),
                 HCCL_E_NETWORK);
 
+    qpInfo_ = Hccl::QpInfo();
     rdmaConnStatus_ = RdmaConnStatus::CLOSED;
     return HCCL_SUCCESS;
 }

```

### src/framework/next/comms/endpoints/reged_mems/roce_mem.cc
```diff
@@ -28,6 +28,8 @@ HcclResult RoceRegedMemMgr::RegisterMemory(HcommMem mem, const char *memTag, voi
 {
     HCCL_INFO("[%s] Begin", __FUNCTION__);
     CHK_PTR_NULL(this->localRdmaRmaBufferMgr_);
+    CHK_PTR_NULL(memHandle);
+    CHK_PTR_NULL(memTag);
 
     // 构造LocalRdmaRmaBuffer
     std::shared_ptr<Hccl::Buffer> localBufferPtr = nullptr;
@@ -76,6 +78,7 @@ HcclResult RoceRegedMemMgr::UnregisterMemory(void* memHandle)
 {
     HCCL_INFO("[%s] Begin", __FUNCTION__);
     CHK_PTR_NULL(this->localRdmaRmaBufferMgr_);
+    CHK_PTR_NULL(memHandle);
 
     Hccl::LocalRdmaRmaBuffer* buffer = static_cast<Hccl::LocalRdmaRmaBuffer*>(memHandle);
     auto bufferInfo = buffer->GetBufferInfo();

```

### src/framework/op_base/src/op_base.cc
```diff
@@ -2979,7 +2979,7 @@ HcclResult HcclCommDestroy(HcclComm comm)
                 return HCCL_SUCCESS;
             }
             hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
-            CHK_RET(HcclCommDestroyV2(hcclComm->GetCommunicatorV2()));
+            void* commV2 = hcclComm->GetCommunicatorV2();
             string group;
             group = hcclComm->GetIdentifier();
             HcclOpInfoCtx& opBaseHcom = GetHcclOpInfoCtx();
@@ -2991,6 +2991,7 @@ HcclResult HcclCommDestroy(HcclComm comm)
                 HCCL_ERROR("[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(), deviceLogicId);
                 return HCCL_E_PARA;
             }
+            CHK_RET(HcclCommDestroyV2(commV2));
             return HCCL_SUCCESS;
         }());
 #endif

```

### src/framework/op_base/src/op_base_mc2.cc
```diff
@@ -234,7 +234,12 @@ HcclResult HcclDevMemAcquire(HcclComm comm, const char *memTag, uint64_t *size,
     CHK_PTR_NULL(comm);
     CHK_PTR_NULL(size);
     CHK_PTR_NULL(addr);
-    HCCLV2_FUNC_RUN(HcclDevMemAcquireV2(comm, memTag, size, addr, newCreated));
+    const char *indOp = getenv("HCCL_INDEPENDENT_OP");
+    if (indOp == nullptr || strcmp(indOp, "") == 0) {
+        HCCLV2_FUNC_RUN(HcclDevMemAcquireV2(comm, memTag, size, addr, newCreated));
+    }
+    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
+    HCCLV2_FUNC_RUN(HcclDevMemAcquireV2(hcclComm->GetCommunicatorV2(), memTag, size, addr, newCreated));
     return HCCL_SUCCESS;
 }
 

```

### src/legacy/framework/communicator/communicator_impl.cc
```diff
@@ -14,6 +14,7 @@
 #include "orion_adapter_rts.h"
 #include "hccl_exception.h"
 #include "null_ptr_exception.h"
+#include "runtime_api_exception.h"
 #include "exception_util.h"
 #include "hccp_hdc_manager.h"
 #include "hccp_peer_manager.h"
@@ -153,6 +154,7 @@ void CommunicatorImpl::InitDpuKernel() {
     }
     HCCL_INFO("[InitDpuKernel]all FlushHandle init success.");
     /* kernel Launch */
+    CHK_RET_THROW(RuntimeApiException, "InitAndLaunchDpuKernel Failed", InitAndLaunchDpuKernel());
 }
 
 std::unordered_set<IpAddress> CommunicatorImpl::GetHostIpFromRankGraph()
@@ -2182,7 +2184,7 @@ CommunicatorImpl::~CommunicatorImpl()
 HcclResult CommunicatorImpl::DestroyDpuKernelResource()
 {
     // 终止Dpu Kernel的TaskRun
-    if (!IsNeedDpu()) {
+    if (!isDpuKernelLaunched) {
         return HCCL_SUCCESS;
     }
 
@@ -2954,10 +2956,6 @@ HcclResult CommunicatorImpl::LaunchDpuKernel(aclrtFuncHandle &funcHandle)
 
 HcclResult CommunicatorImpl::InitAndLaunchDpuKernel()
 {
-    if (!IsNeedDpu()) {
-        return HCCL_SUCCESS;
-    }
-
     HCCL_INFO("[CommunicatorImpl::%s] Start to Launch Dpu Kernel", __func__);
     // 设置XPU
     HCCL_INFO("[CommunicatorImpl::%s] Switch to Dpu Ctx", __func__);
@@ -2989,6 +2987,7 @@ HcclResult CommunicatorImpl::InitAndLaunchDpuKernel()
     }
 
     HCCL_INFO("[CommunicatorImpl::%s] Launch Dpu Kernel End", __func__);
+    isDpuKernelLaunched = true;
     return HCCL_SUCCESS;
 }
 

```

### src/legacy/framework/communicator/communicator_impl.h
```diff
@@ -354,8 +354,6 @@ public:
     }
 
     HcclResult CreateBarrierMemory(void *&sendBuf, void *&recvBuf, uint64_t count);
-    // DPU
-    HcclResult InitAndLaunchDpuKernel();
 
     HcclResult HcomSelectAlg(const CollOpParams &opParams, int32_t aivCoreLimit, bool &ifAiv, std::string &algName);
     HcclResult CalcBlockDim(const CollOpParams &opParams, int32_t aivCoreLimit, std::string &algName,
@@ -431,6 +429,7 @@ private:
     bool isSuspended = false;
     bool isCleaned = false;
     bool isAicpuKernelLaunched = false;
+    bool isDpuKernelLaunched = false;
     bool isWorldGroup = false;
     bool aivClearEnable = false;
 
@@ -516,6 +515,7 @@ private:
     HcclResult PrepareDpuKernelResource(aclrtFuncHandle &funcHandle);
     HcclResult DestroyDpuKernelResource();
     HcclResult WaitDpuKernelThreadTerminate();
+    HcclResult InitAndLaunchDpuKernel();
 
     HcclResult Init(const CommParams &commParams, std::unique_ptr<RankGraph> &inputRankGraph, DevId inputDevLogicId);
     HcclResult Init(const CommParams &commParams, std::unique_ptr<RankGraph> &inputRankGraph,

```

### src/legacy/framework/communicator/hccl_communicator.cc
```diff
@@ -435,11 +435,6 @@ HcclResult HcclCommunicator::GetKFCWorkSpace(const char *memTag, uint64_t *size,
     }
     return HcclResult::HCCL_SUCCESS;
 }
-// Dpu Kernel Launch
-HcclResult HcclCommunicator::LaunchDpuKernel()
-{
-    return pimpl->InitAndLaunchDpuKernel();
-}
 
 HcclResult HcclCommunicator::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t **ranks, uint32_t *rankNum)
 {

```

### src/legacy/framework/entrance/op_base/op_base_v2.cc
```diff
@@ -386,19 +386,9 @@ HcclResult HcclCommInitClusterInfoConfigV2(
     HCCL_RUN_INFO("[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], commId[%s].",
         __func__, DURATION_US(TIME_NOW() - startut), clusterInfo, rank, config->hcclCommName);
 
-    // 拉起KFC kernel
-    CHK_RET(HostKFCServerInit(*comm));
-
     return HCCL_SUCCESS;
 }
 
-HcclResult HostKFCServerInit(HcclComm comm)
-{
-    CHK_PTR_NULL(comm);
-    Hccl::HcclCommunicator *communicator = static_cast<Hccl::HcclCommunicator *>(comm);
-    return communicator->LaunchDpuKernel();
-}
-
 HcclResult HcclTaskRegisterV2(HcclComm comm, const char *msgTag, Callback cb)
 {
     HCCL_RUN_INFO("[HcclTaskRegisterV2] start to register task");

```

### src/legacy/framework/entrance/op_base/op_base_v2.h
```diff
@@ -168,7 +168,6 @@ HcclResult HcclGetRanksByTopoInstV2(HcclComm comm, uint32_t netLayer, uint32_t t
                                   uint32_t *rankNum);
 HcclResult HcommFlushV2();
 HcclResult HcclGetCclBuffer(HcclComm comm, uintptr_t &cclBufferAddr, size_t &cclBufferSize, HcclMemType &cclBufferMemType);
-HcclResult HostKFCServerInit(HcclComm comm);
 
 HcclResult HcclRankGraphGetEndpointNumV2(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t *num);
 

```

### src/legacy/include/hccl_communicator.h
```diff
@@ -108,8 +108,6 @@ public:
  
     HcclResult GetRankGraphV2(void *&rankGraph);
     HcclResult HcclGetCclBuffer(uintptr_t &cclBufferAddr, size_t &cclBufferSize, HcclMemType &cclBufferMemType);
-    // Dpu Kernel Launch
-    HcclResult LaunchDpuKernel();
     HcclResult GetConfigInCCLbufferSize(uint64_t *cclBufSize);
     HcclResult GetNetLayers(uint32_t **netLayers, uint32_t *netLayerNum);
     HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t *rankNum);

```

### src/legacy/unified_platform/resource/buffer/exchange_rdma_buffer_dto.h
```diff
@@ -21,29 +21,31 @@ public:
     {
     }
 
-    ExchangeRdmaBufferDto(u64 addr, u64 size, u32 rkey) : addr(addr), size(size), rkey(rkey)
+    ExchangeRdmaBufferDto(u64 addr, u64 size, u32 rkey, const char *memTag)
+        : addr(addr), size(size), rkey(rkey), memTag(memTag)
     {
     }
 
     void Serialize(Hccl::BinaryStream &stream) override
     {
-        stream << addr << size << rkey;
+        stream << addr << size << rkey << memTag;
     }
 
     void Deserialize(Hccl::BinaryStream &stream) override
     {
-        stream >> addr >> size >> rkey;
+        stream >> addr >> size >> rkey >> memTag;
     }
 
     std::string Describe() const override
     {
-        return StringFormat("ExchangeRdmaBufferDto[addr=0x%llx, size=0x%llx, rkey=%lu]", addr, size, rkey);
+        return StringFormat("ExchangeRdmaBufferDto[addr=0x%llx, size=0x%llx, rkey=%lu, memTag=%s]", addr, size, rkey,
+                            memTag.c_str());
     }
 
     u64 addr{0};
     u32 size{0};
     u32 rkey{0};
-    u8  key[RDMA_MEM_KEY_MAX_LEN]{0};
+    std::string memTag{""};
 };
 
 } // namespace Hccl

```

### src/legacy/unified_platform/resource/buffer/local_rdma_rma_buffer.cc
```diff
@@ -68,7 +68,7 @@ string LocalRdmaRmaBuffer::Describe() const
 std::unique_ptr<Serializable> LocalRdmaRmaBuffer::GetExchangeDto()
 {
     std::unique_ptr<ExchangeRdmaBufferDto> dto
-        = make_unique<ExchangeRdmaBufferDto>(buf->GetAddr(), buf->GetSize(), this->rkey);
+        = make_unique<ExchangeRdmaBufferDto>(buf->GetAddr(), buf->GetSize(), this->rkey, buf->GetMemTag());
     return std::unique_ptr<Serializable>(dto.release());
 }
 

```

### src/legacy/unified_platform/resource/buffer/remote_rma_buffer.cc
```diff
@@ -93,8 +93,8 @@ RemoteRdmaRmaBuffer::RemoteRdmaRmaBuffer(RdmaHandle rdmaHandle, const Serializab
     addr = dto.addr;
     size = dto.size;
     rkey = dto.rkey;
-    memTag = "HcclBuffer"; // TODO： 临时处理
-    HCCL_INFO("[RemoteRdmaRmaBuffer]addr = %llu; size = %u; rkey = %u", addr, size, rkey);
+    memTag = dto.memTag;
+    HCCL_INFO("[RemoteRdmaRmaBuffer]addr = %llu; size = %u; rkey = %u, memTag = %s", addr, size, rkey, memTag.c_str());
 }
 
 RemoteRdmaRmaBuffer::~RemoteRdmaRmaBuffer()

```

### src/platform/common/adapter/adapter_rts.cc
```diff
@@ -488,7 +488,7 @@ HcclResult __hrtGetDeviceType(DevType &devType)
 #endif
     //  根据芯片版本号获取芯片类型
     HCCL_DEBUG("[hrtGetDeviceType]socName = %s.", socName.c_str());
-    if (socName.find("Ascend950") != std::string::npos) {
+    if (socName.find("Ascend910_958b") != std::string::npos) {
         devType = DevType::DEV_TYPE_910_95;
         g_deviceType = devType;
         HCCL_DEBUG("[hrtGetDeviceType]DeviceType = %d.", static_cast<s32>(g_deviceType));

```

### test/ut/stub/llt_next_orion_stub.cc
```diff
@@ -144,7 +144,7 @@ HcclResult HostCpuRoceChannel::Init()
     return HCCL_SUCCESS;
 }
 
-HcclResult HostCpuRoceChannel::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char** memTags) const
+HcclResult HostCpuRoceChannel::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char** memTags)
 {
     return HCCL_SUCCESS;
 }

```
