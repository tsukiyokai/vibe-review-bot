# PR #1082: rankGraph modify

- 作者: xumochi
- 分支: master -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/1082
- 描述: 增加netlayer校验，添加打印日志，适配collcomm下三个endpoint接口

## 变更文件 (11 个, 其中 C/C++ 文件 10 个)

- [modified] src/framework/communicator/impl/independent_op/hccl_independent_rank_graph.cc (+131, -26) *
- [modified] src/framework/communicator/impl/independent_op/rank_graph/rank_graph_v2.cc (+17, -0) *
- [modified] src/framework/communicator/impl/independent_op/rank_graph/rank_graph_v2.h (+4, -0) *
- [modified] src/hccl_next/framework/topo/new_topo_builder/rank_graph/rank_graph.cc (+10, -1) *
- [modified] src/orion/framework/topo/new_topo_builder/rank_graph/rank_graph.cc (+11, -1) *
- [modified] src/orion/interface/rank_graph_interface.cc (+103, -17) *
- [modified] src/orion/interface/rank_graph_interface.h (+4, -0) *
- [added] test/ut/framework/communicator/ut_HcclRankGraph_API_test.cc (+348, -0) *
- [modified] test/ut/stub/CMakeLists.txt (+1, -0)
- [added] test/ut/stub/llt_hccl_stub_rank_graph.cc (+81, -0) *
- [added] test/ut/stub/llt_hccl_stub_rank_graph.h (+33, -0) *

## Diff 内容

### src/framework/communicator/impl/independent_op/hccl_independent_rank_graph.cc
```diff
@@ -74,7 +74,12 @@ HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcR
             RankGraph* rankGraph = collComm->GetRankGraph();
             CHK_PTR_NULL(rankGraph);
             ret = rankGraph->GetLinks(netLayer, srcRank, dstRank, links, linkNum);
-            return ret;
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success, linkNum [%u]", __func__, *linkNum);
+            return HCCL_SUCCESS;
         }());
     hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);  
     HCCL_RUN_INFO("Entry-%s: comm[%s], netLayer%u], srcRank[%u], dstRank[%u]", __func__,
@@ -90,29 +95,32 @@ HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcR
     return HCCL_SUCCESS;
 }
 
-HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
+HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
 {
     CHK_PTR_NULL(comm);
     CHK_PTR_NULL(netLayers);
     CHK_PTR_NULL(netLayerNum);
-    
     HcclResult ret = HCCL_SUCCESS;
-    HCCLV2_FUNC_RUN(
-    [&]() -> HcclResult {
-        const char *indOp = getenv("HCCL_INDEPENDENT_OP");
+    HCCLV2_FUNC_RUN([&]() -> HcclResult {
+        const char* indOp = getenv("HCCL_INDEPENDENT_OP");
         if (indOp == nullptr || strcmp(indOp, "") == 0) {
             CHK_RET(HcclGetNetLayersV2(comm, netLayers, netLayerNum));
             return HCCL_SUCCESS;
         }
-        hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
+        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
         CollComm* collComm = hcclComm->GetCollComm();
         CHK_PTR_NULL(collComm);
         RankGraph* rankGraph = collComm->GetRankGraph();
         CHK_PTR_NULL(rankGraph);
         ret = rankGraph->GetNetLayers(netLayers, netLayerNum);
-        return ret;
+        if (ret != HCCL_SUCCESS) {
+            HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+            return ret;
+        }
+        HCCL_RUN_INFO("[%s] success, netLayerNum size[%u]", __func__, *netLayerNum);
+        return HCCL_SUCCESS;
     }());
-    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
+    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
     ret = hcclComm->GetNetLayers(netLayers, netLayerNum);
     if (ret != HCCL_SUCCESS) {
         HCCL_ERROR("[%s] Failed to GetCommNetLayers ret[%d]", __func__, ret);
@@ -142,7 +150,12 @@ HcclResult HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, Com
             RankGraph* rankGraph = collComm->GetRankGraph();
             CHK_PTR_NULL(rankGraph);
             ret = rankGraph->GetInstTopoTypeByNetLayer(netLayer, topoType);
-            return ret;
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success, topoType [%d]", __func__, *topoType);
+            return HCCL_SUCCESS;
         }());
     hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
     ret = hcclComm->GetInstTopoTypeByNetLayer(netLayer, topoType);
@@ -173,7 +186,12 @@ HcclResult HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uin
         RankGraph* rankGraph = collComm->GetRankGraph();
         CHK_PTR_NULL(rankGraph);
         ret = rankGraph->GetInstSizeByNetLayer(netLayer, rankNum);
-        return ret;
+        if (ret != HCCL_SUCCESS) {
+            HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+            return ret;
+        }
+        HCCL_RUN_INFO("[%s] success, rankNum [%u]", __func__, *rankNum);
+        return HCCL_SUCCESS;
     }());
     hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
     ret = hcclComm->GetInstSizeByNetLayer(netLayer, rankNum);
@@ -205,7 +223,12 @@ HcclResult HcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32
             RankGraph* rankGraph = collComm->GetRankGraph();
             CHK_PTR_NULL(rankGraph);
             ret = rankGraph->GetInstRanksByNetLayer(netLayer, ranks, rankNum);
-            return ret;
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success, rankNum [%u]", __func__, *rankNum);
+            return HCCL_SUCCESS;
         }());
     hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
     ret = hcclComm->GetInstRanksByNetLayer(netLayer, ranks, rankNum);
@@ -236,7 +259,12 @@ HcclResult HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer,
             RankGraph* rankGraph = collComm->GetRankGraph();
             CHK_PTR_NULL(rankGraph);
             ret = rankGraph->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
-            return ret;
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success, listSize [%u]", __func__, *listSize);
+            return HCCL_SUCCESS;
         }());
     hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
     ret = hcclComm->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
@@ -268,9 +296,14 @@ HcclResult HcclGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t **
             CHK_PTR_NULL(rankGraph);
             RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
             ret = rankGraphV2->GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum);
-            return ret;
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success, topoInstNum [%u]", __func__, *topoInstNum);
+            return HCCL_SUCCESS;
         }());
-    return HCCL_SUCCESS;
+    return HCCL_E_NOT_SUPPORT;
 }
 
 HcclResult HcclGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo *topoType)
@@ -292,9 +325,14 @@ HcclResult HcclGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId
             CHK_PTR_NULL(rankGraph);
             RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
             ret = rankGraphV2->GetTopoType(netLayer, topoInstId, topoType);
-            return ret;
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success, topoType [%d]", __func__, *topoType);
+            return HCCL_SUCCESS;
         }());
-    return HCCL_SUCCESS;
+    return HCCL_E_NOT_SUPPORT;
 }
 
 HcclResult HcclGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t **ranks, uint32_t *rankNum)
@@ -317,17 +355,43 @@ HcclResult HcclGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t top
             CHK_PTR_NULL(rankGraph);
             RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
             ret = rankGraphV2->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
-            return ret;
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success, rankNum [%u]", __func__, *rankNum);
+            return HCCL_SUCCESS;
         }());
-    return HCCL_SUCCESS;
+    return HCCL_E_NOT_SUPPORT;
 }
 
 HcclResult HcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t *num)
 {
     CHK_PTR_NULL(comm);
     CHK_PTR_NULL(num);
-    HCCLV2_FUNC_RUN(HcclRankGraphGetEndpointNumV2(comm, layer, topoInstId, num));
-    return HCCL_SUCCESS;
+    HcclResult ret = HCCL_SUCCESS;
+    HCCLV2_FUNC_RUN(
+        [&]() -> HcclResult {
+            const char *indOp = getenv("HCCL_INDEPENDENT_OP");
+            if (indOp == nullptr || strcmp(indOp, "") == 0) {
+                CHK_RET(HcclRankGraphGetEndpointNumV2(comm, layer, topoInstId, num));
+                return HCCL_SUCCESS;
+            }
+            hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
+            CollComm* collComm = hcclComm->GetCollComm();
+            CHK_PTR_NULL(collComm);
+            RankGraph* rankGraph = collComm->GetRankGraph();
+            CHK_PTR_NULL(rankGraph);
+            RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
+            ret = rankGraphV2->GetEndpointNum(layer, topoInstId, num);
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success, num [%u]", __func__, *num);
+            return HCCL_SUCCESS;
+        }());
+    return HCCL_E_NOT_SUPPORT;
 }
 
 HcclResult HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t *descNum, EndpointDesc *endpointDesc)
@@ -335,8 +399,29 @@ HcclResult HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t
     CHK_PTR_NULL(comm);
     CHK_PTR_NULL(descNum);
     CHK_PTR_NULL(endpointDesc);
-    HCCLV2_FUNC_RUN(HcclRankGraphGetEndpointDescV2(comm, layer, topoInstId, descNum, endpointDesc));
-    return HCCL_SUCCESS;
+    HcclResult ret = HCCL_SUCCESS;
+    HCCLV2_FUNC_RUN(
+        [&]() -> HcclResult {
+            const char *indOp = getenv("HCCL_INDEPENDENT_OP");
+            if (indOp == nullptr || strcmp(indOp, "") == 0) {
+                CHK_RET(HcclRankGraphGetEndpointDescV2(comm, layer, topoInstId, descNum, endpointDesc));
+                return HCCL_SUCCESS;
+            }
+            hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
+            CollComm* collComm = hcclComm->GetCollComm();
+            CHK_PTR_NULL(collComm);
+            RankGraph* rankGraph = collComm->GetRankGraph();
+            CHK_PTR_NULL(rankGraph);
+            RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
+            ret = rankGraphV2->GetEndpointDesc(layer, topoInstId, descNum, endpointDesc);
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success", __func__);
+            return HCCL_SUCCESS;
+        }());
+    return HCCL_E_NOT_SUPPORT;
 }
 
 HcclResult HcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc *endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void *info)
@@ -344,8 +429,29 @@ HcclResult HcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const En
     CHK_PTR_NULL(comm);
     CHK_PTR_NULL(endpointDesc);
     CHK_PTR_NULL(info);
-    HCCLV2_FUNC_RUN(HcclRankGraphGetEndpointInfoV2(comm, rankId, endpointDesc, endpointAttr, infoLen, info));
-    return HCCL_SUCCESS;
+    HcclResult ret = HCCL_SUCCESS;
+    HCCLV2_FUNC_RUN(
+        [&]() -> HcclResult {
+            const char *indOp = getenv("HCCL_INDEPENDENT_OP");
+            if (indOp == nullptr || strcmp(indOp, "") == 0) {
+                CHK_RET(HcclRankGraphGetEndpointInfoV2(comm, rankId, endpointDesc, endpointAttr, infoLen, info));
+                return HCCL_SUCCESS;
+            }
+            hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
+            CollComm* collComm = hcclComm->GetCollComm();
+            CHK_PTR_NULL(collComm);
+            RankGraph* rankGraph = collComm->GetRankGraph();
+            CHK_PTR_NULL(rankGraph);
+            RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
+            ret = rankGraphV2->GetEndpointInfo(rankId, endpointDesc, endpointAttr, infoLen, info);
+            if (ret != HCCL_SUCCESS) {
+                HCCL_ERROR("[%s] Failed to ret[%d]", __func__, ret);
+                return ret;
+            }
+            HCCL_RUN_INFO("[%s] success", __func__);
+            return HCCL_SUCCESS;
+        }());
+    return HCCL_E_NOT_SUPPORT;
 }
 
 HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode *mode)
@@ -366,7 +472,6 @@ HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode *mode)
 HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
 {
     // 入参合法性校验
-    // TODO: 老代码呢
     CHK_PTR_NULL(comm);
     CHK_PTR_NULL(rankSize);
     HCCLV2_FUNC_RUN(

```

### src/framework/communicator/impl/independent_op/rank_graph/rank_graph_v2.cc
```diff
@@ -83,4 +83,21 @@ HcclResult RankGraphV2::GetRanksByTopoInst(const uint32_t netLayer, const uint32
     return pImpl->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
 }
 
+HcclResult RankGraphV2::GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t *num)
+{
+    return pImpl->GetEndpointNum(netLayer, topoInstId, num);
+}
+
+HcclResult RankGraphV2::GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t *descNum,
+                                        EndpointDesc *endpointDesc)
+{
+    return pImpl->GetEndpointDesc(netLayer, topoInstId, descNum, endpointDesc);
+}
+
+HcclResult RankGraphV2::GetEndpointInfo(uint32_t rankId, const EndpointDesc *endPointDesc, EndpointAttr endpointAttr,
+                                        uint32_t infoLen, void *info)
+{
+    return pImpl->GetEndpointInfo(rankId, endPointDesc, endpointAttr, infoLen, info);
+}
+
 };  // namespace hccl

```

### src/framework/communicator/impl/independent_op/rank_graph/rank_graph_v2.h
```diff
@@ -31,6 +31,10 @@ public:
     HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
     HcclResult GetTopoType(const uint32_t netLayer, const uint32_t topoInstId, CommTopo* topoType);
     HcclResult GetRanksByTopoInst(const uint32_t netLayer, const uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
+    HcclResult GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t *num);
+    HcclResult GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t *descNum, EndpointDesc *endpointDesc);
+    HcclResult GetEndpointInfo(uint32_t rankId, const EndpointDesc *endPointDesc, EndpointAttr endpointAttr,
+                               uint32_t infoLen, void *info);
 
 private:
     std::unique_ptr<Hccl::IRankGraph> pImpl;

```

### src/hccl_next/framework/topo/new_topo_builder/rank_graph/rank_graph.cc
```diff
@@ -362,6 +362,14 @@ HcclResult GetCommAddr(CommAddr &commAddr, const IpAddress &ipAddr)
     return HCCL_SUCCESS;
 }
 
+static EndpointLocType AddrPositionToEndpointLoc(AddrPosition pos) {
+    switch (pos) {
+        case AddrPosition::HOST:    return ENDPOINT_LOC_TYPE_HOST;
+        case AddrPosition::DEVICE:  return ENDPOINT_LOC_TYPE_DEVICE;
+        default: return ENDPOINT_LOC_TYPE_RESERVED;
+    }
+}
+
 HcclResult RankGraph::GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum,
                                       EndpointDesc* endpointDesc)
 {
@@ -388,7 +396,8 @@ HcclResult RankGraph::GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint3
                 auto it = protocolMap.find(protocol);
                 CommProtocol commProtocol = (it != protocolMap.end()) ? it->second : COMM_PROTOCOL_RESERVED;
                 endpointDesc[count].protocol = commProtocol;
-                endpointDesc[count].loc.locType = static_cast<EndpointLocType>(static_cast<int>(iface->GetPos()));
+                endpointDesc[count].loc.locType = AddrPositionToEndpointLoc(iface->GetPos());
+                HCCL_INFO("[RankGraph::GetEndpointDesc] local type is %d", endpointDesc[count].loc.locType);
                 peer->SetEndpointToIface(endpointDesc[count].commAddr, endpointDesc[count].protocol, iface);
                 count++;
             }

```

### src/orion/framework/topo/new_topo_builder/rank_graph/rank_graph.cc
```diff
@@ -362,6 +362,15 @@ HcclResult GetCommAddr(CommAddr &commAddr, const IpAddress &ipAddr)
     return HCCL_SUCCESS;
 }
 
+static EndpointLocType AddrPositionToEndpointLoc(AddrPosition pos) {
+    switch (pos) {
+        case AddrPosition::HOST:    return ENDPOINT_LOC_TYPE_HOST;
+        case AddrPosition::DEVICE:  return ENDPOINT_LOC_TYPE_DEVICE;
+        default: return ENDPOINT_LOC_TYPE_RESERVED;
+    }
+}
+
+
 HcclResult RankGraph::GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum,
                                       EndpointDesc* endpointDesc)
 {
@@ -388,7 +397,8 @@ HcclResult RankGraph::GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint3
                 auto it = protocolMap.find(protocol);
                 CommProtocol commProtocol = (it != protocolMap.end()) ? it->second : COMM_PROTOCOL_RESERVED;
                 endpointDesc[count].protocol = commProtocol;
-                endpointDesc[count].loc.locType = static_cast<EndpointLocType>(static_cast<int>(iface->GetPos()));
+                endpointDesc[count].loc.locType = AddrPositionToEndpointLoc(iface->GetPos());
+                HCCL_INFO("[RankGraph::GetEndpointDesc] local type is %d", endpointDesc[count].loc.locType);
                 peer->SetEndpointToIface(endpointDesc[count].commAddr, endpointDesc[count].protocol, iface);
                 count++;
             }

```

### src/orion/interface/rank_graph_interface.cc
```diff
@@ -63,6 +63,12 @@ namespace Hccl {
         HCCL_RUN_INFO("Entry-IRankGraph::GetInstTopoTypeByNetLayer");
         CHK_PTR_NULL(rankGraphPtr_);
         RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetInstTopoTypeByNetLayer] netLayer[%u] is invalid",netLayer);
+            return HCCL_E_PARA;
+        }
         auto type = rankGraph->GetNetType(netLayer);
         static const std::unordered_map<NetType, CommTopo> netTypeMap = {
                 {NetType::CLOS, CommTopo::COMM_TOPO_CLOS},
@@ -85,6 +91,12 @@ namespace Hccl {
         HCCL_RUN_INFO("Entry-IRankGraph::GetInstSizeByNetLayer");
         CHK_PTR_NULL(rankGraphPtr_);
         RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetInstSizeByNetLayer] netLayer[%u] is invalid",netLayer);
+            return HCCL_E_PARA;
+        }
         u32 num = rankGraph->GetLocalInstSize(netLayer);
         *rankNum = static_cast<uint32_t>(num);
         return HCCL_SUCCESS;
@@ -95,6 +107,12 @@ namespace Hccl {
         HCCL_RUN_INFO("Entry-IRankGraph::GetInstRanksByNetLayer");
         CHK_PTR_NULL(rankGraphPtr_);
         RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetInstRanksByNetLayer] netLayer[%u] is invalid",netLayer);
+            return HCCL_E_PARA;
+        }
         u32 num = 0;
         rankListVec_.clear();
         rankGraph->GetLocalInstRanks(netLayer, rankListVec_, num);
@@ -108,6 +126,12 @@ namespace Hccl {
         HCCL_RUN_INFO("Entry-IRankGraph::GetInstSizeListByNetLayer");
         CHK_PTR_NULL(rankGraphPtr_);
         RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetInstSizeListByNetLayer] netLayer[%u] is invalid", netLayer);
+            return HCCL_E_PARA;
+        }
         u32 size = 0;
         instSizeVec_.clear();
         auto ret = rankGraph->GetNetInstanceList(netLayer, instSizeVec_, size);
@@ -146,7 +170,7 @@ namespace Hccl {
         return HCCL_SUCCESS;
     }
 
-    static HcclResult SetEndpoinLoc(EndpointLocType &locType, const AddrPosition &position)
+    static HcclResult SetEndpointLoc(EndpointLocType &locType, const AddrPosition &position)
     {
         if (position == AddrPosition::DEVICE) {
             locType = ENDPOINT_LOC_TYPE_DEVICE;
@@ -180,7 +204,7 @@ namespace Hccl {
                     HCCL_ERROR("[IRankGraph::%s] SetCommAddress FAILED for srcConn: %s.", __func__, srcConnInterface->Describe().c_str());
                     return result;
                 }
-                CHK_RET(SetEndpoinLoc(commLink.srcEndpointDesc.loc.locType, srcConnInterface->GetPos()));
+                CHK_RET(SetEndpointLoc(commLink.srcEndpointDesc.loc.locType, srcConnInterface->GetPos()));
 
                 // 设置目标端点
                 std::shared_ptr<NetInstance::ConnInterface> dstConnInterface = link.GetTargetIface();
@@ -191,7 +215,7 @@ namespace Hccl {
                     return result;
                 }
 
-                CHK_RET(SetEndpoinLoc(commLink.dstEndpointDesc.loc.locType, dstConnInterface->GetPos()));
+                CHK_RET(SetEndpointLoc(commLink.dstEndpointDesc.loc.locType, dstConnInterface->GetPos()));
 
                 if (commLink.srcEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                     std::shared_ptr<NetInstance::Node> srcNode = peer2peer->GetSourceNode();
@@ -244,7 +268,7 @@ namespace Hccl {
                 HCCL_ERROR("[IRankGraph::%s] SetCommAddress FAILED for srcConn: %s.", __func__, srcInterface->Describe().c_str());
                 return result;
             }
-            CHK_RET(SetEndpoinLoc(commLink.srcEndpointDesc.loc.locType, srcInterface->GetPos()));
+            CHK_RET(SetEndpointLoc(commLink.srcEndpointDesc.loc.locType, srcInterface->GetPos()));
             if (commLink.srcEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                 std::shared_ptr<NetInstance::Node> srcNode = peer2net->GetSourceNode();
                 std::shared_ptr<NetInstance::Peer> srcPeer = std::dynamic_pointer_cast<NetInstance::Peer>(srcNode);
@@ -257,7 +281,7 @@ namespace Hccl {
                 HCCL_ERROR("[IRankGraph::%s] SetCommAddress FAILED for dstConn: %s.", __func__, dstInterface->Describe().c_str());
                 return result;
             }
-            CHK_RET(SetEndpoinLoc(commLink.dstEndpointDesc.loc.locType, dstInterface->GetPos()));
+            CHK_RET(SetEndpointLoc(commLink.dstEndpointDesc.loc.locType, dstInterface->GetPos()));
 
             linkListVec.emplace_back(std::move(commLink));
         }
@@ -270,6 +294,12 @@ namespace Hccl {
         HCCL_RUN_INFO("Entry-IRankGraph::GetLinks");
         CHK_PTR_NULL(rankGraphPtr_);
         RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetLinks] netLayer[%u] is invalid", netLayer);
+            return HCCL_E_PARA;
+        }
         std::vector<NetInstance::Path> paths = rankGraph->GetPaths(netLayer, srcRank, dstRank);
         linkListVec_.clear();
         // 遍历每条path
@@ -309,10 +339,10 @@ namespace Hccl {
         HCCL_RUN_INFO("Entry-IRankGraph::GetTopoInstsByLayer");
         CHK_PTR_NULL(rankGraphPtr_);
         RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
-        auto currNetType = rankGraph->GetNetType(netLayer);
-        if (currNetType != NetType::TOPO_FILE_DESC) {
-            HCCL_ERROR("[IRankGraph::GetTopoInstsByLayer] Only support TOPO_FILE_DESC netType ,current netType is [%d]",
-                       currNetType);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetTopoInstsByLayer] netLayer[%u] is invalid", netLayer);
             return HCCL_E_PARA;
         }
         u32 num = 0;
@@ -328,10 +358,10 @@ namespace Hccl {
         HCCL_RUN_INFO("Entry-IRankGraph::GetTopoType");
         CHK_PTR_NULL(rankGraphPtr_);
         RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
-        auto currNetType = rankGraph->GetNetType(netLayer);
-        if (currNetType != NetType::TOPO_FILE_DESC) {
-            HCCL_ERROR("[IRankGraph::GetTopoInstsByLayer] Only support TOPO_FILE_DESC netType ,current netType is [%d]",
-                       currNetType);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetTopoType] netLayer[%u] is invalid", netLayer);
             return HCCL_E_PARA;
         }
         Hccl::TopoType type;
@@ -359,10 +389,10 @@ namespace Hccl {
         HCCL_RUN_INFO("Entry-IRankGraph::GetRanksByTopoInst");
         CHK_PTR_NULL(rankGraphPtr_);
         RankGraph* rankGraph = static_cast<RankGraph*>(rankGraphPtr_);
-        auto currNetType = rankGraph->GetNetType(netLayer);
-        if (currNetType != NetType::TOPO_FILE_DESC) {
-            HCCL_ERROR("[IRankGraph::GetTopoInstsByLayer] Only support TOPO_FILE_DESC netType ,current netType is [%d]",
-                       currNetType);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetRanksByTopoInst] netLayer[%u] is invalid", netLayer);
             return HCCL_E_PARA;
         }
         u32 num = 0;
@@ -375,4 +405,60 @@ namespace Hccl {
         *rankNum = ranksVec_.size();
         return HCCL_SUCCESS;
     }
+
+    HcclResult IRankGraph::GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t *num)
+    {
+        HCCL_RUN_INFO("Entry-IRankGraph::GetEndpointNum");
+        CHK_PTR_NULL(rankGraphPtr_);
+        RankGraph *rankGraph = static_cast<RankGraph *>(rankGraphPtr_);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetEndpointNum] netLayer[%u] is invalid", netLayer);
+            return HCCL_E_PARA;
+        }
+        auto ret = rankGraph->GetEndpointNum(netLayer, topoInstId, num);
+        if (ret != HCCL_SUCCESS) {
+            HCCL_ERROR("[IRankGraph::GetEndpointNum] Faild to get endpoint num at netLayer [%u] with topoInstId",
+                       netLayer, topoInstId);
+            return ret;
+        }
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult IRankGraph::GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t *descNum,
+                                           EndpointDesc *endpointDesc)
+    {
+        HCCL_RUN_INFO("Entry-IRankGraph::GetEndpointDesc");
+        CHK_PTR_NULL(rankGraphPtr_);
+        RankGraph *rankGraph = static_cast<RankGraph *>(rankGraphPtr_);
+        u32 rankId = rankGraph->GetMyRank();
+        std::set<u32> levels = rankGraph->GetLevels(rankId);
+        if (levels.find(netLayer) == levels.end()) {
+            HCCL_ERROR("[IRankGraph::GetEndpointDesc] netLayer[%u] is invalid", netLayer);
+            return HCCL_E_PARA;
+        }
+        auto ret = rankGraph->GetEndpointDesc(netLayer, topoInstId, descNum, endpointDesc);
+        if (ret != HCCL_SUCCESS) {
+            HCCL_ERROR("[IRankGraph::GetEndpointDesc] Failed to get endpoint desc at netLayer [%u] with descNum [%u]",
+                       netLayer, descNum);
+            return ret;
+        }
+        return HCCL_SUCCESS;
+    }
+
+    HcclResult IRankGraph::GetEndpointInfo(uint32_t rankId, const EndpointDesc *endPointDesc, EndpointAttr endpointAttr,
+                                           uint32_t infoLen, void *info)
+    {
+        HCCL_RUN_INFO("Entry-IRankGraph::GetEndpointInfo");
+        CHK_PTR_NULL(rankGraphPtr_);
+        RankGraph *rankGraph = static_cast<RankGraph *>(rankGraphPtr_);
+        HcclResult ret = rankGraph->GetEndpointInfo(rankId, endPointDesc, endpointAttr, infoLen, info);
+        if (ret != HCCL_SUCCESS) {
+            HCCL_ERROR("[IRankGraph::GetEndpointInfo] Faild to get endpoint info with rank [%u]", rankId);
+            return ret;
+        }
+        return HCCL_SUCCESS;
+    }
+
 } // namespace Hccl

```

### src/orion/interface/rank_graph_interface.h
```diff
@@ -35,6 +35,10 @@ namespace Hccl {
         HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
         HcclResult GetTopoType(const uint32_t netLayer, const uint32_t topoInstId, CommTopo* topoType);
         HcclResult GetRanksByTopoInst(const uint32_t netLayer, const uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
+        HcclResult GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t *num);
+        HcclResult GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t *descNum, EndpointDesc *endpointDesc);
+        HcclResult GetEndpointInfo(uint32_t rankId, const EndpointDesc *endPointDesc, EndpointAttr endpointAttr,
+                                           uint32_t infoLen, void *info);
 
     private:
         void *rankGraphPtr_;

```

### test/ut/framework/communicator/ut_HcclRankGraph_API_test.cc
```diff
@@ -0,0 +1,348 @@
+#include "hccl_api_base_test.h"
+#include "hccl_comm_pub.h"
+#include "llt_hccl_stub_rank_graph.h"
+
+class HcclRankGraphTest :public BaseInit{
+    public:
+    void SetUp() override
+    {
+        BaseInit::SetUp();
+    }
+    void TearDown() override
+    {
+        BaseInit::TearDown();
+        GlobalMockObject::verify();
+    }
+    protected:
+     void SetUpCommAndGraph(std::shared_ptr<hccl::hcclComm>& hcclCommPtr, std::shared_ptr<Hccl::RankGraph>& rankGraphV2, void*& comm, HcclResult& ret){
+        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910_95)).will(returnValue(HCCL_SUCCESS));
+
+        bool isDeviceSide{false};
+        MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
+        MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
+
+        setenv("HCCL_INDEPENDENT_OP","1",1);
+
+        RankGraphStub rankGraphStub;
+        rankGraphV2 = rankGraphStub.Create2PGraph();
+
+        void* commV2 = (void*)0x2000;
+        uint32_t rank = 1;
+
+        HcclMem cclBuffer;
+        cclBuffer.size = 1;
+        cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
+        cclBuffer.addr = (void*)0x1000;
+
+        char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
+        hcclCommPtr = std::make_shared<hccl::hcclComm>(1,1,commName);
+        ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName);
+        CollComm* collComm = hcclCommPtr->GetCollComm();
+        comm = static_cast<HcclComm>(hcclCommPtr.get());
+
+     }
+};
+
+TEST_F(HcclRankGraphTest, Ut_HcclGetRankSize_When_ValidParam_Expect_Return_HCCL_SUCCESS)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t rankSize = 0;
+    ret = HcclGetRankSize(comm, &rankSize);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    EXPECT_EQ(rankSize,2);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclGetRankSize_When_Param_Is_InVaild_Expect_Return_Error)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t rankSize = 0;
+    ret = HcclGetRankSize(nullptr, &rankSize);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetLayers_When_ValidParam_Expect_Return_HCCL_SUCCESS)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t netLayerNum = 0;
+    uint32_t* netLayers;
+    ret = HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    EXPECT_EQ(netLayerNum, 1);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetLayers_When_Param_Is_InVaild_Expect_Return_Error)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t netLayerNum = 0;
+    uint32_t* netLayers;
+    ret = HcclRankGraphGetLayers(nullptr, &netLayers, &netLayerNum);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetLayers(comm, nullptr, &netLayerNum);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRankSizeByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t rankNum = 0;
+    uint32_t netLayer = 0;
+    ret = HcclRankGraphGetRankSizeByLayer(comm, netLayer, &rankNum);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    EXPECT_EQ(rankNum, 2);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRankSizeByLayer_When_Param_Is_InVaild_Expect_Return_Error)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t rankNum = 0;
+    uint32_t netLayer = 0;
+    ret = HcclRankGraphGetRankSizeByLayer(nullptr, netLayer, &rankNum);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetRankSizeByLayer(comm, 10, &rankNum);
+    EXPECT_EQ(ret, HCCL_E_PARA);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRanksByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t rankNum = 0;
+    uint32_t* ranks;
+    uint32_t netLayer = 0;
+    ret = HcclRankGraphGetRanksByLayer(comm, netLayer, &ranks, &rankNum);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    EXPECT_EQ(rankNum, 2);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRanksByLayer_When_Param_Is_InVaild_Expect_Return_Error)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t rankNum = 0;
+    uint32_t netLayer = 0;
+    uint32_t* ranks;
+    ret = HcclRankGraphGetRanksByLayer(nullptr, netLayer, &ranks, &rankNum);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetRanksByLayer(comm, netLayer, nullptr, &rankNum);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetRanksByLayer(comm, 10, &ranks, &rankNum);
+    EXPECT_EQ(ret, HCCL_E_PARA);
+}
+
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetTopoTypeByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    CommTopo type;
+    uint32_t netLayer = 0;
+    ret = HcclRankGraphGetTopoTypeByLayer(comm, netLayer, &type);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    EXPECT_EQ(CommTopo::COMM_TOPO_CUSTOM, type);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetTopoTypeByLayer_When_Param_Is_InVaild_Expect_Return_Error)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    CommTopo type;
+    uint32_t netLayer = 0;
+    ret = HcclRankGraphGetTopoTypeByLayer(nullptr, netLayer, &type);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetTopoTypeByLayer(comm, 10, &type);
+    EXPECT_EQ(ret, HCCL_E_PARA);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetInstSizeListByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t netLayer = 0;
+    uint32_t listSize = 0;
+    uint32_t* instSizeList;
+    ret = HcclRankGraphGetInstSizeListByLayer(comm, netLayer, &instSizeList, &listSize);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    EXPECT_EQ(listSize, 1);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetInstSizeListByLayer_When_Param_Is_InVaild_Expect_Return_Error)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t netLayer = 0;
+    uint32_t listSize = 0;
+    uint32_t* instSizeList;
+    ret = HcclRankGraphGetInstSizeListByLayer(nullptr, netLayer, &instSizeList, &listSize);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetInstSizeListByLayer(comm, netLayer, nullptr, &listSize);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetInstSizeListByLayer(comm, 10, &instSizeList, &listSize);
+    EXPECT_EQ(ret, HCCL_E_PARA);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetLinks_When_ValidParam_Expect_Return_HCCL_SUCCESS)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t linkNum = 0;
+    uint32_t netLayer = 0;
+    uint32_t srcRank = 0;
+    uint32_t dstRank = 1;
+    CommLink* links;
+    ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &linkNum);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    EXPECT_EQ(linkNum, 1);
+    EXPECT_EQ(links[0].linkAttr.hop, 1);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetLinks_When_Param_Is_InVaild_Expect_Return_Error)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t linkNum = 0;
+    uint32_t netLayer = 0;
+    uint32_t srcRank = 0;
+    uint32_t dstRank = 1;
+    CommLink* links;
+    ret = HcclRankGraphGetLinks(nullptr, netLayer, srcRank, dstRank, &links, &linkNum);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, nullptr, &linkNum);
+    EXPECT_EQ(ret, HCCL_E_PTR);
+    ret = HcclRankGraphGetLinks(comm, 10, srcRank, dstRank, &links, &linkNum);
+    EXPECT_EQ(ret, HCCL_E_PARA);
+}
+
+TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetEndpointInfo_When_ValidParam_Expect_Return_HCCL_SUCCESS)
+{
+    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
+    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
+    void* comm;
+    HcclResult ret;
+
+    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
+
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+
+    uint32_t num = 0;
+    uint32_t topoInstId = 0;
+    uint32_t netLayer = 0;
+    ret = HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &num);
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    EXPECT_EQ(num, 1);  
+
+    uint32_t descNum = num;
+    std::unique_ptr<EndpointDesc[]>  endpointDesc(new EndpointDesc[descNum]);
+    ret = HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &num, endpointDesc.get()); 
+    EXPECT_EQ(ret, HCCL_SUCCESS);
+    for(uint32_t i = 0; i<num;i++){
+        EXPECT_EQ(endpointDesc[i].protocol, COMM_PROTOCOL_UBC_CTP);
+        uint32_t infoLen = sizeof(EndpointAttrBwCoeff);
+        EndpointAttrBwCoeff bwCoeff{};
+        ret = HcclRankGraphGetEndpointInfo(comm, 0, &endpointDesc[i], ENDPOINT_ATTR_BW_COEFF, infoLen, &bwCoeff);
+        EXPECT_EQ(ret, HCCL_SUCCESS);
+    }
+}

```

### test/ut/stub/llt_hccl_stub_rank_graph.cc
```diff
@@ -0,0 +1,81 @@
+/**
+ * Copyright (c) 2026 Huawei Technologies Co., Ltd.
+ * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
+ * CANN Open Software License Agreement Version 2.0 (the "License").
+ * Please refer to the License for details. You may not use this file except in compliance with the License.
+ * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
+ * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
+ * See LICENSE in the root of the software repository for the full text of the License.
+ */
+#include "llt_hccl_stub_rank_graph.h"
+#include "rank_graph_builder.h"
+
+namespace hccl {
+
+std::shared_ptr<Hccl::NetInstance::Peer> RankGraphStub::InitPeer(Hccl::RankId rankId, Hccl::LocalId localId, Hccl::DeviceId deviceId)
+{
+    std::shared_ptr<Hccl::NetInstance::Peer> peer = std::make_shared<Hccl::NetInstance::Peer>(rankId, localId, localId, deviceId);
+    return peer;
+}
+
+std::shared_ptr<Hccl::NetInstance> RankGraphStub::InitNetInstance(u32 netLayer, std::string id)
+{
+    std::shared_ptr<Hccl::NetInstance> netInst;
+    if (netLayer == 0) {
+        netInst = std::make_shared<Hccl::InnerNetInstance>(netLayer, id);
+    } else {
+        netInst = std::make_shared<Hccl::ClosNetInstance>(netLayer, id);
+    }
+    return netInst;
+}
+
+std::shared_ptr<Hccl::NetInstance::ConnInterface> RankGraphStub::InitConnInterface(Hccl::IpAddress addr)
+{
+    Hccl::AddrPosition pos = Hccl::AddrPosition::DEVICE;
+    Hccl::LinkType inputLinkType = Hccl::LinkType::PEER2PEER;
+    std::set<Hccl::LinkProtocol> inputLinkProtocol = {Hccl::LinkProtocol::UB_CTP};
+    std::set<std::string> ports = {"0/1"};
+    Hccl::TopoType topoType = Hccl::TopoType::CLOS;
+    uint32_t topoInstId = 0;
+    std::shared_ptr<Hccl::NetInstance::ConnInterface> iface = std::make_shared<Hccl::NetInstance::ConnInterface>(addr, ports, pos, inputLinkType, inputLinkProtocol, topoType, topoInstId);
+    return iface;
+}
+
+
+std::shared_ptr<Hccl::RankGraph> RankGraphStub::Create2PGraph()
+{
+    Hccl::RankGraph  rankGraph(0);
+    std::shared_ptr<Hccl::NetInstance> netInstLayer0 = InitNetInstance(0, "layer0");
+    std::shared_ptr<Hccl::NetInstance::Peer> peer0 = InitPeer(0, 0, 0);
+    std::shared_ptr<Hccl::NetInstance::Peer> peer1 = InitPeer(1, 1, 1);
+    
+    peer0->AddNetInstance(netInstLayer0);
+    peer1->AddNetInstance(netInstLayer0);
+    netInstLayer0->AddRankId(peer0->GetRankId());
+    netInstLayer0->AddRankId(peer1->GetRankId());
+    netInstLayer0->AddNode(peer0);
+    netInstLayer0->AddNode(peer1);
+
+    char rank0Address[] = "192.168.1.0";
+    Hccl::IpAddress rank0Addr(rank0Address);
+    auto iface0 = InitConnInterface(rank0Addr);
+
+    char rank1Address[] = "192.168.1.1";
+    Hccl::IpAddress rank1Addr(rank1Address);
+    auto iface1 = InitConnInterface(rank1Addr);
+
+    peer0->AddConnInterface(0, iface0);
+    peer1->AddConnInterface(0, iface1);
+
+    Hccl::LinkType type = Hccl::LinkType::PEER2PEER;
+    std::set<Hccl::LinkProtocol> protocols = {Hccl::LinkProtocol::UB_CTP};
+    auto link = std::make_shared<Hccl::NetInstance::Link>(peer0,peer1,iface0,iface1,type,protocols);
+    netInstLayer0->AddLink(link);
+
+    rankGraph.AddPeer(peer0);
+    rankGraph.AddPeer(peer1);
+    rankGraph.AddNetInstance(netInstLayer0);
+    rankGraph.InitInnerRanks();
+    return std::make_shared<Hccl::RankGraph>(rankGraph);
+}
+} // namespace hccl
\ No newline at end of file

```

### test/ut/stub/llt_hccl_stub_rank_graph.h
```diff
@@ -0,0 +1,33 @@
+/**
+ * Copyright (c) 2026 Huawei Technologies Co., Ltd.
+ * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
+ * CANN Open Software License Agreement Version 2.0 (the "License").
+ * Please refer to the License for details. You may not use this file except in compliance with the License.
+ * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
+ * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
+ * See LICENSE in the root of the software repository for the full text of the License.
+ */
+
+#ifndef LLT_HCCL_STUB_RANK_GRAPH_H
+#define LLT_HCCL_STUB_RANK_GRAPH_H
+
+#include <vector>
+#include <hccl_types.h>
+#include "rank_gph.h"
+
+namespace hccl {
+
+class RankGraphStub {
+public:
+    explicit RankGraphStub() = default;
+    ~RankGraphStub() = default;
+    std::shared_ptr<Hccl::RankGraph> Create2PGraph();
+private:
+    std::shared_ptr<Hccl::NetInstance::Peer> InitPeer(Hccl::RankId rankId, Hccl::LocalId localId, Hccl::DeviceId deviceId);
+    std::shared_ptr<Hccl::NetInstance> InitNetInstance(uint32_t netLayer, std::string id);
+    std::shared_ptr<Hccl::NetInstance::ConnInterface> InitConnInterface(Hccl::IpAddress addr);
+};
+
+}  // namespace hccl
+
+#endif
\ No newline at end of file

```

> 注: 以下非 C/C++ 文件未纳入审查: test/ut/stub/CMakeLists.txt
