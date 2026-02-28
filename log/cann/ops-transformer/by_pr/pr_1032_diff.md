# PR #1032: Onesided Surport Netlayer

- 作者: linyixin4
- 分支: master -> master
- 链接: https://gitcode.com/cann/hcomm-dev/pulls/1032
- 描述: Onesided Surport Netlayer

## 变更文件 (4 个, 其中 C/C++ 文件 4 个)

- [modified] src/orion/framework/communicator/aicpu/one_sided_component/one_sided_component_lite.cpp (+5, -1) *
- [modified] src/orion/framework/resource_manager/transport/aicpu/connected_link_mgr.cc (+5, -5) *
- [modified] src/orion/framework/service/one_sided_service/hccl_one_sided_service.cc (+25, -6) *
- [modified] src/orion/framework/service/one_sided_service/hccl_one_sided_service.h (+2, -0) *

## Diff 内容

### src/orion/framework/communicator/aicpu/one_sided_component/one_sided_component_lite.cpp
```diff
@@ -38,7 +38,11 @@ HcclResult OneSidedComponentLite::Orchestrate(const HcclAicpuOpLite &op, InsQueP
     }
 
     RankId rmtRankId = op.sendRecvRemoteRank;
-    vector<LinkData> link = linkMgr_->GetLinks(0, rmtRankId);
+    vector<LinkData> link = linkMgr_->GetLinks(rmtRankId);
+    if(link.size() == 0){
+        HCCL_ERROR("[OneSidedComponentLite][Orchestrate] link size invalid");
+        return HCCL_E_PARA;
+    }
     HCCL_INFO("[%s] Orchestrate Mode: Instruction %d.", __func__, rmtRankId);
     if (op.algOperator.opType == OpType::BATCHGET) {
         std::unique_ptr<Instruction> ins = std::make_unique<InsBatchOneSidedRead>(rmtRankId, link[0], usrInSlice, usrOutSlice);

```

### src/orion/framework/resource_manager/transport/aicpu/connected_link_mgr.cc
```diff
@@ -13,12 +13,12 @@
 namespace Hccl {
 const vector<LinkData> &ConnectedLinkMgr::GetLinks(RankId dstRank)
 {
-    for (auto levelMap : levelRankPairLinkDataMap) {
-        if (levelRankPairLinkDataMap[levelMap.first].find(dstRank) != levelRankPairLinkDataMap[levelMap.first].end()
-            && levelRankPairLinkDataMap[levelMap.first][dstRank].size() > 0) {
+    for (u32 level = 0; level < MAX_NET_LAYER; level++) {
+        if (levelRankPairLinkDataMap[level].find(dstRank) != levelRankPairLinkDataMap[level].end()
+            && levelRankPairLinkDataMap[level][dstRank].size() > 0) {
             HCCL_INFO("[ConnectedLinkMgr][GetLinks] level[%u], dstRank[%d], links.size[%u]",
-                levelMap.first, dstRank, levelRankPairLinkDataMap[levelMap.first][dstRank].size());
-            return levelRankPairLinkDataMap[levelMap.first][dstRank];
+                level, dstRank, levelRankPairLinkDataMap[level][dstRank].size());
+            return levelRankPairLinkDataMap[level][dstRank];
         }
     }
     HCCL_WARNING("[ConnectedLinkMgr][GetLinks] links is empty, dstRank[%d]", dstRank);

```

### src/orion/framework/service/one_sided_service/hccl_one_sided_service.cc
```diff
@@ -56,8 +56,18 @@ LinkData HcclOneSidedService::GetLinkData(RankId remoteRankId)
 {
     if (linkDataMap_.find(remoteRankId) == linkDataMap_.end()) {
         // 组建linkData
-        LinkData linkData(comm_->GetRankGraph()->GetPaths(0, comm_->GetMyRank(), remoteRankId)[0]);
-        linkDataMap_.emplace(remoteRankId, linkData);
+        for(u32 level = 0; level < MAX_NET_LAYER; level++) {
+            auto paths = comm_->GetRankGraph()->GetPaths(level, comm_->GetMyRank(), remoteRankId);
+            if(paths.size() != 0) {
+                LinkData linkData(paths[0]);
+                linkDataMap_.emplace(remoteRankId, linkData);
+                HCCL_INFO("[HcclOneSidedService][GetLinkData] linkData[%s]", linkDataMap_.at(remoteRankId).Describe().c_str());
+                return linkDataMap_.at(remoteRankId);
+            }
+        }
+        MACRO_THROW(InternalException,
+                    StringFormat("[HcclOneSidedService][GetLinkData] sRankId[%u] dRankId[%u] netInst has no path.",
+                                comm_->GetMyRank(), remoteRankId));
     }
     HCCL_INFO("[HcclOneSidedService][GetLinkData] linkData[%s]", linkDataMap_.at(remoteRankId).Describe().c_str());
     return linkDataMap_.at(remoteRankId);
@@ -69,8 +79,6 @@ HcclResult HcclOneSidedService::CheckLink(LinkData linkData) const
     CHK_PRT_RET(
         (linkData.GetLinkProtocol() != LinkProtocol::UB_CTP && linkData.GetLinkProtocol() != LinkProtocol::UB_TP),
         HCCL_ERROR("[HcclOneSidedService][CheckLink] Proto is not UB, not support"), HCCL_E_NOT_SUPPORT);
-    CHK_PRT_RET(linkData.GetHop() > 1, HCCL_ERROR("[HcclOneSidedService][CheckLink]Hop is greater than 1, not support"),
-                HCCL_E_NOT_SUPPORT);
     return HCCL_SUCCESS;
 }
 
@@ -443,6 +451,18 @@ DevBuffer *HcclOneSidedService::PackResToKernelLanuch(CollAlgOpReq &opReq, bool
     return devMem.get();
 }
 
+void HcclOneSidedService::FillLevelRank(OpType opType, RankId remoteRankId, CollAlgOpReq &opReq) const
+{
+    opReq.algName = OpTypeToString(opType);
+    auto levels = comm_->GetRankGraph()->GetLevels(comm_->GetMyRank());
+    for (auto level : levels) {
+        auto paths = comm_->GetRankGraph()->GetPaths(level, comm_->GetMyRank(), remoteRankId);
+        if(paths.size() != 0) {
+            opReq.resReq.levelRankPairs.push_back(make_pair(level, remoteRankId));
+        }
+    }
+}
+
 HcclResult HcclOneSidedService::BatchOpKernelLaunch(OpType opType, RankId remoteRankId, const HcclOneSideOpDesc *desc,
                                                     u32 descNum, shared_ptr<Stream> stream)
 {
@@ -457,8 +477,7 @@ HcclResult HcclOneSidedService::BatchOpKernelLaunch(OpType opType, RankId remote
         desc->localAddr, desc->remoteAddr, desc->count, desc->dataType);
 
     CollAlgOpReq opReq;
-    opReq.algName = OpTypeToString(opType);
-    opReq.resReq.levelRankPairs.push_back(make_pair(0, remoteRankId));
+    FillLevelRank(opType, remoteRankId, opReq);
     HCCL_INFO("[HcclOneSidedService][BatchOpKernelLaunch] FillOneSidedOperator start");
     // 填充通信算子信息
     FillOneSidedOperator(opType, remoteRankId, desc);

```

### src/orion/framework/service/one_sided_service/hccl_one_sided_service.h
```diff
@@ -93,6 +93,8 @@ private:
     void OneSidedAicpuKernelLaunch(HcclKernelLaunchParam &param, Stream &stream)const ;
 
     std::unordered_map<std::string, std::shared_ptr<DevBuffer>> OneSidedLoadMap;
+
+    void FillLevelRank(OpType opType, RankId remoteRankId, CollAlgOpReq &req) const;
 };
 } // namespace Hccl
 

```
