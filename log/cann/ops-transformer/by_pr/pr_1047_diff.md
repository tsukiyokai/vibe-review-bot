# PR #1047: fix offload

- 作者: chenjunting
- 分支: master -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/1047
- 描述: fix offload

## 变更文件 (1 个, 其中 C/C++ 文件 1 个)

- [modified] src/framework/op_base/src/op_base.cc (+8, -4) *

## Diff 内容

### src/framework/op_base/src/op_base.cc
```diff
@@ -356,13 +356,12 @@ HcclResult HcclCommInitCollComm(uint32_t rank, void **commV2, HcclComm *comm)
     HCCL_INFO("HcclCommInitCollComm start.");
 #if (!defined (HCCD)) && (!defined (CCL_KERNEL_AICPU))
     HcclUs startut = TIME_NOW();
-
-    // 图模式
+<<<<<<< HEAD
     u32 rankNum = 0;
     CHK_RET(HcclGetRankSizeV2(*commV2, &rankNum));
+=======
+>>>>>>> 3a358d9c5bc20b0f7238aebe75ff9d703b230edf
     char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
-    CHK_RET(HcclGetCommNameV2(*commV2, commName));
-    CHK_RET(HcomSetGroupTopoInfo(commName, rankNum));
     //获取cclbuffer
     uintptr_t cclBufferAddr{0};
     std::size_t cclBufferSize{0};
@@ -834,6 +833,11 @@ HcclResult HcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank,
         [&]() -> HcclResult {
             void *commV2 = nullptr;
             CHK_RET(HcclCommInitClusterInfoConfigV2(clusterInfo, rank, config, &commV2));
+            u32 rankNum = 0;
+            CHK_RET(HcclGetRankSizeV2(commV2, &rankNum));
+            char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
+            CHK_RET(HcclGetCommNameV2(commV2, commName));
+            CHK_RET(HcomSetGroupTopoInfo(commName, rankNum));
             const char *indOp = getenv("HCCL_INDEPENDENT_OP");
             if (indOp == nullptr || strcmp(indOp, "") == 0) {
                 *comm = commV2;

```
