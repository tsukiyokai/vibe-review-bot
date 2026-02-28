# PR #1150: monitor time standard motify

- 作者: lilin_137
- 分支: master -> master
- 链接: https://gitcode.com/cann/hcomm-dev/merge_requests/1150
- 描述: monitor time standard motify

## 变更文件 (3 个, 其中 C/C++ 文件 3 个)

- [modified] src/framework/common/src/config/env_config.cc (+5, -5) *
- [modified] src/framework/device/framework/aicpu_communicator.cc (+2, -2) *
- [modified] src/framework/device/framework/aicpu_zero_copy_exchanger.cc (+2, -2) *

## Diff 内容

### src/framework/common/src/config/env_config.cc
```diff
@@ -35,7 +35,6 @@ const std::string INCONSISTENT_CHECK_CONFIG = "inconsistent_check:";
 const std::string CONNECTION_FAULT_DETECTION_TIME = "connection_fault_detection_time:";
 const std::string TASK_MONITOR_INTERVAL = "task_monitor_interval:";
 constexpr static const s32 HCCL_MAX_LINK_TIME_OUT_S  = (120 * 60); // HCCL 最大探测超时时间设置为120*60s
-constexpr static const s32 HCCL_MAX_TASK_MONITOR_INTERVAL = 2147483647; // HCCL 最大task监控时长，与notify最大值保持一致
 HcclResult InitEnvConfig()
 {
     std::lock_guard<std::mutex> lock(g_envConfigMutex);
@@ -654,13 +653,14 @@ HcclResult ParseMonitor(std::string &taskMonitorInterval, s32 &monitorTime)
             "is invalid, errorno[%d]", taskMonitorInterval.c_str(), ret), ret);
     }
 
+    s32 maxTimeInMs = HCCL_MAX_LINK_TIME_OUT_S * 1000;
     if (monitorTime == 0) {
         g_envConfig.dfsTaskMonitorInterval = 0;
-    } else if (monitorTime >= 0 && monitorTime <= HCCL_MAX_TASK_MONITOR_INTERVAL) {
+    } else if (monitorTime >= 0 && monitorTime <= maxTimeInMs) {
         g_envConfig.dfsTaskMonitorInterval = monitorTime;
     } else { // 不在允许范围内报错
         HCCL_ERROR("[ParseDFSConfig] HCCL_DFS_CONFIG-task_monitor_interval[%d] is invalid, except: [0, %d]",
-            monitorTime, HCCL_MAX_TASK_MONITOR_INTERVAL);
+            monitorTime, maxTimeInMs);
         return HCCL_E_PARA;
     }
     return HCCL_SUCCESS;
@@ -727,7 +727,7 @@ HcclResult ParseDFSConfig()
         g_envConfig.dfsConnectionFaultDetectionTime = HCCL_MIN_CONNECT_FAULT_DETECTION_TIME;
         HCCL_RUN_INFO("[HCCL_ENV][Parse] HCCL_DFS_CONFIG cluster_heartbeat set by environment to [%d], "
             "stuck_detection set by environment to [%d], connection_fault_detection_time[%d]s inconsistentCheckSwitch[%d],"
-            "task_monitor_interval[%u]s", g_envConfig.enableClusterHeartBeat, g_envConfig.opCounterEnable,
+            "task_monitor_interval[%u]ms", g_envConfig.enableClusterHeartBeat, g_envConfig.opCounterEnable,
             g_envConfig.dfsConnectionFaultDetectionTime, g_envConfig.inconsistentCheckSwitch, g_envConfig.dfsTaskMonitorInterval);
         return HCCL_SUCCESS;
     }
@@ -748,7 +748,7 @@ HcclResult ParseDFSConfig()
 
     HCCL_RUN_INFO("[HCCL_ENV][Parse] HCCL_DFS_CONFIG cluster_heartbeat set by environment to [%d], "
         "stuck_detection set by environment to [%d], connection_fault_detection_time[%d]s inconsistentCheckSwitch[%d],"
-        "task_monitor_interval[%u]s", g_envConfig.enableClusterHeartBeat, g_envConfig.opCounterEnable,
+        "task_monitor_interval[%u]ms", g_envConfig.enableClusterHeartBeat, g_envConfig.opCounterEnable,
         g_envConfig.dfsConnectionFaultDetectionTime, g_envConfig.inconsistentCheckSwitch, g_envConfig.dfsTaskMonitorInterval);
     return HCCL_SUCCESS;
 }

```

### src/framework/device/framework/aicpu_communicator.cc
```diff
@@ -3167,8 +3167,8 @@ HcclResult HcclCommAicpu::StreamTaskMonitor(void)
         }
 
         auto timeVal = DURATION_US(curTime - streamMontior.historyTime).count();
-        if (timeVal >= taskMonitorInterval_ * 1000 * 1000) {
-            HCCL_RUN_INFO("[StreamTaskMonitor]prof monitor streamId:%d, sqid:%d, head:%u, tail:%u, time %s %s",
+        if (timeVal >= taskMonitorInterval_ * 1000) {
+            HCCL_RUN_INFO("[StreamTaskMonitor]prof monitor streamId:%d, sqid:%d, head:%u, tail:%u, time %s us %s",
                 stream.id(), stream.sqId(), sqHead, sqTail, std::to_string(timeVal).c_str(), tmp.c_str());
             HCCL_RUN_INFO("[StreamTaskMonitor]prof monitor %s", GetTaskExceptionOpInfo(sqHead,sqeContextBuffer).c_str());
             PrintTaskExceptionTaskQue(sqHead, sqeContextBuffer, true);

```

### src/framework/device/framework/aicpu_zero_copy_exchanger.cc
```diff
@@ -57,11 +57,11 @@ HcclResult AicpuZeroCopyExchanger::ExchangeAddress(const std::string &tag, void
     CHK_RET(UpdateTransportAddress());
     HcclUs endut = TIME_NOW();
     auto timeVal = DURATION_US(endut - startut).count();
-    if (taskMonitorInterval_ != 0 && timeVal >= taskMonitorInterval_) {
+    if (taskMonitorInterval_ != 0 && timeVal >= taskMonitorInterval_ * 1000) {
         std::string endInfo;
         endInfo.reserve(100);
         endInfo = "task time: " + std::to_string(timeVal) + " us," +
-            "taskMonitor" + std::to_string(taskMonitorInterval_) + " us";
+            "taskMonitor" + std::to_string(taskMonitorInterval_ * 1000) + " us";
         HCCL_RUN_INFO("[ExchangeAddress] %s, %s", tag.c_str(), endInfo.c_str());
     }
     return HCCL_SUCCESS;

```
