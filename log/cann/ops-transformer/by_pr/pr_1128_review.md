# Code Review: PR #1128

| 属性 | 值 |
|------|------|
| 标题 | ccu dfx and ut. |
| 作者 | acjr0011 |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/1128](https://gitcode.com/cann/hcomm-dev/merge_requests/1128) |
| 审查时间 | 2026-02-23 20:43:16 |
| 审查工具 | Claude Code (`codereview` skill) |
| 发现 | 严重 3 / 一般 3 / 建议 1 |

---

## 变更概述

本 PR 对 CCU (Communication Compute Unit) 异常处理的 DFX 诊断功能进行增强，主要变更：

- task_exception_handler.cpp: 重构 `ProcessCcuException` 和 `ProcessCcuMC2Exception`，从 runtime 异常信息中直接获取 `missionStatus` 和 `instrId`（原先通过 `GetCcuMissionContext` 查询寄存器获取）；新增 `ccum_dfx_info` 结构体和 `PrintPanicLogInfo` 用于打印 CCU 寄存器诊断信息；字段名从 `sqeInfo`/`ccuTaskNum` 更新为 `missionInfo`/`ccuMissionNum`
- ccu_error_handler.cpp: `CcuErrorHandler::GetCcuErrorMsg` 移除内部 `GetCcuMissionContext` 调用，改由调用方传入 `missionStatus` 和 `currIns`
- ccu_dfx.cpp / ccu_dfx.h / ccu_error_handler.h: 同步更新函数签名
- 测试文件 (3 个): 同步更新测试用例

涉及 9 个文件，核心是将 status/instrId 的获取从被调方移到调用方，使诊断信息的数据源更直接。

## 审查发现

共发现 7 个问题（严重 3 / 一般 3 / 建议 1）

---

### #1 [严重] 格式字符串缺少分隔空格，日志输出粘连

- 位置: `src/legacy/framework/dfx/task_exception/task_exception_handler.cpp:325, 326, 327`
- 规则: 3.1.3（格式字符串参数匹配——此处为格式可读性缺陷）
- 置信度: 确定

问题代码:

    HCCL_ERROR("CCU DFX INFO: SQE_RECV_CNT[%u] SQE_SEND_CNT[%u] MISSION_DFX[%u]"
                "TIF_SQE_CNT[%u] TIF_CQE_CNT[%u] CIF_SQE_CNT[%u] CIF_CQE_CNT[%u]"
                "SQE_DROP_CNT[%u] SQE_ADDR_LEN_ERR_DROP_CNT[%u] ccumIsEnable[%u]",

分析:
C++ 相邻字符串字面量会直接拼接。第一行末尾 `MISSION_DFX[%u]"` 与第二行开头 `"TIF_SQE_CNT` 之间无空格，第二行末尾 `CIF_CQE_CNT[%u]"` 与第三行开头 `"SQE_DROP_CNT` 之间同理无空格。实际输出会是 `MISSION_DFX[3]TIF_SQE_CNT[4]` 和 `CIF_CQE_CNT[7]SQE_DROP_CNT[8]` 这样的粘连形式，降低诊断日志可读性。10 个 `%u` 与 10 个参数数量匹配正确，不存在 UB。

修复建议:
在每行结尾的闭引号前或下一行开引号后加空格：

    HCCL_ERROR("CCU DFX INFO: SQE_RECV_CNT[%u] SQE_SEND_CNT[%u] MISSION_DFX[%u] "
                "TIF_SQE_CNT[%u] TIF_CQE_CNT[%u] CIF_SQE_CNT[%u] CIF_CQE_CNT[%u] "
                "SQE_DROP_CNT[%u] SQE_ADDR_LEN_ERR_DROP_CNT[%u] ccumIsEnable[%u]",

---

### #2 [严重] 不必要的 const_cast，违反 const 正确性

- 位置: `src/legacy/framework/dfx/task_exception/task_exception_handler.cpp:319`
- 规则: 2.10.6（只读形参缺 const）/ 2.7.1（避免不必要的类型转换）
- 置信度: 确定

问题代码:

    struct ccum_dfx_info *info = reinterpret_cast<struct ccum_dfx_info *>(const_cast<uint8_t*>(panicLog));

分析:
`panicLog` 是 `const uint8_t*`，函数只读取数据不修改。`const_cast` 移除 const 是不必要的，`reinterpret_cast<const struct ccum_dfx_info*>(panicLog)` 即可完成转换。去掉 const 限定后，若后续有人误通过 `info` 指针修改数据会导致未定义行为。

修复建议:

    const struct ccum_dfx_info *info = reinterpret_cast<const struct ccum_dfx_info*>(panicLog);

---

### #3 [严重] 循环未对 ccuMissionNum 做上界检查，存在数组越界风险

- 位置: `src/legacy/framework/dfx/task_exception/task_exception_handler.cpp:337, 440`
- 规则: 红线 1.2（数组越界）
- 置信度: 待确认（取决于 runtime 层是否保证 ccuMissionNum 不超过数组容量）

问题代码:

    // ProcessCcuMC2Exception 中:
    for (uint32_t i = 0; i < ccuExDetailInfo.ccuMissionNum; ++i) {
        const auto& missionInfo = ccuExDetailInfo.missionInfo[i];

    // ProcessCcuException 中:
    for (uint32_t i = 0; i < ccuExDetailInfo.ccuMissionNum; ++i) {
        const auto& missionInfo = ccuExDetailInfo.missionInfo[i];

分析:
`ccuMissionNum` 来自 runtime 异常信息，属于外部数据。`missionInfo` 数组的容量上界是 `FUSION_SUB_TASK_MAX_CCU_NUM`（值为 8，已确认见 `rt_external_base.h:227`）。如果 runtime 返回的 `ccuMissionNum` 大于 8，循环会越界访问。同模块 `coll_service_device_mode.cpp:339` 已有类似的上界检查（`taskParams.size() > FUSION_SUB_TASK_MAX_CCU_NUM`），此处应保持一致。`ProcessCcuException` 的注释写"ccuMissionNum 为 1"，但缺乏代码强制约束。

修复建议:
在循环前增加上界保护：

    const uint32_t missionNum = std::min(static_cast<uint32_t>(ccuExDetailInfo.ccuMissionNum),
                                          static_cast<uint32_t>(FUSION_SUB_TASK_MAX_CCU_NUM));
    for (uint32_t i = 0; i < missionNum; ++i) {

---

### #4 [一般] PrintPanicLogInfo 在 query_result 失败后仍继续打印可能无效的数据

- 位置: `src/legacy/framework/dfx/task_exception/task_exception_handler.cpp:321`
- 规则: 2.1.3（逻辑合理性）
- 置信度: 待确认（取决于 query_result != 0 时其余字段是否仍然有意义）

问题代码:

    if (info->query_result != 0) {
        HCCL_ERROR("get ccu dfx info fail, ccu dfx info not all correct");
    }

分析:
当 `query_result` 表示查询失败时，结构体中其余字段（各计数器值）可能未被正确填充。但函数仅打印一条 error 后继续输出所有 DFX 计数器，输出的可能是无效数据，造成误导。如果查询失败时数据确实无效，应 `return` 提前退出。

修复建议:

    if (info->query_result != 0) {
        HCCL_ERROR("get ccu dfx info fail, ccu dfx info not all correct");
        return;
    }

---

### #5 [一般] 结构体命名违反 PascalCase 规范

- 位置: `src/legacy/framework/dfx/task_exception/task_exception_handler.cpp:303`
- 规则: 1.1.1（类/结构体使用大驼峰命名）
- 置信度: 确定

问题代码:

    struct ccum_dfx_info {

分析:
CANN 编码规范要求结构体名使用大驼峰（PascalCase）。`ccum_dfx_info` 使用了全小写下划线风格。

修复建议:

    struct CcumDfxInfo {

---

### #6 [一般] 测试桩函数签名未同步更新

- 位置: `test/legacy/ut/unified_platform/ccu/ccu_dfx/ut_ccu_dfx.cpp:66`
- 规则: 2.1.3（冗余/不一致代码）
- 置信度: 确定（已确认 `GetCcuErrorMsg` 签名已改为 5 参数，但 stub 仍为 3 参数，见 `ccu_dfx.h` 和 `ccu_error_handler.h`）

问题代码:

    void GetCcuErrorMsgExcptionStub(int32_t deviceId, const ParaCcu &ccuTaskParam, vector<CcuErrorInfo> &errorInfo)

分析:
`GetCcuErrorMsg` 的签名已更新为 5 参数（增加 `uint16_t status, uint16_t instrId`），但 `GetCcuErrorMsgExcptionStub` 仍保留旧的 3 参数签名。虽然当前测试未通过 MOCKER 注册此 stub（相关 mock 已在 diff 中移除），但该函数作为遗留代码签名不一致，后续维护者如果尝试复用会遇到编译错误。另外函数名 `Excption` 拼写也有误（应为 `Exception`）。

修复建议:
要么删除该未使用的 stub 函数，要么更新签名：

    void GetCcuErrorMsgExceptionStub(int32_t deviceId, uint16_t missionStatus, uint16_t currIns,
                                     const ParaCcu &ccuTaskParam, vector<CcuErrorInfo> &errorInfo)

---

### #7 [建议] 多处混合 Tab 和空格缩进

- 位置: `src/legacy/framework/dfx/task_exception/task_exception_handler.h:43`，`src/legacy/framework/dfx/task_exception/task_exception_handler.cpp:332, 352, 428`
- 规则: 1.2.1（缩进一致性）
- 置信度: 确定

问题代码:

    // task_exception_handler.h:43 — 行首 空格+Tab 混合
     	static void PrintCcuErrorInfo(uint32_t deviceId, uint16_t status, uint16_t instrId, const TaskInfo& taskInfo);

    // task_exception_handler.cpp:332 — 空行含 空格+Tab+空格 的尾随空白
    // task_exception_handler.cpp:352 — 行首 空格+Tab+多空格 缩进
    // task_exception_handler.cpp:428 — 函数定义行首多余空格

分析:
项目统一使用空格缩进，这几处引入了 Tab 字符或多余前导空格，应统一清理。

修复建议:
将 Tab 替换为空格，移除多余的前导/尾随空白。

---

## 总结

本 PR 的核心重构方向合理——将 `missionStatus` 和 `instrId` 从被调方寄存器查询改为由 runtime 异常信息直接传入，简化了数据流。新增的 `PrintPanicLogInfo` 增强了 CCU 异常的诊断能力。

建议优先处理 3 个严重问题：格式字符串缺少空格（确定）导致诊断日志不可读；不必要的 const_cast（确定）违反 const 正确性；循环缺少上界检查（待确认）可能造成数组越界。
