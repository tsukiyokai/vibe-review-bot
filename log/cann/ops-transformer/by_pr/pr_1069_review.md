# Code Review: PR #1069

| 属性 | 值 |
|------|------|
| 标题 | [WIP]UT for AICPU data plane |
| 作者 | p_ch |
| 链接 | [https://gitcode.com/cann/hcomm-dev/merge_requests/1069](https://gitcode.com/cann/hcomm-dev/merge_requests/1069) |
| 审查时间 | 2026-02-24 10:30:48 |
| 审查工具 | Claude Code (`codereview` skill) |

---

## 变更概述

本 MR（标题为 "[WIP]UT for AICPU data plane"）仅修改了 `test/ut/CMakeLists.txt`，将 12 个现有 UT 子目录的 `add_subdirectory()` 调用全部注释掉。未新增任何代码或测试目录。

涉及 1 个文件，12 行删除 / 12 行新增（纯注释化）。

## 审查发现

共发现 2 个问题（严重 1 / 一般 1）

---

### #1 [严重] 注释掉全部现有 UT 子目录，合入将导致 UT 构建为空

- 位置: `test/ut/CMakeLists.txt:253-264`
- 规则: 工程质量 — 测试完整性
- 置信度: 确定

问题代码:

    # add_subdirectory(misc)
    # add_subdirectory(impl)
    # add_subdirectory(device)
    # add_subdirectory(platform/hcom)
    # add_subdirectory(platform/resource/dispatcher)
    # add_subdirectory(platform/resource/notify)
    # add_subdirectory(platform/resource/transport)
    # add_subdirectory(platform/ping_mesh)
    # add_subdirectory(framework/communicator)
    # add_subdirectory(framework/op_base_api)
    # add_subdirectory(framework/communicator/impl)
    # add_subdirectory(framework/resource)

分析: 这 12 个 `add_subdirectory` 涵盖了 misc、impl、device、platform、framework 等全部现有 UT 模块。全部注释掉后，`bash build.sh --ut` 将不编译任何测试用例，CI 的 UT 门禁形同虚设。PR 标题声称是 "UT for AICPU data plane"，但未新增任何 AICPU 相关测试目录或代码，变更内容与 PR 意图不符。

修复建议: 恢复所有被注释的 `add_subdirectory` 调用。如果新增的 AICPU data plane UT 有编译依赖冲突，应仅注释掉冲突的特定目录并在 commit message 中说明原因，而非全部禁用。待新 UT 代码就绪后一并提交：

    add_subdirectory(misc)
    add_subdirectory(impl)
    add_subdirectory(device)
    add_subdirectory(platform/hcom)
    add_subdirectory(platform/resource/dispatcher)
    add_subdirectory(platform/resource/notify)
    add_subdirectory(platform/resource/transport)
    add_subdirectory(platform/ping_mesh)
    add_subdirectory(framework/communicator)
    add_subdirectory(framework/op_base_api)
    add_subdirectory(framework/communicator/impl)
    add_subdirectory(framework/resource)
    add_subdirectory(framework/next/aicpu_dataplane)  # 新增 AICPU UT

---

### #2 [一般] WIP 状态的调试性变更不应提交合入请求

- 位置: `test/ut/CMakeLists.txt:253-264`
- 规则: 工程规范 — 提交质量
- 置信度: 确定

问题代码:

    # add_subdirectory(misc)
    ...
    # add_subdirectory(framework/resource)

分析: PR 标题带有 `[WIP]` 标记，描述仅为 "wip: edit CMake"。注释掉全部测试目录是典型的本地调试手段（排除编译错误以聚焦新代码），不应作为独立 MR 提交到 master。这类变更如果被误合入，会导致所有 UT 失效而无人察觉。

修复建议: 将此 MR 标记为 Draft/WIP 状态（禁止合入），待 AICPU data plane 的 UT 代码开发完成后，与新增测试代码一起提交完整的 MR，确保：(1) 现有 UT 目录全部保留；(2) 新增 UT 目录正常编译通过。

---

## 总结

本 MR 的实质是开发过程中的调试性修改——为了让新代码编译通过而临时注释掉全部现有 UT。这不应合入 master。建议作者完成 AICPU data plane UT 的开发后，恢复所有被注释的测试目录，连同新测试代码一起提交完整的 MR。
