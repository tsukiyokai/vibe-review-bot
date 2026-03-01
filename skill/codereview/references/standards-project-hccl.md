# 【项目级】HCCL项目规则

## 1 网络安全

### 规则 1.1 token认证类信息属于华为公司产品网络安全红线中规定的敏感信息，禁止打印。

### 规则 1.2 在UB芯片上，当前只需要关注tokenId和tokenValue不能打印。

### 规则 1.3 当前RDMA协议中使用的rkey和lkey信息不用于鉴权功能，仅作为索引辅助硬件找到注册的内存，因此不属于敏感信息。

## 2 HCCL项目级编码规则

- tokenId/tokenValue禁止入日志（网络安全红线）。RDMA rkey/lkey不属于敏感信息（避免误报）
- 返回值：用 `CHK_RET()` 检查，仅日志用 `CHK_PRT()`
- 日志：必须用 `HCCL_DEBUG`/`HCCL_INFO`/`HCCL_WARNING`/`HCCL_ERROR`/`HCCL_RUN_INFO`，禁止printf/cout
- 内存分配：堆上用 `NEW_NOTHROW`，智能指针用 `CHK_SMART_PTR_NULL()` 检查
- 错误上报：输入 `RPT_INPUT_ERR`，环境 `RPT_ENV_ERR`，内部 `RPT_INNER_ERR_PRT`，外调 `RPT_CALL_ERR`

## 3 高价值缺陷模式

HCCL实际审查中发现的高频严重缺陷，保持高度敏感：

1. sizeof(容器)误用：`sizeof(std::vector<T>)` = 对象大小（24），非数据大小。用 `.size()`
2. 值传递悬垂指针：`void F(Type p) { m_ = &p; }` — 值拷贝返回后销毁，m_悬垂
3. 秒转毫秒溢出：`uint32_t ms = seconds * 1000` — seconds > ~4.3M时溢出
4. 格式字符串不匹配：HCCL_INFO/ERROR中 `%` 占位符数/类型与实参不一致 = UB
5. 跨文件遗漏清理：删除结构体成员但未清理其他文件引用 → 内存布局错误。必须grep
6. 构造失败后析构崩溃：构造中途失败 → 析构清理未初始化成员（空指针delete）
7. 局部变量指针逃逸：返回局部变量指针/引用；lambda捕获局部变量但异步执行时已销毁
8. thread_local + dlclose：dlopen的.so中thread_local → dlclose时析构crash
9. gm偏移用int32：gm内存偏移/大小必须用int64，int32溢出
10. 整数不转浮点：可整数计算时禁止转浮点，必要时转更高精度整数类型
11. 通信算子融合同步缺失：多轮计算和集合通信之间需增加核间同步
12. 差一错误：循环边界、`\0`终结符、数组长度计算中的off-by-one
