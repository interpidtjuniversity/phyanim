## 实现方案

1. 修改 `phyanim/server/app.py`：
   - 新增 `GET /get_status?script_name=...`。
   - 复用现有 `script_name` 安全校验和持久化任务查询。
   - 已知任务统一返回 HTTP `200`，响应只包含：
     ```json
     {"status": "queued|generating|rendering|succeeded|failed"}
     ```
   - 参数缺失或非法返回 `400`，任务不存在返回 `404`。

2. 精简 `GET /get_video?script_name=...`：
   - 成功时只返回 `video/mp4` 视频流，不再承担状态查询职责。
   - 任务仍在 `queued/generating/rendering` 时返回 HTTP `409`，提示视频尚未就绪。
   - 任务失败时返回 HTTP `409`，不泄漏内部渲染错误。
   - 任务或成功状态对应的视频文件不存在时返回 `404`。
   - 参数错误仍返回 `400`；这些非成功响应保留简洁 JSON 错误体，`/get_video` 唯一的成功响应类型是视频流。

3. 保持 `GET /get_code` 现有行为不变；它仍可在代码未就绪时返回任务状态，本次只拆分 `get_video` 的职责。

4. 更新 `app.py` 顶部接口说明，加入 `/get_status`，并明确 `/get_video` 仅返回已完成视频。

5. 修改 `tests/test_server_api.py`：
   - 后台任务轮询改为调用 `/get_status`。
   - 覆盖五种状态均只返回 `{status: ...}`，且 HTTP 为 `200`。
   - 覆盖 `/get_status` 的非法参数和未知任务。
   - 覆盖 `/get_video` 对处理中、失败、文件缺失的 `409/404` 行为。
   - 保留并验证成功时返回 MP4 字节流。
   - 运行完整 pytest，并执行 `git diff --check`。