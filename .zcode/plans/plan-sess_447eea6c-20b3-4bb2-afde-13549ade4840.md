## 实现方案

### 一、接口契约

1. **修改 `POST /generate_video`**
   - 请求格式继续兼容现有的 `user_prompt` / `prompt` 与 `image_urls`。
   - 请求到达后立即生成一个不可碰撞且可安全用于文件名的唯一标识，例如：
     ```text
     Scene_20260726_153012_a1b2c3d4
     ```
   - 该值作为无扩展名的 `script_name`，同时用于：
     - Python 文件：`<media_root>/code/<script_name>.py`
     - Manim 场景类名和最终视频名：`<script_name>.mp4`
     - 持久化任务状态：`<media_root>/jobs/<script_name>.json`
   - 接口不再等待 LLM 或 Manim，创建后台任务后立即返回 HTTP `202`：
     ```json
     {
       "script_name": "Scene_...",
       "status": "queued"
     }
     ```
   - 修复当前 `user_id.strip()` 在字段缺失时产生未捕获 500 的问题；`user_id` 不再参与文件路径。对 JSON、prompt 和 `image_urls` 做明确校验。

2. **新增 `GET /get_video?script_name=...`**
   - 严格校验 `script_name`，拒绝路径分隔符、`..` 等路径穿越输入。
   - 根据持久化状态返回：
     - `queued` / `generating` / `rendering`：HTTP `202`，JSON 状态。
     - `failed`：HTTP `500`，返回稳定失败标识和经过清理的错误信息。
     - `succeeded`：在 `<media_root>/media/videos` 下递归查找文件名严格等于 `<script_name>.mp4` 的非 partial 文件，并用 `send_file` 返回 `video/mp4`。
     - 未知任务：HTTP `404`。
   - 即使状态标记成功，也会验证对应 MP4 确实存在；不会再采用“整个目录最新 MP4”，避免并发请求互相拿错视频。

3. **新增 `GET /get_code?script_name=...`**
   - 同样严格校验 `script_name`。
   - 代码尚未生成时返回 HTTP `202` 和任务状态。
   - 代码生成成功后读取 `<media_root>/code/<script_name>.py`，返回 HTTP `200`：
     ```json
     {
       "script_name": "Scene_...",
       "code": "..."
     }
     ```
   - 任务在代码生成前失败则返回 HTTP `500`；未知任务或代码文件不存在则返回 `404`。

### 二、后台任务与进程监控

4. **新增持久化任务管理器**（放在 `phyanim/server` 下的独立模块）
   - 使用进程内线程锁保护任务状态，并把每次状态变化原子写入 `<media_root>/jobs/<script_name>.json`（临时文件 + replace），避免查询读到半写文件。
   - 状态机：
     ```text
     queued -> generating -> rendering -> succeeded
                                  \-> failed
                     \-> failed
     ```
   - 每个任务由后台 daemon thread 执行完整流程：
     1. 调用 LLM planner 生成 render 代码。
     2. 将代码写入 `<script_name>.py`。
     3. 用 `subprocess.Popen` 启动独立 Python 渲染进程。
     4. 持久化 PID 和 `rendering` 状态。
     5. 调用 `communicate()` 监控退出；退出码为 0 且精确视频文件存在才记为 `succeeded`，否则记为 `failed`。
   - stdout/stderr 可记录到任务状态或日志中供诊断，但 API 只暴露截断、清理后的错误，不把完整内部输出直接返回客户端。
   - 保留现有同步 `render_code()` 的行为，抽取/新增可复用的“写脚本、构建环境、启动进程”辅助函数，避免破坏已有调用方。

5. **服务重启恢复**
   - 最终状态及代码/视频均通过 job JSON 和文件持久保留，重启后仍可查询。
   - 创建任务管理器时扫描已有 job JSON：
     - `succeeded` / `failed` 保持原状态。
     - 对重启时遗留的 `queued` / `generating` / `rendering`：若精确视频已经存在则恢复为 `succeeded`；否则标记为 `failed`，错误标识为任务被服务重启中断。
   - 说明：父进程退出后无法可靠重新取得原 `Popen` 的退出码和管道，因此不伪造“继续监控”；以磁盘产物恢复成功，否则明确记为中断失败。

### 三、现有代码调整

6. **`phyanim/server/app.py`**
   - app 创建时初始化一个任务管理器并保存在 `app.extensions`，供三个接口共享。
   - `generate_video` 只负责校验、提交任务并返回 ID。
   - `get_video`、`get_code` 只做安全查询和文件响应。
   - 保持现有 `/health`，可补充活动/持久化任务信息，但不改变已有字段。

7. **`phyanim/llm/runner.py`**
   - 抽取后台任务可复用的脚本写入、环境构建和 `Popen` 启动逻辑。
   - 现有 `render_code()` 继续通过公共逻辑同步等待并返回脚本路径，保持兼容。

8. **`phyanim/llm/planner.py`**
   - 修正 Windows 下 Manim media 路径注入的字符串转义，使用合法 Python 字面量，防止反斜杠路径破坏生成代码。
   - 将唯一 `script_name` 同时作为场景名，使 MP4 可按唯一标识精确查找。

### 四、测试

9. **新增 Flask API/任务管理测试**，所有外部 LLM 和 Manim 执行均 mock，不进行真实联网或视频渲染：
   - `generate_video` 立即返回 `202` 和唯一 `script_name`。
   - LLM 生成和渲染确实在后台发生。
   - 两个同一时刻请求不会发生脚本名、场景名或视频串线。
   - `get_video` 对 queued/generating/rendering、succeeded、failed、unknown 分别返回 `202/200/500/404`。
   - 成功时只返回精确 `<script_name>.mp4`，忽略更新但无关的视频及 partial 文件。
   - `get_code` 在未就绪、成功、失败、未知任务下的响应。
   - 非法 `script_name` 无法逃逸 code/jobs/video 目录。
   - job JSON 可以在新 manager 实例中恢复；中断状态按磁盘视频是否存在恢复为成功或失败。
   - `Popen` 非零退出码和“退出码为 0 但没有视频”都标记失败。
   - 保留并运行现有测试，确认同步 `render_code()` 兼容。