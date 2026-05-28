from manim import *

class Example3DirectorLoop(Scene):
    def construct(self):
        # --- 1. 基础设施 ---
        time_tracker = ValueTracker(0)
        
        # 模拟你的物理对象：一个上下浮动的圆
        physics_obj = Circle().set_color(BLUE)
        physics_obj.add_updater(lambda m: m.shift(UP * 0.1 * np.sin(time_tracker.get_value() * PI)))
        self.add(physics_obj)
        
        # 模拟你的 Annotation Mobjects
        anno_A = Text("A").shift(UP * 2)
        anno_B = Text("B").shift(UP * 2)
        anno_C = Text("C").shift(UP * 2)
        
        # --- 2. 定义动作列表 (Action List) ---
        # 这就像剧本：[时间, 动作类型, 参数, 持续时长]
        actions = [
            (1.0, "fade_in", anno_A, 0.5),      # 1秒时，A淡入
            (3.0, "transform", (anno_A, anno_B), 1.0), # 3秒时，A变B
            (5.0, "transform", (anno_B, anno_C), 0.5), # 5秒时，B变C
            (7.0, "fade_out", anno_C, 0.5),     # 7秒时，C淡出
        ]
        
        # --- 3. 导演循环 ---
        current_time = 0.0
        total_duration = 8.0
        
        for action in actions:
            t_trigger, action_type, params, duration = action
            
            # A. 推进物理时间到事件发生点
            dt = t_trigger - current_time
            if dt > 0:
                self.play(
                    time_tracker.animate.set_value(t_trigger),
                    run_time=dt,
                    rate_func=linear
                )
                current_time = t_trigger
            
            # B. 执行事件 (同时推进物理时间)
            if action_type == "fade_in":
                self.play(
                    time_tracker.animate.set_value(current_time + duration),
                    FadeIn(params),
                    run_time=duration,
                    rate_func=linear
                )
                
            elif action_type == "fade_out":
                self.play(
                    time_tracker.animate.set_value(current_time + duration),
                    FadeOut(params),
                    run_time=duration,
                    rate_func=linear
                )
                
            elif action_type == "transform":
                src, dst = params
                # FadeTransform 会自动处理 FadeOut(src) 和 FadeIn(dst)
                self.play(
                    time_tracker.animate.set_value(current_time + duration),
                    FadeTransform(src, dst),
                    run_time=duration,
                    rate_func=linear
                )
            
            # 更新当前时间指针
            current_time += duration

        # C. 播放剩余时间
        if current_time < total_duration:
            self.play(
                time_tracker.animate.set_value(total_duration),
                run_time=total_duration - current_time,
                rate_func=linear
            )
