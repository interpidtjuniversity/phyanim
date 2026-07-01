## 🧪 PhyAnim – 计算/存储/渲染分离的物理动画引擎

**后端**：`scipy` · `heyoka`（可选）

一个完全代码驱动的、事件导向的物理动画框架。  
你只需声明物理对象、方程和事件，引擎会自动求解并渲染出精确的科普动画。

---

### 🎬 预览

| 带电粒子在静电场中的弹性碰撞 (碰撞2次) | 水平动量守恒 & 能量守恒 |
|:---:|:---:|
| [![charged_particle_collision](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/charged_particle_collision.gif)](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/charged_particle_collision.mp4) | [![conservation_of_momentum](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/conservation_of_momentum.gif)](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/conservation_of_momentum.mp4) | [!
| *两个带正/负电粒子在匀强电场中运动并发生弹性碰撞* | *可移动半圆形轨道，系统水平动量与机械能守恒* | 

| 弹性碰撞| 弹簧摆|
|:---:|:---:|
| [![spring_collision](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/spring_collision.gif)](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/spring_collision.mp4) | [![spring_pendulum](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/spring_pendulum.gif)](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/spring_pendulum.mp4) | [!
| *两个小球的弹性碰撞（通过弹簧m1=2kg,m2=1kg）* | *弹簧摆* | 
### 📓 强大的事件驱动标注系统
| 弹性碰撞求解 | 
|:---:|
| [![spring_collision](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/release1_feature.gif)](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/release1_feature.mp4) | [!
| *支持公式展示，向量标注，文本标注，以及跟随移动* |
### physics🪡&&render🪡双线渲染线，physics线可暂停或缩放
| 弹性碰撞求解标注| 霍尔效应|
|:---:|:---:|
| [![release1_feature](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/release1_feature.gif?v=1)](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/release1_feature.mp4) | [![hall_effect](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/hall_effect.gif)](https://physic-video.oss-cn-beijing.aliyuncs.com/phyanim/1080p60/hall_effect.mp4) | [!
| *冻结碰撞瞬间并添加额外标注文本* |*瞬时受力分析*| 
