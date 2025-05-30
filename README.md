# 2025CCF-网易雷火联合基金课题数据库

回收课题相关数据，请注意文件中的readme和各目录要求：

code：存放项目源代码，按功能或模块组织，需安装依赖确保运行。

demo：提供样例数据和示例结果，含演示脚本与配置，可以速览项目功能。

model：存放训练好的模型文件，使用时要确保正确加载。

readme：即当前目录，项目文档说明，如运行指南、使用手册，使用前建议详读。

本项目为第14题：基于人机协作的无人货柜结算的实时质量控制算法研究，采用yolo+deepsort算法实现。由于时间关系，只实现了部分商品的识别和追踪。
## 1. 环境配置
```bash
conda create -n cargo python=3.7
conda activate cargo
pip install -r requirements.txt
```

## 2. 项目运行
```bash
# 下载yolo预训练模型（可选，若不手动下载程序也会自动下载）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt -O ./Model/yolov5lu.pt
python ./Demo/demo_cargo.py --video_path ./Code/cargo_videos/4124382499356963766.mp4
```
## 3. 输出结果
代码将根据输入的视频生成添加了商品追踪框的结果视频，保存在Code文件夹下，并在命令行中输出识别的商品类别。


同时针对课题详细情况，提醒您：​

1、预计5月底回收课题结果，请您提前准备，比赛结果实时评测，评测频率分为两个阶段，5月24日前，每周评测1-2次，5月24-5月底，每2天评测1次。

2、数据回收阶段，请各位老师提供Github账号，并将课题结果上传至github，我们将通过Github进行结果回收

3、Fork后请维护私有仓库，并邀请账号【Lac-bit】 加入仓库，后续我们将拉取最新提交的结果更新榜单成绩。
