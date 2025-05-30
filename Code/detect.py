import cv2
import os
from ultralytics import YOLO
import time
import argparse

def detect_objects_in_video(video_path, model_path='yolov5lu.pt', output_path='results/output.mp4',
                            conf_threshold=0.25, show_video=True, save_video=True, fps=None):
    """
    使用 YOLOv8 模型对视频进行目标检测

    参数:
    - video_path: 输入视频路径，设为0表示使用摄像头
    - model_path: 模型路径，默认为预训练的 YOLOv8 Nano 模型
    - output_path: 输出视频路径
    - conf_threshold: 置信度阈值，过滤低置信度检测结果
    - show_video: 是否显示视频
    - save_video: 是否保存视频
    - fps: 输出视频的帧率，如果为None则使用输入视频的帧率
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载模型
    model = YOLO(model_path)

    # 打开视频文件或摄像头
    if video_path == 0:
        print("正在使用摄像头进行实时检测...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"正在处理视频: {video_path}")
        cap = cv2.VideoCapture(video_path)

    # 获取视频的宽度、高度和原始帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频尺寸: {width}x{height}, 帧率: {fps:.2f} FPS")

    # 初始化视频写入器
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 模型推理
            results = model(frame, conf=conf_threshold)

            # 获取检测结果并绘制
            result = results[0]
            annotated_frame = result.plot()

            # 显示当前帧率
            current_time = time.time()
            fps_current = frame_count / (current_time - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps_current:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 保存视频帧
            if save_video:
                out.write(annotated_frame)

            # 显示结果
            if show_video:
                cv2.imshow('YOLOv5', annotated_frame)
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # 释放资源
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()

    # 输出统计信息
    total_time = time.time() - start_time
    print(f"处理完成! 共处理 {frame_count} 帧，耗时 {total_time:.2f} 秒")
    print(f"平均 FPS: {frame_count / total_time:.2f}")
    if save_video:
        print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv85lu')
    parser.add_argument('--video', type=str, default='cargo_videos/4124382722695262447.mp4',
                        help='输入视频路径，设为0表示使用摄像头')
    parser.add_argument('--Model', type=str, default='Model/yolov5lu.pt',
                        help='模型路径')
    parser.add_argument('--output', type=str, default='results/output.mp4',
                        help='输出视频路径')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示视频')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存视频')
    parser.add_argument('--fps', type=float, default=None,
                        help='输出视频的帧率')
    args = parser.parse_args()

    # 检查是使用摄像头还是视频文件
    video_source = 0 if args.video == '0' else args.video

    detect_objects_in_video(
        video_path=video_source,
        model_path=args.model,
        output_path=args.output,
        conf_threshold=args.conf,
        show_video=not args.no_display,
        save_video=not args.no_save,
        fps=args.fps
    )
