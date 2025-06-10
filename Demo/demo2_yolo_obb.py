import argparse
from ultralytics import YOLO
from collections import defaultdict
import numpy as np


def main():
    class_names = ['molihuacha', 'binghongcha', 'nongfushanquan', 'dongfangshuye']
    parser = argparse.ArgumentParser(description='饮料多目标追踪系统（YOLOv11-OBB）')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频路径')
    parser.add_argument('--Model', type=str, default="./Model/best.pt", help='模型路径')
    parser.add_argument('--ground_truth', type=str, default=None, help='真实类别（如: molihuacha）')
    parser.add_argument('--show_all', action='store_true', help='实时显示检测结果')

    args = parser.parse_args()
    print(f"视频路径: {args.video_path} | 模型: {args.Model}")

    # 初始化统计
    stats = {
        'total_frames': 0,  # 总帧数
        'detection_frames': 0,  # 有检测的帧数
        'correct_frames': 0,  # 正确识别帧数
        'class_counts': defaultdict(int),  # 各类别总出现次数
        'misdetect_counts': defaultdict(int)  # 误检类别统计
    }

    # 加载模型（YOLOv11-OBB）
    model = YOLO(args.Model)

    # 逐帧处理（通过enumerate获取帧索引）
    for frame_idx, result in enumerate(model.track(
            source=args.video_path,
            show=args.show_all,
            conf=0.5,
            save=True,
            persist=True  # 保持跟踪ID
    )):
        stats['total_frames'] += 1

        # 核心：从obb属性提取检测结果（适配YOLOv11-OBB）
        if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
            obb_data = result.obb.cpu().numpy()  # 获取OBB检测结果
            # class_ids = obb_data[:, 0].astype(int)  # 类别ID在第6列（索引5）
            class_ids = [int(obb.cls.item()) for obb in obb_data]
            # 提取当前帧的类别集合
            current_classes = set()
            for cls_id in class_ids:
                class_name = class_names[cls_id]
                current_classes.add(class_name)
                stats['class_counts'][class_name] += 1  # 统计类别总出现次数

            # 评估逻辑（仅处理有检测的帧）
            stats['detection_frames'] += 1
            if args.ground_truth:
                # 正确识别：包含真实类别
                if args.ground_truth in current_classes:
                    stats['correct_frames'] += 1
                # 误检：包含非真实类别的情况
                for cls in current_classes:
                    if cls != args.ground_truth:
                        stats['misdetect_counts'][cls] += 1

    # 计算精确率（仅在有检测帧时计算）
    precision = stats['correct_frames'] / stats['detection_frames'] if stats['detection_frames'] > 0 else 0.0

    # 输出结果
    print("\n===== 检测统计 =====")
    print(f"总帧数: {stats['total_frames']}")
    print(f"有检测的帧数: {stats['detection_frames']}")
    print(f"正确识别帧数: {stats['correct_frames']}")
    print(f"精确率: {precision:.4f}")

    print("\n===== 类别分布 =====")
    for cls, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"- {cls}: {count} 次")

    if args.ground_truth:
        print("\n===== 误检统计 =====")
        if stats['misdetect_counts']:
            for cls, count in sorted(stats['misdetect_counts'].items(), key=lambda x: x[1], reverse=True):
                print(f"- {cls}: {count} 次（误检）")
        else:
            print("无")


if __name__ == '__main__':
    main()
