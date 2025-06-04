#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import argparse
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

from Code.deep_sort import preprocessing
from Code.deep_sort import nn_matching
from Code.deep_sort.detection import Detection
from Code.deep_sort.tracker import Tracker
from Code.tools import generate_detections as gdet

from Code.yoloDetector import YOLOv5Detector

warnings.filterwarnings('ignore')
class_names = ['molihuacha', 'binghongcha', 'nongfushanquan', 'dongfangshuye']


def get_class_name(logit, cls_names=None):
    if cls_names is None:
        cls_names = class_names
    if logit is None:
        return "unknown"
    class_id = np.argmax(logit)  # 取 logits 最大值的索引
    return cls_names[class_id]


def main(yolo, video_path, ground_truth_class=None, show_all_frames=False):
    # 按帧统计指标
    frame_stats = {
        'total': 0,  # 总帧数
        'detection_frames': 0,  # 有检测结果的帧数
        'correct_frames': 0,  # 正确识别的帧数
        'false_alarm_frames': 0,  # 误检的帧数
        'class_history': []  # 每帧的识别结果
    }

    # 记录每个类别的出现帧数
    class_frame_counts = defaultdict(int)

    # 记录误检的类别及其出现帧数
    misdetected_classes = defaultdict(int)

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = './Model/reid_logits.pb'  # DeepSort的模型文件
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture(video_path)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频

    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    if writeVideo_flag:
        w = int(video_capture.get(3))  # 在视频流的帧的宽度
        h = int(video_capture.get(4))  # 在视频流的帧的高度
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('./Code/cargo_result.mp4', fourcc, 15, (w, h))

    fps = 0.0
    frame_idx = 0

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if not ret:
            break
        frame_idx += 1
        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)

        features, logits = encoder(frame, boxs)

        detections = [Detection(bbox, 0.5, feature, logit) for bbox, feature, logit in zip(boxs, features, logits)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # 当前帧的检测结果
        frame_classes = set()

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # 获取类别
            c_name = get_class_name(track.logit, class_names)

            # 记录当前帧的所有检测类别
            frame_classes.add(c_name)

            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, f"ID:{track.track_id} {c_name}", (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # 更新帧统计
        frame_stats['total'] += 1
        frame_stats['class_history'].append((frame_idx, frame_classes.copy()))

        # 记录每个类别的出现帧数
        for cls in frame_classes:
            class_frame_counts[cls] += 1

        # 评估当前帧的检测结果
        if ground_truth_class and len(frame_classes) > 0:
            # 只评估有检测结果的帧
            frame_stats['detection_frames'] += 1

            # 正确识别：检测到目标类别
            if ground_truth_class in frame_classes:
                frame_stats['correct_frames'] += 1

            # 误检：检测到任何非目标类别的情况
            false_alarms = [cls for cls in frame_classes if cls != ground_truth_class]
            if false_alarms:
                frame_stats['false_alarm_frames'] += 1
                # 记录误检的类别
                for cls in false_alarms:
                    misdetected_classes[cls] += 1

        cv2.namedWindow("resized", 0);
        cv2.resizeWindow("resized", 1500, 940);
        cv2.imshow("resized", frame)

        if writeVideo_flag:
            out.write(frame)

        fps = (fps + (1. / (time.time() - t1))) / 2

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()

    print("\n===== 按帧评估结果 =====")
    print(f"总帧数: {frame_stats['total']}")
    print(f"有检测结果的帧数: {frame_stats['detection_frames']}")

    if ground_truth_class:
        # 计算精确率（只考虑有检测结果的帧）
        tp = frame_stats['correct_frames']  # 正确检测的帧数
        fp = frame_stats['false_alarm_frames']  # 误检的帧数

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"\n按帧统计:")
        print(f"  正确识别: {frame_stats['correct_frames']} 帧")
        print(f"  误检: {frame_stats['false_alarm_frames']} 帧")

        print(f"\n精确率 (Precision): {precision:.4f}")

        # 输出检测到的类别及其出现帧数
        print("\n===== 检测到的类别 =====")
        print(f"真实类别: {ground_truth_class}")
        print(f"真实类别出现: {class_frame_counts.get(ground_truth_class, 0)} 帧")

        print("\n误检类别:")
        if misdetected_classes:
            for cls, count in sorted(misdetected_classes.items(), key=lambda x: x[1], reverse=True):
                print(f"- {cls}: {count} 帧")
        else:
            print("无")

        # 输出类别分布详情
        if show_all_frames:
            print("\n===== 类别分布详情 =====")
            for frame_id, classes in frame_stats['class_history']:
                if classes:  # 只显示有检测结果的帧
                    status = "✅正确" if ground_truth_class in classes else "❌错误"
                    print(f"帧 {frame_id}: {', '.join(sorted(classes))} ({status})")

        # 分析类别切换情况
        class_change_count = 0
        prev_classes = None

        for frame_id, classes in frame_stats['class_history']:
            if prev_classes is not None and classes != prev_classes:
                class_change_count += 1
            prev_classes = classes

        # print(f"\n类别切换次数: {class_change_count}")
        # print(f"平均每 {frame_stats['total'] / (class_change_count + 1):.2f} 帧发生一次类别切换")


if __name__ == '__main__':
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='饮料多目标追踪系统')
    parser.add_argument('--video_path', type=str, required=True,
                        help='输入视频文件路径')
    parser.add_argument('--Model', type=str, default="./Model/yolov5lu.pt",
                        help='YOLO模型路径 (默认: "./Model/yolov5lu.pt")')
    parser.add_argument('--ground_truth', type=str, default=None,
                        help='真实出现的饮料类别 (如: "molihuacha")')
    parser.add_argument('--show_all', action='store_true',
                        help='显示所有帧的类别分布详情')

    args = parser.parse_args()

    print(f"开始处理视频: {args.video_path}")
    print(f"使用模型: {args.Model}")

    # 解析真实类别
    ground_truth_class = args.ground_truth.strip() if args.ground_truth else None

    if ground_truth_class and ground_truth_class not in class_names:
        print(f"警告: 指定的类别 {ground_truth_class} 不在预定义类别列表中")

    yolo_detector = YOLOv5Detector(model_path=args.Model)
    main(yolo_detector, args.video_path, ground_truth_class, args.show_all)
    end = datetime.datetime.now()
    print('running time:%s Seconds' % (end - start))
