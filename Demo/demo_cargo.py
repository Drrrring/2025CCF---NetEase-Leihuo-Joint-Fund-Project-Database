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

from Code.deep_sort import preprocessing
from Code.deep_sort import nn_matching
from Code.deep_sort.detection import Detection
from Code.deep_sort.tracker import Tracker
from Code.tools import generate_detections as gdet

from Code.yoloDetector import YOLOv5Detector

warnings.filterwarnings('ignore')
class_names = ['molihuacha', 'binghongcha', 'nongfushanquan', 'wulongcha']


def get_class_name(logit, cls_names=None):
    if cls_names is None:
        cls_names = class_names
    if logit is None:
        return "unknown"
    class_id = np.argmax(logit)  # 取 logits 最大值的索引
    return cls_names[class_id]


def RealseMultiRecogCount(FrameCount, threshold):
    if len(FrameCount) != 0:
        FrameCount.sort()  #sort a in an ascending order
        item_count = 0  #the number of item in one frame
        for firstX in range(0, len(FrameCount) - 1):  #threshold is a int that define the extend of error
            secondX = firstX + 1
            if FrameCount[firstX][0] - FrameCount[secondX][0] < -threshold or FrameCount[firstX][0] - \
                    FrameCount[secondX][0] > threshold:
                if FrameCount[firstX][1] - FrameCount[secondX][1] < -threshold or FrameCount[firstX][1] - \
                        FrameCount[secondX][1] > threshold:
                    item_count += 1
        item_count += 1
        return item_count
    else:
        return 0


def main(yolo, video_path):

    appeared_beverages = set()  # 存储出现的饮料种类
    track_class_history = {}  # 存储每个track_id的类别历史

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = './Model/reid_logits.pb'  # DeepSort的模型文件
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture('./Code/cargo_videos/4124382722695262447.mp4')  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频

    if writeVideo_flag:
        w = int(video_capture.get(3))  #在视频流的帧的宽度
        h = int(video_capture.get(4))  #在视频流的帧的高度
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('./Code/cargo_result.mp4', fourcc, 15, (w, h))

        frame_index = -1
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3 按帧读取视频，ret,frame是获cap.read()方法的两个返回值。
        # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
        if not ret:
            break
        t1 = time.time()
        image = Image.fromarray(frame[..., ::-1])  #bgr to rgb
        boxs = yolo.detect_image(image)

        features, logits = encoder(frame, boxs)

        # score to 1.0 here). 原本置信度是1
        # detections = [Detection(bbox, 0.5, feature) for bbox, feature in zip(boxs, features)]
        # 加上logits
        detections = [Detection(bbox, 0.5, feature, logit) for bbox, feature, logit in zip(boxs, features, logits)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            # 获取类别
            c_name = get_class_name(track.logit, class_names)

            # 记录当前track的类别
            if track.track_id not in track_class_history:
                track_class_history[track.track_id] = []
            track_class_history[track.track_id].append(c_name)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, f"trackId: {track.track_id}, class: {c_name}", (int(bbox[0]), int(bbox[1])), 0,
                        5e-3 * 200, (0, 255, 0), 2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.namedWindow("resized", 0);
        cv2.resizeWindow("resized", 1500, 940);
        cv2.imshow("resized", frame)
        #cv2.imshow('', frame)

        if writeVideo_flag:
            out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1. / (time.time() - t1))) / 2

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    for track_id, class_list in track_class_history.items():
        # 取每个track_id中出现次数最多的类别作为最终类别
        most_common_class = max(set(class_list), key=class_list.count)
        appeared_beverages.add(most_common_class)

    print("\n视频中出现的饮料种类:")
    for beverage in appeared_beverages:
        print(f"- {beverage}")


if __name__ == '__main__':
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='饮料多目标追踪系统')
    parser.add_argument('--video_path', type=str, required=True,
                        help='输入视频文件路径')
    parser.add_argument('--Model', type=str, default="./Model/yolov5lu.pt",
                        help='YOLO模型路径 (默认: "./Model/yolov5lu.pt")')

    args = parser.parse_args()

    print(f"开始处理视频: {args.video_path}")
    print(f"使用模型: {args.Model}")
    yolo_detector = YOLOv5Detector(model_path=args.Model)
    main(yolo_detector, args.video_path)
    end = datetime.datetime.now()
    print('running time:%s Seconds' % (end - start))
