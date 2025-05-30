from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


class YOLOv5Detector:
    def __init__(self, model_path="yolov5n.pt", conf_thres=0.25):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        results = self.model(
            source=image,
            conf=self.conf_thres,
            verbose=False
        )[0]

        return_boxes = []

        # 获取类别ID列表
        class_ids = results.boxes.cls.cpu().numpy().astype(int).tolist()
        # 遍历每个检测框及其对应的类别ID
        for box, cls_id in zip(results.boxes.xywhn.tolist(), class_ids):
            # 通过类别ID获取类别名称
            class_name = results.names[cls_id]

            # 仅处理"bottle"类别
            if class_name == "bottle":
                # 坐标转换
                x = int(box[0] * image.width)
                y = int(box[1] * image.height)
                w = int(box[2] * image.width)
                h = int(box[3] * image.height)

                # 处理越界
                x = max(0, x)
                y = max(0, y)
                w = max(0, w)
                h = max(0, h)

                return_boxes.append([x, y, w, h])

        return return_boxes

    @staticmethod
    def letterbox_image(image, size):
        """实现与YOLOv3一致的letterbox缩放（等比例缩放+黑边填充）"""
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image


# 主函数中替换初始化逻辑
if __name__ == '__main__':
    # 初始化 YOLOv8 检测器
    yolo_detector = YOLOv5Detector(model_path="yolov8n.pt", class_id=27, conf_thres=0.6)
    # main(yolo_detector)  # 传入检测器实例
