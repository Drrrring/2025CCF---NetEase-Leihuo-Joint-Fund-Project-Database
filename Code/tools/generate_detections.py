# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf


# 原来的
# def _run_in_batches(f, data_dict, out, batch_size):
#     data_len = len(out)
#     num_batches = int(data_len / batch_size)
#
#     s, e = 0, 0
#     for i in range(num_batches):
#         s, e = i * batch_size, (i + 1) * batch_size
#         batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
#         out[s:e] = f(batch_data_dict)
#     if e < len(out):
#         batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
#         out[e:] = f(batch_data_dict)
# 新的增加了logits
def _run_in_batches(f, data_dict, out_list, batch_size):
    """
    支持多输出的批量处理函数

    Args:
        f: 返回多个结果的函数（如返回 (features, logits) 的元组）
        data_dict: 输入数据字典（如 {input_var: data_x}）
        out_list: 输出数组列表（如 [out_features, out_logits]）
        batch_size: 批次大小
    """
    data_len = len(out_list[0])  # 以第一个输出的长度为准
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        batch_res = f(batch_data_dict)  # 返回多个结果（如 (features_batch, logits_batch)）

        # 将每个结果填充到对应的输出数组中
        for arr, res in zip(out_list, batch_res):
            arr[s:e] = res

    # 处理剩余数据（当 data_len % batch_size != 0 时）
    if e < data_len:
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        batch_res = f(batch_data_dict)
        for arr, res in zip(out_list, batch_res):
            arr[e:] = res


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

# 原来的
# class ImageEncoder(object):
#
#     def __init__(self, checkpoint_filename, input_name="images",
#                  output_name="features"):
#         self.session = tf.Session()
#         with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
#             graph_def = tf.GraphDef()
#             graph_def.ParseFromString(file_handle.read())
#         tf.import_graph_def(graph_def, name="net")
#         self.input_var = tf.get_default_graph().get_tensor_by_name(
#             "net/%s:0" % input_name)
#         self.output_var = tf.get_default_graph().get_tensor_by_name(
#             "net/%s:0" % output_name)
#
#         assert len(self.output_var.get_shape()) == 2
#         assert len(self.input_var.get_shape()) == 4
#         self.feature_dim = self.output_var.get_shape().as_list()[-1]
#         self.image_shape = self.input_var.get_shape().as_list()[1:]
#
#     def __call__(self, data_x, batch_size=32):
#         out = np.zeros((len(data_x), self.feature_dim), np.float32)
#         _run_in_batches(
#             lambda x: self.session.run(self.output_var, feed_dict=x),
#             {self.input_var: data_x}, out, batch_size)
#         return out


# 新的加上logits
class ImageEncoder(object):
    def __init__(self, checkpoint_filename, input_name="images",
                 output_names=["features", "logits"]):  # 关键修改：接受多个输出名
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")

        # 输入节点
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            f"net/{input_name}:0")

        # 输出节点（多输出）
        self.output_vars = [
            tf.get_default_graph().get_tensor_by_name(f"net/{name}:0")
            for name in output_names
        ]

        # 验证维度
        assert len(self.output_vars[0].get_shape()) == 2  # features
        assert len(self.output_vars[1].get_shape()) == 2  # logits
        assert len(self.input_var.get_shape()) == 4

        # 保存特征维度和图像形状
        self.feature_dim = self.output_vars[0].get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        # 初始化输出数组（features和logits）
        out_features = np.zeros((len(data_x), self.feature_dim), np.float32)
        out_logits = np.zeros((len(data_x), self.output_vars[1].shape[1]), np.float32)

        # 定义批量运行函数
        def _run_fn(feed_dict):
            return self.session.run(self.output_vars, feed_dict=feed_dict)

        # 批量执行
        _run_in_batches(
            _run_fn,
            {self.input_var: data_x},
            [out_features, out_logits],  # 同时填充两个数组
            batch_size
        )

        return out_features, out_logits  # 返回元组


# 原来的
# def create_box_encoder(model_filename, input_name="images",
#                        output_name="features", batch_size=32):
#     image_encoder = ImageEncoder(model_filename, input_name, output_name)
#     image_shape = image_encoder.image_shape
#
#     def encoder(image, boxes):
#         image_patches = []
#         for box in boxes:
#             patch = extract_image_patch(image, box, image_shape[:2])
#             if patch is None:
#                 print("WARNING: Failed to extract image patch: %s." % str(box))
#                 patch = np.random.uniform(
#                     0., 255., image_shape).astype(np.uint8)
#             image_patches.append(patch)
#         image_patches = np.asarray(image_patches)
#         return image_encoder(image_patches, batch_size)
#
#     return encoder
# 新的增加了logtis
def create_box_encoder(model_filename, input_name="images",
                      output_names=["features", "logits"],  # 与.pb文件中的基础名称一致
                      batch_size=32):
    image_encoder = ImageEncoder(
        model_filename,
        input_name=input_name,
        output_names=output_names
    )
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print(f"WARNING: Failed to extract image patch: {box}.")
                patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        features, logits = image_encoder(image_patches, batch_size)
        return features, logits

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--Model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


def inspect_pb_model(model_path):
    with tf.Session() as sess:
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="net")
        tensor_names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        print("输出节点名称:")
        print([name for name in tensor_names if "features" in name or "logits" in name])


if __name__ == "__main__":
    # main()
    inspect_pb_model("../../Model/reid_logits.pb")