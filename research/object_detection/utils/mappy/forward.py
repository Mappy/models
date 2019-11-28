from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import argparse
import matplotlib
import json
import numpy as np
import tensorflow as tf

import requests
from io import BytesIO

from datetime import datetime

from object_detection.utils.mappy.config import cfg
from object_detection.utils.mappy.api import get_pano_tiles, get_pano, push_detection, next_id
from object_detection.utils.mappy.timer import Timer
from object_detection.utils import visualization_utils as vis_util

from distutils.version import StrictVersion

from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '../../../../../mappy_trained_models/model_dir_tien/exported_graphs/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../../data', 'mappy_blur_label_map.pbtxt')

matplotlib.use('Agg')

DEBUG_SAVE_BLURED_PANO = False
PANO_IMAGE_WIDTH = 2048 * 6
TILE_IMAGE_WIDTH = 2048
TILE_IMAGE_HEIGHT = 2048
CAT_SHAPE_DICT = {1: 'circle', 2: 'rectangle'}


def forward(pano_id, tf_detection_graph):
    """ Detect object classes in an image using pre-computed object proposals.
        Returns: Result, time"""
    timer = Timer()
    timer.tic()

    results = []
    pano_boxes_and_classes = {}

    tiles = get_pano_tiles(pano_id)

    # for image_path in TEST_IMAGE_PATHS:
    # image = Image.open(image_path)
    for tile, index in tiles.items():
        print("{}, {}".format(tile, index))
        response = requests.get(tile, auth=requests.auth.HTTPBasicAuth(cfg.BO.id, cfg.BO.password))

        image = Image.open(BytesIO(response.content))

        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, tf_detection_graph)

        boxes_classes_map = boxes_and_classes(
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores']
        )

        for box, classe in boxes_classes_map.items():
            ymin, xmin, ymax, xmax = box
            (left, top, dx, dy) = ((xmin * TILE_IMAGE_WIDTH + index * TILE_IMAGE_WIDTH) / PANO_IMAGE_WIDTH,
                                   ymin,
                                   (xmax * TILE_IMAGE_WIDTH - xmin * TILE_IMAGE_WIDTH) / PANO_IMAGE_WIDTH,
                                   ymax - ymin)

            pano_boxes_and_classes[(left, top, dx, dy)] = classe

    if len(pano_boxes_and_classes) > 0:
        ####################
        #  for the debug
        if DEBUG_SAVE_BLURED_PANO:
            pano_image = get_pano(pano_id)
            pano_image = load_image_into_numpy_array(pano_image)

            for box, classe in pano_boxes_and_classes.items():
                left, top, dx, dy = box
                vis_util.draw_bounding_box_on_image_array(
                    pano_image,
                    top * TILE_IMAGE_HEIGHT,
                    left * PANO_IMAGE_WIDTH,
                    (top + dy) * TILE_IMAGE_HEIGHT,
                    (left + dx) * PANO_IMAGE_WIDTH,
                    thickness=4,
                    use_normalized_coordinates=False
                )

            cv2.imwrite('../../../../../tmp/{}.jpg'.format(pano_id), pano_image)
        ####################

        for box, classe in pano_boxes_and_classes.items():
            x, y, dx, dy = box
            results.append({"type": CAT_SHAPE_DICT.get(classe),
                            "x": x, "y": y, "dx": dx, "dy": dy})

    timer.toc()

    return results, timer.total_time


def parse_args():
    parser = argparse.ArgumentParser(description='Mappy Tensorflow Object Dectection Faster R-CNN Blur')
    parser.add_argument('--id', dest='id', default=None, help='Optional test on id')
    parser.add_argument('--ids', type=str, dest='ids', default=[], action='store', help='Optional test on a list of id')
    parser.add_argument('--limit', type=int, dest='limit', default=None, help='Optional test on id')
    return parser.parse_args()


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def boxes_and_classes(
        boxes,
        classes,
        scores,
        max_boxes_to_draw=20,
        min_score_thresh=.82):
    box_to_class_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            box_to_class_map[box] = classes[i]

    return box_to_class_map


if __name__ == '__main__':
    args = parse_args()

    im_id = args.id
    num_limit = args.limit

    logfile = '../../../../../logs/run_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(logfile)

    # Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    i = 0
    infinite_loop = True
    while infinite_loop:
        if not im_id:
            im_id = next_id()
        else:
            infinite_loop = False

        if im_id is False:
            break

        print(im_id)
        pano_annotations, det_time = forward(im_id, detection_graph)

        if len(pano_annotations) > 0:
            print("n {} : {}  |  {} sc | {} ".format(i, im_id, det_time, json.dumps(pano_annotations)))

            with open(logfile, 'a') as f:
                f.write("n {} : {}  |  {} sc\n".format(i, im_id, det_time))

            push_detection(pano_annotations, im_id)

        i += 1
        im_id = None

        if num_limit is not None and i >= num_limit:
            break

    print("End")
