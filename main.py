#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import tensorflow as tf


backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default=0)
ap.add_argument("-c", "--class", help="name of class", default="person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")
crossed_line = dict()
masked = dict()

line_x = 320


def main(yolo):


    start = time.time()
    # Definition of the parameters
    max_cosine_distance = 0.5  # 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3  # 非极大抑制的阈值

    counter = []
    # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_path = "./output/output.avi"
    video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vidout = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()
        line_x = int(frame.shape[1]/2)
        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            subimg = frame[int(max(bbox[1], 0)): int(bbox[3]), int(max(0,bbox[0])): int(bbox[2])]
            inp = subimg[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            if num_detections > 0:
                classId = int(out[3][0][0])
                if classId == 1:
                    masked[track.track_id] = True
                elif classId == 2:
                    masked[track.track_id] = False
                elif track.track_id not in masked.keys():
                    masked[track.track_id] = None
            mask_str = ''

            if masked[track.track_id]:
                mask_str = 'mask'
            elif not masked[track.track_id]:
                mask_str = 'no mask'
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, (color), 2)
            if len(class_names) > 0:
                class_name = class_names[0]
                cv2.putText(frame, str(mask_str), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color), 2)

            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # track_id[center]
            pts[track.track_id].append(center)
            if len(pts[track.track_id]) > 1:
                left_side = False
                right_side = False
                for pt in pts[track.track_id]:
                    if pt[0] < line_x:
                        left_side = True
                    else:
                        right_side = True
                    if left_side and right_side and track.track_id not in crossed_line.keys():
                        crossed_line[track.track_id] = True
                        break

            # thickness = 5
            # center point
            # cv2.circle(frame, (center), 1, color, thickness)

            # draw motion path
            # for j in range(1, len(pts[track.track_id])):
            #     if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
            #         continue
            #     thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            #     cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)
            #     # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        count = len(set(counter))
        count_line_crossed = list(crossed_line.values()).count(True)
        count_masked = list(masked.values()).count(True)
        count_no_masked = list(masked.values()).count(False)
        cv2.putText(frame, "Total Person Counter: " + str(count_line_crossed), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0), 2)
        # cv2.putText(frame, "Current Person Counter: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        # cv2.putText(frame, "Masked: {} No-Masked: {}".format(count_masked, count_no_masked), (20, 140), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (250, 250, 250), 2)
        cv2.namedWindow("YOLO3_Deep_SORT", 0);
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
        cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            # save a frame
            vidout.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps = (fps + (1. / (time.time() - t1))) / 2
        # print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
        print(' Found')

    else:
        print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        vidout.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with tf.gfile.FastGFile(os.path.join('model_fmd', 'frozen_inference_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        main(YOLO())
