import os
import sys
import numpy as np
import tensorflow as tf
import cv2 as cv
import serial
import time
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



# Label
Label = None

# frozen_inference_graph의 경로
PATH_TO_CKPT = '/home/junsoofeb/py_project/traffic_light_detection/frozen_inference_graph.pb'

# label_map.pbtxt의 경로 
PATH_TO_LABELS = '/home/junsoofeb/py_project/traffic_light_detection/TL_label_map.pbtxt'

# label_map.pbtxt의 class 개수 
NUM_CLASSES = 1

# 학습된 모델에 넣을 이미지 경로, 이미지는 target.jpg로 저장되어 매 frame 마다 덮어 쓰여진다.
image_path = '/home/junsoofeb/py_project/traffic_light_detection/img/000054.jpg'
target_img = None

# 출력 이미지의 크기, inch단위
IMAGE_SIZE = (12, 8)


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def show(image, win_name = ""):
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imshow(win_name, image)
    cv.waitKey()
    cv.destroyAllWindows()
    


def detect_objects(image_np, sess, detection_graph):
    global Label
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})


   
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    
    
    # 찾은 레이블 저장, 못찾았을 시 재시도
    Label = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
    if Label == []:
        print("retry!")
        main()
    print(Label)
    Label = Label[0]['name']
    
    return image_np

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)



def check_target():
    global image_path

    LABEL = None
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_process = detect_objects(image_np, sess, detection_graph)
                
                show(image_process)
                
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

def main():
    check_target()


main()
    
'''
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                result = [category_index.get(i) for i in classes[0]][0]['name']
                print(result)
'''










    
