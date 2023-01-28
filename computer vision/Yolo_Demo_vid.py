#######################
#       YOLOv3        #
#######################

import numpy as np
import cv2

# we are not going to bother with objects less than 30% prob
THRESHOLD = 0.3
SUPPRESSION_THRESHOLD = 0.3
YOLO_IMAGE_SIZE = 320

def find_objects(model_outputs):

    bounding_box_locations = []
    class_ids = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            if confidence > THRESHOLD:
                w, h = int(prediction[2]*YOLO_IMAGE_SIZE), int(prediction[3]*YOLO_IMAGE_SIZE)
                x, y = int(prediction[0]*YOLO_IMAGE_SIZE-w/2), int(prediction[1]*YOLO_IMAGE_SIZE-h/2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)
    return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values

def show_detected_images(img, bounding_box_ids, all_bounding_boxes, class_ids, confidence_values, width_ratio, height_ratio):

    for index in bounding_box_ids:
        #bounding_box = all_bounding_boxes[index[0]]
        bounding_box = all_bounding_boxes[index]
        x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])

        # transform
        x = int(x*width_ratio)
        y = int(y*height_ratio)
        w = int(w*width_ratio)
        h = int(h*height_ratio)

        #if class_ids[index[0]] == 2:
        if class_ids[index] == 2:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'CAR' + str(int(confidence_values[index] * 100)) + "%"
            #class_with_confidence = 'CAR' + str(int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,0,0), 1)

        #if class_ids[index[0]] == 0:
        if class_ids[index] == 0:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'PERSON' + str(int(confidence_values[index] * 100)) + "%"
            #class_with_confidence = 'PERSON' + str(int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,0,0), 1)

        #if class_ids[index[0]] == 14:
        if class_ids[index] == 14:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'BENCH' + str(int(confidence_values[index] * 100)) + "%"
            #class_with_confidence = 'BENCH' + str(int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,0,0), 1)



video_path = "C:/Users/gunho/PycharmProjects/CV-ML-demonstration/data/videos/yolo_test.mp4"
capture = cv2.VideoCapture(video_path)

# there are 80 (90) possible output classes
# COCO dataset classes: after 2017, 90 classes are available.
# 0: person, 4: motorcycle, 2: car
classes = ['person', 'motorcycle', 'car']

config = "C:/Users/gunho/PycharmProjects/CV-ML-demonstration/weights/yolov3.cfg"
weights = "C:/Users/gunho/PycharmProjects/CV-ML-demonstration/weights/yolov3.weights"
NN = cv2.dnn.readNetFromDarknet(config, weights)
# run the algo with CPU or with GPU?
# we use CPU here. SELF NOTE: check its function with GPUs
NN.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
NN.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

##for video
while True:

    # after reading a frame we just have to do
    # the same operations as with single images
    frame_grabbed, frame = capture.read()

    if not frame_grabbed:
        break

    original_width, original_height = frame.shape[1], frame.shape[0]

    # the image into a BLOB (0-1) RGB - BGR
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True,
                                 crop=False)  # normalize pixel intensity and rescale
    NN.setInput(blob)  # input for the YOLO network

    layer_names = NN.getLayerNames()
    # YOLO has 3 output layer - note: the indexes are starting with 1
    # print(NN.getUnconnectedOutLayers())
    output_names = [layer_names[index - 1] for index in NN.getUnconnectedOutLayers()]

    outputs = NN.forward(output_names)
    predicted_objects, bbox_locations, class_label_ids, conf_values = find_objects(outputs)
    show_detected_images(image, predicted_objects, bbox_locations, class_label_ids, conf_values,
                         original_width / YOLO_IMAGE_SIZE, original_height / YOLO_IMAGE_SIZE)
    cv2.imshow('Yolo Algorithm', image)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()