import cv2
import numpy as np
from webapp import settings
from webapp.yolo import YOLOv8

# CHECK THE CURRENT WORKING DIRECTORY
settings.cwd()
model_path = settings.DETECTION_MODEL_ONNX
# print(settings.DATASETS_DIR.exists())

# Initialize YOLOv8 object detector
yolov8_detector = YOLOv8(model_path, conf_thres=0.4, iou_thres=0.5)


def predict_image(image_path):
        img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
        # img = imread_from_url(img_url)
        image = cv2.imread(str(image_path))
        # capDetect Objects
        yolov8_detector(image)

        # Draw detections
        combined_img = yolov8_detector.draw_detections(image)
        cv2.namedWindow("Output", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Output", combined_img)
        cv2.waitKey(0)


def predict_webcam(capture=cv2.VideoCapture(0), rtsp=False):
    if rtsp:
        rtsp_url = input("Enter RTSP URL: ")
        capture = cv2.VideoCapture(rtsp_url)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_AUTOSIZE)
    while capture.isOpened():

        # Read frame from the video
        ret, frame = capture.read()

        if not ret:
            break

        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)

        combined_img = yolov8_detector.draw_detections(frame)
        cv2.imshow("Detected Objects", combined_img)

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # image_path = settings.DATASETS_DIR / 'normal_json' / 'val' / '0_60_60_0_01717.jpg'
    # predict_image(image_path)
    predict_webcam()
