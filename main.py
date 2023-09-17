import cv2
import numpy as np
import urllib.request
import os
from webapp import settings
from webapp.yolo import YOLOv8
from cap_from_youtube import cap_from_youtube

# CHECK THE CURRENT WORKING DIRECTORY
settings.cwd()
model_path = settings.DETECTION_MODEL_ONNX
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# print(settings.DATASETS_DIR.exists())

# Initialize YOLOv8 object detector
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)


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


def predict_mjpeg():
    video_url = input("Enter MJPEG URL: ")
    stream = urllib.request.urlopen(video_url)
    byte_arr = bytes()
    cv2.namedWindow("Detected Objects", cv2.WINDOW_FULLSCREEN)
    while True:
        byte_arr += stream.read(512)
        a = byte_arr.find(b'\xff\xd8')
        b = byte_arr.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = byte_arr[a:b + 2]
            byte_arr = byte_arr[b + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, scores, class_ids = yolov8_detector(frame)

            combined_img = yolov8_detector.draw_detections(frame)
            cv2.imshow('MPJEG Stream', combined_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)


def predict_webcam(capture=cv2.VideoCapture(0), rtsp=False):
    if rtsp:
        rtsp_url = input("Enter RTSP URL: ")
        capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_GUI_NORMAL)
    while capture.isOpened():

        # Read frame from the video
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            print("Frame is empty")
            break

        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)

        combined_img = yolov8_detector.draw_detections(frame)
        cv2.imshow("Detected Objects", combined_img)

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def youtube():
    videoUrl = 'https://youtu.be/Snyg0RqpVxY'
    cap = cap_from_youtube(videoUrl, resolution='720p')
    start_time = 5  # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

    # Initialize YOLOv7 model
    model_path = "models/yolov8m.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    while cap.isOpened():

        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            break

        try:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue

        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)

        combined_img = yolov8_detector.draw_detections(frame)
        cv2.imshow("Detected Objects", combined_img)

if __name__ == '__main__':
    # image_path = settings.DATASETS_DIR / 'normal_json' / 'val' / '0_60_60_0_01717.jpg'
    # predict_image(image_path)
    predict_mjpeg()
    # predict_webcam(rtsp=True)
