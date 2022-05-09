import cv2
from queue import Queue
from yolov5_detect_image import Y5Detect, draw_boxes_detection
import time
from threading import Thread


y5_model = Y5Detect(weights='/home/duyngu/Downloads/model/weights/best.pt')
class_names = y5_model.class_names


def video_capture(frame_detect_queue, frame_origin_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_detect_queue.put(image_rgb)
        frame_origin_queue.put(frame)

    cap.release()


def inference(frame_detect_queue, detections_queue):
    while cap.isOpened():
        image_rgb = frame_detect_queue.get()
        bboxes, labels, scores = y5_model.predict(image_rgb)
        detections_queue.put([bboxes, labels, scores])

    cap.release()


def drawing(detections_queue, frame_origin_queue, frame_final_queue):
    while cap.isOpened():
        frame_origin = frame_origin_queue.get()
        bboxes, labels, scores = detections_queue.get()
        if frame_origin is not None:
            image = draw_boxes_detection(frame_origin, bboxes, scores=scores, labels=labels, class_names=class_names)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if frame_final_queue.full() is False:
                frame_final_queue.put(image)
            else:
                time.sleep(10)
    cap.release()


if __name__ == '__main__':
    frame_detect_queue = Queue(maxsize=1)
    frame_origin_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    frame_final_queue = Queue(maxsize=1)
    input_path = "/home/duyngu/Desktop/Check_Label/output.avi"
    cap = cv2.VideoCapture(input_path)
    Thread(target=video_capture, args=(frame_detect_queue, frame_origin_queue)).start()
    Thread(target=inference, args=(frame_detect_queue, detections_queue)).start()
    Thread(target=drawing, args=(detections_queue, frame_origin_queue, frame_final_queue)).start()

    while True:
        if cap.isOpened():
            cv2.namedWindow('output')
            image = frame_final_queue.get()
            cv2.imshow('output', image)
            time.sleep(10)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow('output')
                break

    cv2.destroyAllWindows()


