from ultralytics import YOLO
import cv2
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = 'CCWW1621.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

while ret:
    ret, frame = cap.read()

    if ret:
        print("Frame read successfully")
        results = model.track(frame,persist=True,tracker="bytetrack.yaml")

        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()     