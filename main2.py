import cv2 as cv
from ultralytics  import YOLO as y

model = y('yolov8n.pt')

cap = cv.VideoCapture("demo.mp4")

if not cap.isOpened():
    print("Video Not Play")
    exit()

fps = int(cap.get(cv.CAP_PROP_FPS))

output_width= 1080
output_height = 1920

# fourcc = cv.VideoWriter(*'mp4v')
#
# out = cv.VideoWriter("output4mp4",fourcc,fps,(output_width,output_height))

smooth_x = 0

while True:
    ret,frame = cap.read()
    if not ret:
        break
    result = model(frame)
    largest_area = 0
    best_box = None
    all_boxes = []

    for r in result:
        for box in r.boxes:

            cls = int(box.cls[0])

            if cls ==0:
                x1,y1,x2,y2 = map(int ,box.xyxy[0])



            cv.imshow("frame",frame)
            if cv.waitKey(18) & 0xFF == ord('q'):
                break
cap.release()
cv.destroyAllWindows()

