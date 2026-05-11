import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv.VideoCapture("demo.mp4")

if not cap.isOpened():
    print("Please check video")
    exit()


fps = int(cap.get(cv.CAP_PROP_FPS))

# output crop size
output_width = 1080
output_height = 1920

# codec
fourcc = cv.VideoWriter_fourcc(*'mp4v')

# save video
out = cv.VideoWriter(
    "output1.mp4",
    fourcc,
    fps,
    (output_width, output_height)
)
smooth_x = 0
while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    largest_area = 0
    best_box = None
    all_boxes = []

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 0:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                area = (x2 - x1) * (y2 - y1)

                if area > largest_area:
                    largest_area = area
                    best_box = (x1, y1, x2, y2)
                all_boxes.append((x1, y1, x2, y2))


    if best_box:

        x1, y1, x2, y2 = best_box
        for bx in all_boxes:

            px1, py1, px2, py2 = bx

            color = (255, 0, 0)

            if bx == best_box:
                color = (0, 255, 0)

            # cv.rectangle(
            #     frame,
            #     (px1, py1),
            #     (px2, py2),
            #     color,
            #     2
            # )

        center_x = (x1 + x2) // 2

        smooth_x = int(
            0.8 * smooth_x + 0.2 * center_x
        )

        frame_h, frame_w = frame.shape[:2]

        crop_w = int(frame_h * 9 / 16)
        margin = 100

        left = smooth_x - crop_w // 2 - margin
        right = smooth_x + crop_w // 2 + margin

        if left < 0:
            left = 0
            right = crop_w

        if right - left > frame_w:
            left = 0 
            right = frame_w
            left = frame_w - crop_w

        cropped = frame[:, left:right]

        final_output = cv.resize(
            cropped,
            (1080, 1920)
        )

        out.write(final_output)

        cv.imshow(
            "Auto Reframe",
            final_output
        )
        if cv.waitKey(18) & 0xFF == ord('q'):
            break

cap.release()
out.release()

cv.destroyAllWindows()