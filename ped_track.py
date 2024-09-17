import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import argparse
import torch

from road_seg import run_inference, load_model, draw_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

# Segmentation Mode

seg_model = load_model(
    "saved_model/model_float32.onnx", use_gpu=torch.cuda.is_available()
)
input_size = [512, 896]

# Load a model
model_path = "best-medium.pt"
model_name = model_path.split("/")[-1].split(".")[0]
model = YOLO(model_path)  # load an official model
model.overrides["conf"] = 0.8  # NMS confidence threshold
model.overrides["iou"] = 0.4  # NMS IoU threshold
model.overrides["agnostic_nms"] = False  # NMS class-agnostic
model.overrides["max_det"] = 1000  # maximum number of detections per image
# model.overrides['classes'] = 2 ## define classes
names = model.names
# names = {value: key for key, value in names.items()}
# keep only person
names = {key: value for key, value in names.items() if value == "person"}
colors = np.random.randint(0, 255, size=(80, 3), dtype="uint8")

tracking_trajectories = {}
PREDICTION_STEPS = 30


def process(image):
    global input_video_name
    bboxes = []

    results = model.track(
        image, verbose=False, device=device, persist=True, tracker="botsort.yaml"
    )
    # image_width, image_height = image.shape[1], image.shape[0]
    segmentation_map = run_inference(seg_model, input_size, image)
    image, bit_mask = draw_mask(
        image,
        0.5,
        segmentation_map,
    )

    for id_ in list(tracking_trajectories.keys()):
        if id_ not in [
            int(bbox.id)
            for predictions in results
            if predictions is not None
            for bbox in predictions.boxes
            if bbox.id is not None
        ]:
            del tracking_trajectories[id_]

    for predictions in results:
        if predictions is None:
            continue

        if predictions.boxes is None or predictions.boxes.id is None:
            continue

        for bbox in predictions.boxes:
            intersection_ids = []

            # Draw trajectories
            for id_, trajectory in tracking_trajectories.items():
                for i in range(1, len(trajectory)):
                    cv2.line(
                        image,
                        (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                        (int(trajectory[i][0]), int(trajectory[i][1])),
                        (255, 255, 255),
                        2,
                    )
                smooth, smoothed_trajectory = smooth_trajectory(trajectory)
                if not smooth:
                    continue
                predicted_trajectory = linear_extrapolation(list(smoothed_trajectory))
                # check if the predicted trajectory intersects with the road bit_mask
                intersection_trajectory = np.concatenate(
                    (np.array(smoothed_trajectory), np.array(predicted_trajectory))
                )
                for point in intersection_trajectory:
                    if (
                        0 <= int(point[0]) < bit_mask.shape[1]
                        and 0 <= int(point[1]) < bit_mask.shape[0]
                    ):
                        if bit_mask[int(point[1]), int(point[0])] != 1:
                            print("Predicted trajectory intersects with road")
                            intersection_ids.append(id_)
                            line_color = (0, 0, 255)  # Red color for prediction
                            break
                        else:
                            line_color = (0, 255, 0)  # Green color for prediction

                for i in range(1, len(predicted_trajectory)):
                    cv2.line(
                        image,
                        (
                            int(predicted_trajectory[i - 1][0]),
                            int(predicted_trajectory[i - 1][1]),
                        ),
                        (
                            int(predicted_trajectory[i][0]),
                            int(predicted_trajectory[i][1]),
                        ),
                        line_color,
                        2,
                    )

            for scores, class_id, bbox_coords, id_ in zip(
                bbox.conf, bbox.cls, bbox.xyxy, bbox.id
            ):
                # classes is a tensor with class index
                if class_id.item() not in names:
                    continue
                xmin = bbox_coords[0]
                ymin = bbox_coords[1]
                xmax = bbox_coords[2]
                ymax = bbox_coords[3]
                bboxes.append([bbox_coords, scores, class_id, id_])

                # label = (' '+f'ID: {int(id_)}'+' '+str(predictions.names[int(class_id)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                label = (
                    " "
                    + f"ID: {int(id_)}"
                    + " "
                    + str(predictions.names[int(class_id)])
                    + " "
                )

                if int(id_) in intersection_ids:
                    label += "WARNING!!"
                    box_color = (0, 0, 255)  # Red color for warning
                    box_thickness = 3
                # else if the box itself is not intersecting with the road
                elif np.any(
                    bit_mask[
                        int(ymin) + 10 : int(ymax) - 10, int(xmin) + 10 : int(xmax) - 10
                    ]
                    != 1
                ):
                    label += "WARNING!!"
                    box_color = (0, 0, 255)  # Red color for warning
                    box_thickness = 3
                else:
                    box_color = (0, 255, 0)  # Green color for normal
                    box_thickness = 2

                cv2.rectangle(
                    image,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    box_color,
                    box_thickness,
                )
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                dim, baseline = text_size[0], text_size[1]
                cv2.rectangle(
                    image,
                    (int(xmin), int(ymin)),
                    ((int(xmin) + dim[0] // 3) - 20, int(ymin) - dim[1] + baseline),
                    (30, 30, 30),
                    cv2.FILLED,
                )
                cv2.putText(
                    image,
                    label,
                    (int(xmin), int(ymin) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                centroid_x = (xmin + xmax) / 2
                centroid_y = (ymin + ymax) / 2

                person_height = ymax - ymin
                foot_center = (centroid_x, centroid_y + person_height / 4)
                centroid = (centroid_x, centroid_y)

                # Append centroid to tracking_points
                if id_ is not None and int(id_) not in tracking_trajectories:
                    tracking_trajectories[int(id_)] = deque(maxlen=15)
                if id_ is not None:
                    # tracking_trajectories[int(id_)].append((centroid_x, centroid_y))
                    tracking_trajectories[int(id_)].append(foot_center)

    return image


def process_video(source):
    if not os.path.exists("output"):
        os.makedirs("output")
    global input_video_name
    cap = cv2.VideoCapture(int(source) if source == "0" else source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # Change the codec if needed (e.g., 'XVID')
    # input_video_name = source.split('.')[0]  # Get the input video name without extension
    input_video_name = os.path.splitext(os.path.basename(source))[0]
    # print('testing : ', input_video_name)
    out = cv2.VideoWriter(
        f"output/{input_video_name}_output_{model_name}.mp4",
        fourcc,
        fps,
        (input_size[1], input_size[0]),
    )

    if not cap.isOpened():
        print(f"Error: Could not open video file {source}.")
        return

    frameId = 0
    start_time = time.time()
    fps_str = str()

    while True:
        frameId += 1
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (input_size[1], input_size[0]))
        frame1 = frame.copy()
        if not ret:
            break

        frame = process(frame1)

        if frameId % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_current = 10 / elapsed_time  # Calculate FPS over the last 20 frames
            fps_str = f"FPS: {fps_current:.2f}"
            start_time = time.time()  # Reset start_time for the next 20 frames

        cv2.putText(
            frame,
            fps_str,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        out.write(frame)
        cv2.imshow(f"yolo_{source}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def smooth_trajectory(trajectory, window_size=5):
    if len(trajectory) < window_size:
        return False, trajectory  # Not enough points to smooth

    x_coords = np.array([p[0] for p in trajectory])
    y_coords = np.array([p[1] for p in trajectory])

    # Apply a uniform filter to smooth the coordinates
    smoothed_x = np.convolve(x_coords, np.ones(window_size) / window_size, mode="valid")
    smoothed_y = np.convolve(y_coords, np.ones(window_size) / window_size, mode="valid")

    # Since convolution reduces the array size, we need to handle the array edges
    pad_length = len(trajectory) - len(smoothed_x)
    pad_start = pad_length // 2
    pad_end = pad_length - pad_start

    # Pad the smoothed coordinates to match the original trajectory length
    smoothed_x = np.pad(smoothed_x, (pad_start, pad_end), mode="edge")
    smoothed_y = np.pad(smoothed_y, (pad_start, pad_end), mode="edge")

    # Combine x and y into a list of tuples
    smoothed_trajectory = list(zip(smoothed_x, smoothed_y))
    return True, smoothed_trajectory


def linear_extrapolation(trajectory, num_future_steps=PREDICTION_STEPS):
    if len(trajectory) < 2:
        return trajectory  # Not enough data to extrapolate

    # Calculate velocities between consecutive points
    velocities = [
        (
            trajectory[i + 1][0] - trajectory[i][0],
            trajectory[i + 1][1] - trajectory[i][1],
        )
        for i in range(len(trajectory) - 1)
    ]

    # Average the velocities
    avg_velocity = np.mean(velocities, axis=0)

    # Predict future positions
    last_position = trajectory[-1]
    predicted_trajectory = [last_position]
    for _ in range(num_future_steps):
        next_position = (
            last_position[0] + avg_velocity[0],
            last_position[1] + avg_velocity[1],
        )
        predicted_trajectory.append(next_position)
        last_position = next_position

    return predicted_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video with YOLO.")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video file paths or camera indices",
    )

    args = parser.parse_args()

    process_video(args.source)
