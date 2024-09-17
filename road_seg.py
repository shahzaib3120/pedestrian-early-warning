#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, expand dimensions, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Post process:squeeze
    segmentation_map = result[0]
    segmentation_map = np.squeeze(segmentation_map)

    return segmentation_map


def load_model(model_path, use_gpu):
    session_options = onnxruntime.SessionOptions()
    if use_gpu:
        providers = ["CUDAExecutionProvider"]
        print("using GPU for Segmentation")
    else:
        providers = ["CPUExecutionProvider"]
    return onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument(
        "--use_gpu", action="store_true", help="Enable CUDA for ONNX Runtime"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="saved_model/model_float32.onnx",
    )

    args = parser.parse_args()

    model_path = args.model
    input_size = [512, 896]

    score = args.score

    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

    # Initialize video capture
    cap = cv.VideoCapture(cap_device)

    # Load model
    # onnx_session = onnxruntime.InferenceSession(model_path)
    onnx_session = load_model(model_path, args.use_gpu)

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        segmentation_map = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_mask(
            debug_image,
            elapsed_time,
            score,
            segmentation_map,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow("road-segmentation-adas-0001 Demo", debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_mask(image, score, segmentation_map):
    image_width, image_height = image.shape[1], image.shape[0]

    # Match the size
    debug_image = copy.deepcopy(image)
    segmentation_map = cv.resize(
        segmentation_map,
        dsize=(image_width, image_height),
        interpolation=cv.INTER_LINEAR,
    )

    # color list
    color_image_list = []
    # ID 0:BackGround
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = (0, 0, 0)
    color_image_list.append(bg_image)
    # ID 1:Road
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = (255, 0, 0)
    color_image_list.append(bg_image)
    # ID 2:Curb
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    # bg_image[:] = (0, 255, 0)
    bg_image[:] = (255, 0, 0)
    color_image_list.append(bg_image)
    # ID 4:Mark
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    # bg_image[:] = (0, 0, 255)
    bg_image[:] = (255, 0, 0)
    color_image_list.append(bg_image)

    bit_mask = np.ones((image_height, image_width), dtype=np.uint8)

    # Overlay segmentation map
    masks = segmentation_map.transpose(2, 0, 1)
    for index, mask in enumerate(masks):
        if index == 0:
            continue
        # Threshold check by score
        mask = np.where(mask > score, 0, 1)
        bit_mask = np.where(mask == 0, 0, bit_mask)

        # Overlay
        mask = np.stack((mask,) * 3, axis=-1).astype("uint8")
        mask_image = np.where(mask, debug_image, color_image_list[index])
        debug_image = cv.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

    return debug_image, bit_mask


if __name__ == "__main__":
    main()
