import os
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
from super_gradients.training import models
from torchinfo import summary
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
import cv2


dataset_params = {
    'data_dir': 'data',
    'train_images_dir': 'train/train_img',
    'train_labels_dir': 'train/trainyolo',
    'val_images_dir': 'val/val_img',
    'val_labels_dir': 'val/valyolo',
    'test_images_dir': 'test/test_img',
    'test_labels_dir': 'test/testyolo',
    'classes': ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram',  'Tricycle'],
}


def save_predict_images_from_dir(model, image_dir, output_dir, conf=0.60, show=False):
    # Get a list of all image files in the directory
    image_files = list(Path(image_dir).rglob("*.jpg"))

    # Loop over all images and make predictions
    predictions = []
    for image_file in image_files:
        # Load image
        img = Image.open(image_file)

        # Make prediction

        if show == True:
            results = model.predict(img, conf).show()
        else:
            results = model.predict(img, conf).save(output_dir)
        # results.save()  # Save the image with bounding boxes
        predictions.append(results)

    return predictions


def show_predicted_images_from_dir(model, image_dir, conf=0.60):
    # Get a list of all image files in the directory
    image_files = list(Path(image_dir).rglob("*.jpg"))

    # Loop over all images and make predictions
    predictions = []
    for image_file in image_files:
        # Load image
        img = Image.open(image_file)

        # Make prediction

        # if show == True:
        # results = model.predict(img, conf).show()
        # else:
        # results = model.predict(img, conf)
        model.predict(img, conf=conf).show()
        # results.save()  # Save the image with bounding boxes
        # predictions.append(results)

    # return predictions


def cxcyxy_to_xyxy(image_label_txt, image_path):
    # Open image and get dimensions
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Read label file
    with open(image_label_txt, "r") as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert to x_min, y_min, x_max, y_max format
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        x_max = (x_center + width / 2) * image_width
        y_max = (y_center + height / 2) * image_height

        boxes.append([x_min, y_min, x_max, y_max])

    return boxes


def get_boxes_xyxy(model, image_path, conf=0.6):
    ### Input:
    # model : yolo_nas_l // best_model
    # image_path: 'pth/to/local/img_1.jpg'
    # conf = confidence

    ### Output:
    # all_boxes = list of list | coordinate of all bounding boxes

    results = model.predict(image_path, conf)
    all_boxes = []
    for image_prediction in results:
        bboxes = image_prediction.prediction.bboxes_xyxy
        for i, (bbox) in enumerate(bboxes):
            all_boxes.append(bbox.tolist())

    return all_boxes


def box_area(box_xyxy):
    x_min, y_min, x_max, y_max = box_xyxy
    return (x_max - x_min) * (y_max - y_min)


def bbox_iou(box1_xyxy, box2_xyxy):
    x1, y1, x1_max, y1_max = box1_xyxy
    x2, y2, x2_max, y2_max = box2_xyxy

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = box_area(box1_xyxy)
    box2_area = box_area(box2_xyxy)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def draw_boxes_all_models(image_path, prediction_dict):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # convert color space from BGR to RGB

    # Set thickness for bounding boxes
    box_thickness = 3

    # Ground truth boxes
    for box in prediction_dict["Ground_Truth_boxes"]:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), (255, 0, 0), box_thickness
        )  # Red color

    # Default model boxes
    for box in prediction_dict["Default_YoloNas_boxes"]:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), (0, 0, 255), box_thickness
        )  # Blue color

    # Finetuned model boxes
    for box in prediction_dict["Finetuned_YoloNas_boxes"]:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), (0, 255, 0), box_thickness
        )  # Green color

    # Parameters for text size and boldness
    font_scale = 1  # Increase for larger text
    text_thickness = 4  # Increase for bolder text

    # Add labels on top right corner
    cv2.putText(
        image,
        "Ground Truth",
        (image.shape[1] - 300, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 0),
        text_thickness,
    )
    cv2.putText(
        image,
        "Default Model",
        (image.shape[1] - 300, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        text_thickness,
    )
    cv2.putText(
        image,
        "Finetuned Model",
        (image.shape[1] - 300, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        text_thickness,
    )

    # Show image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def get_boxex_for_all_models_car(
    image_path, image_label, default_model, finetuned_model, conf=0.6
):
   
    image_name = image_path
    ### get bounding boxes
    # in case of ground truth (no model prediction)
    gt_boxes = cxcyxy_to_xyxy(image_label, image_path)
    # in case of default yolo_nas
    default_model_boxes = get_boxes_xyxy(default_model, image_path, conf)
    # in case of finetned yolo_nas
    finetuned_model_boxes = get_boxes_xyxy(finetuned_model, image_path, conf)

    prediction_dict = {}

    prediction_dict["image"] = image_name
    prediction_dict["Ground_Truth_boxes"] = gt_boxes
    prediction_dict["Default_YoloNas_boxes"] = default_model_boxes
    prediction_dict["Finetuned_YoloNas_boxes"] = finetuned_model_boxes

    return prediction_dict
def get_boxex_for_all_models(
    image_path, image_label, default_model, finetuned_model, conf=0.6
):
    ### renaming image
    # Split the string at the first occurrence of "_jpg"
    split_name_0 = image_path.split("img_", 1)
    # The split method returns a list, so you need to get the first element
    image_name_0 = "img_" + split_name_0[1]
    # Split the string at the first occurrence of "_jpg"
    split_name = image_name_0.split("_jpg", 1)
    # The split method returns a list, so you need to get the first element
    image_name = split_name[0]

    ### get predictions
    # in case of ground truth (no model prediction)
    gt_image = Image.open(image_path)
    # in case of default yolo_nas
    predicted_image_default = default_model.predict(image_path, conf)
    # in case of finetned yolo_nas
    predicted_image_finetuned = finetuned_model.predict(image_path, conf)

    ### get bounding boxes
    # in case of ground truth (no model prediction)
    gt_boxes = cxcyxy_to_xyxy(image_label, image_path)
    # in case of default yolo_nas
    default_model_boxes = get_boxes_xyxy(default_model, image_path, conf)
    # in case of finetned yolo_nas
    finetuned_model_boxes = get_boxes_xyxy(finetuned_model, image_path, conf)

    prediction_dict = {}

    prediction_dict["image"] = image_name
    prediction_dict["Ground_Truth_boxes"] = gt_boxes
    prediction_dict["Default_YoloNas_boxes"] = default_model_boxes
    prediction_dict["Finetuned_YoloNas_boxes"] = finetuned_model_boxes

    return prediction_dict

def compute_iou_for_all_models(prediction_dict):
    image_name = prediction_dict["image"]
    gt_boxes = prediction_dict["Ground_Truth_boxes"]
    defalt_yoloNas_boxes = prediction_dict["Default_YoloNas_boxes"]
    finetuned_yoloNas_boxes = prediction_dict["Finetuned_YoloNas_boxes"]

    final_iou_default = []  # only keep highest IoU for each ground truth box
    final_pbb_default = (
        []
    )  # only keep the bb with the highest IoU for each ground truth box

    final_iou_finetuned = []  # only keep highest IoU for each ground truth box
    final_pbb_finetuned = (
        []
    )  # only keep the bb with the highest IoU for each ground truth box

    for gt_box in gt_boxes:
        ls_iou_default = []
        for pred_box in defalt_yoloNas_boxes:
            iou = bbox_iou(gt_box, pred_box)
            ls_iou_default.append(iou)
        idx_max = np.argmax(ls_iou_default)  # find the position with the highest IoU
        final_iou_default.append(ls_iou_default[idx_max])
        final_pbb_default.append(defalt_yoloNas_boxes[idx_max])

    for gt_box in gt_boxes:
        ls_iou_finetuned = []
        for pred_box in finetuned_yoloNas_boxes:
            iou = bbox_iou(gt_box, pred_box)
            ls_iou_finetuned.append(iou)
        idx_max = np.argmax(ls_iou_finetuned)  # find the position with the highest IoU
        final_iou_finetuned.append(ls_iou_finetuned[idx_max])
        final_pbb_finetuned.append(finetuned_yoloNas_boxes[idx_max])

    iou_dict = {}
    iou_dict["image"] = image_name
    iou_dict["IOU_with_default_YoloNAS"] = final_iou_default
    iou_dict["IOU_with_finetuned_YoloNAS"] = final_iou_finetuned

    return iou_dict
