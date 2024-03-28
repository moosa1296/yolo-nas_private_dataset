import torch
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import os
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val
from super_gradients.training import dataloaders
import config as cf
from super_gradients.training import Trainer
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback


ROOT_TEST = cf.test_images_dir
OUTPUT_DIR = '/cluster/home/muhammmo/det_output'
image_extensions = {'.jpg', '.jpeg', '.png'} 
all_files = os.listdir(ROOT_TEST)
all_images = [file for file in all_files if os.path.splitext(file)[1].lower() in image_extensions]
os.makedirs(OUTPUT_DIR, exist_ok=True)

best_model = models.get('yolo_nas_l',
                        num_classes=len(cf.classes),
                        checkpoint_path='/cluster/home/muhammmo/yolonas_checkpoints/yolonas_run/RUN_20240322_233508_196064/average_model.pth') 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
best_model = best_model.to(device)

for image in tqdm(all_images, total=len(all_images)):
    image_path = os.path.join(ROOT_TEST, image)
    out = best_model.predict(image_path)
    output_path = os.path.join(OUTPUT_DIR, image)
    out.save(output_path)

def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    height, width, _ = image.shape
    lw = max(round(sum(image.shape) / 2 * 0.003), 2) 
    tf = max(lw - 1, 1)
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1*width)
        ymin = int(y1*height)
        xmax = int(x2*width)
        ymax = int(y2*height)

        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        
        class_name = cf.classes[0]

        color = (0, 0, 255)
        
        cv2.rectangle(
            image, 
            p1, p2,
            color=color, 
            thickness=lw,
            lineType=cv2.LINE_AA
        ) 

        # For filled rectangle.
        w, h = cv2.getTextSize(
            class_name, 
            0, 
            fontScale=lw / 3, 
            thickness=tf
        )[0]

        outside = p1[1] - h >= 3
        new_p2 = p1[0] + w, p2[1] + h + 3 if outside else p2[1] - h - 3

        cv2.rectangle(
            image, 
            (p1[0], p2[1]), new_p2, 
            color=color, 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )  
        cv2.putText(
            image, 
            class_name, 
            (p1[0], p2[1] + h + 2 if outside else p2[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw/3, 
            color=(255, 255, 255), 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return image

def plot(image_path, label_path, num_samples):
    all_training_images = glob.glob(image_path+'/*.jpg')
    all_training_labels = glob.glob(label_path+'/*.txt')
    all_training_images.sort()
    all_training_labels.sort()
    
    temp = list(zip(all_training_images, all_training_labels))
    random.shuffle(temp)
    all_training_images, all_training_labels = zip(*temp)
    all_training_images, all_training_labels = list(all_training_images), list(all_training_labels)
    
    num_images = len(all_training_images)
    
    if num_samples == -1:
        num_samples = num_images
        
    for i in range(num_samples):
        image_name = all_training_images[i].split(os.path.sep)[-1]
        image = cv2.imread(all_training_images[i])
        with open(all_training_labels[i], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label, x_c, y_c, w, h = label_line.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.figure(figsize=(12, 9))
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
        plt.savefig('plot_output.png')

plot(
    image_path=cf.test_images_dir, 
    label_path=cf.test_labels_dir,
    num_samples=5,
)

