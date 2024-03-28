from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import config as cf
import torch

EPOCHS = 20
BATCH_SIZE = 16
WORKERS = 8

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': cf.root_dir,
        'images_dir': cf.train_images_dir,
        'labels_dir': cf.train_labels_dir,
        'classes': cf.classes
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': cf.root_dir,
            'images_dir': cf.val_images_dir,
            'labels_dir': cf.val_labels_dir,
            'classes': cf.classes
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

train_data.dataset.transforms[0]
train_data.dataset.transforms.pop(2)
train_data.dataset.transforms

train_params = {
    'silent_mode': False,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": EPOCHS,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(cf.classes),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(cf.classes),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_095(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(cf.classes),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50:0.95'
}

CHECKPOINT_DIR = '/cluster/home/muhammmo/yolonas_checkpoints'

trainer = Trainer(experiment_name='yolonas_run', ckpt_root_dir=CHECKPOINT_DIR)
model = models.get('yolo_nas_l', num_classes=len(cf.classes), pretrained_weights="coco")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = model.to(device)

trainer.train(model=model, 
              training_params=train_params, 
              train_loader=train_data, 
              valid_loader=val_data)