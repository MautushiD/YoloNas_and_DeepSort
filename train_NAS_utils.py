from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name='yolonas_AI',
                  ckpt_root_dir=CHECKPOINT_DIR)
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



train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 7,
        'num_workers': 2
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 7,
        'num_workers': 2
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 7,
        'num_workers': 2
    }
)



