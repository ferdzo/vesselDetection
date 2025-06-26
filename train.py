import os
from ultralytics import YOLO

model = YOLO("yolo11x.pt")

data_path = 'ships-aerial-images/data.yaml'

train_params = {
    'epochs': 20,               
    'batch': 8,
    'imgsz': 640,
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'optimizer': 'AdamW',
    'device': '0,1',
    'pretrained': True,
    'project': 'runs/train',
    'name': 'vessel_training',
    'exist_ok': True,
    'save_period': 2,
    'workers': 8,
    'auto_augment': 'autoaugment',
}

model.train(data=data_path, **train_params)