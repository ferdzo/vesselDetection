from ultralytics import YOLO

model = YOLO("yolo11l.pt")

data_path = 'ships-aerial-images/data.yaml'

train_params = {
    'epochs': 40,
    'batch': 32,
    'imgsz': 640,
    'lr0': 5e-4,
    'lrf': 0.1,
    'warmup_epochs': 5,
    'warmup_bias_lr': 1e-6,
    'momentum': 0.937,
    'weight_decay': 0.0001,
    'optimizer': 'AdamW',
    'device': '0,1',
    'project': 'runs/train',
    'name': 'vessel_deteciton_v11l',
    'exist_ok': True,
    'save_period': 2,
    'workers': 8,
    'patience': 20,           
    'cos_lr': True,            
}

model.train(data=data_path, **train_params)