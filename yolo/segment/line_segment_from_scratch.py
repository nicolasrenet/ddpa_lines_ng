#!/usr/bin/env python3

from ultralytics import YOLO

model = YOLO("yolov8n-seg.yaml")


results = model.train( cfg='train_1280-640.yaml')

results = model.val()


