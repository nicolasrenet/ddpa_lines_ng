#!/usr/bin/env python3

from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")


results = model.train( cfg='train_640.yaml')

results = model.val()


