from ultralytics import YOLO

model = YOLO("yolov5-p6.yaml")


results = model.train(
        data="words.yaml", 
        plots=True, 
        patience=50, 
        multi_scale=True, 
        epochs=10, 
        imgsz=1280, 
        batch=4, 
        save=True,
        line_width=1, 
        project='runs',
        show_labels=False )

results = model.val()

#model.export()




