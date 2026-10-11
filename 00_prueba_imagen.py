from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # cargar un modelo de detección YOLO26n preentrenado
results = model("https://ultralytics.com/images/bus.jpg", save=True)  # predecir y guardar la imagen anotada