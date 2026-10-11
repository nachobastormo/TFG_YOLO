from ultralytics import YOLO

# Carga el modelo preentrenado que ya tienes descargado
model = YOLO("yolo26n.pt")

# Inferencia continua sobre la webcam
# source=0 -> ID de la webcam integrada
# show=True -> Abre una ventana con el vídeo anotado
# conf=0.5 -> Filtro de confianza mínima del 50%
model.predict(source=0, show=True, conf=0.5)