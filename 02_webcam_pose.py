from ultralytics import YOLO

# 1. Cargar el modelo preentrenado de pose (se descargará solo la primera vez)
model = YOLO("yolo26n-pose.pt")

# 2. Inferencia continua sobre la webcam
# source=0 -> Cámara web integrada
# show=True -> Abre la ventana visual con el esqueleto dibujado
# conf=0.5 -> Umbral mínimo de confianza
model.predict(source=0, show=True, conf=0.5)