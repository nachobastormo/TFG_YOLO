from ultralytics import YOLO

def main():
    # Modelo base YOLO26 Nano Pose (entrenamiento desde cero)
    model = YOLO("yolo26n-pose.pt")

    results = model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,               # GPU RTX 4060
        project="runs/pose",
        name="train_scratch",
        plots=True
    )
    print("¡Entrenamiento desde cero finalizado!")

if __name__ == "__main__":
    main()