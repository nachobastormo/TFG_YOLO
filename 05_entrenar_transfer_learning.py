from ultralytics import YOLO
import torch

def main():
    # Comprobar si CUDA (GPU NVIDIA) está disponible
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo seleccionado: {dispositivo} (CUDA disponible: {torch.cuda.is_available()})")

    # Cargamos los pesos preentrenados en atunes que proporcionó el profesor
    model = YOLO("20260327_yolo-pose-tuna.pt")

    # Transfer Learning / Fine-tuning
    results = model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=dispositivo,
        project="runs/pose",
        name="train_transfer_learning",
        plots=True
    )
    print("¡Entrenamiento con Transfer Learning finalizado!")

if __name__ == "__main__":
    main()