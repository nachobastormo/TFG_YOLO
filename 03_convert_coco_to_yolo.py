import os
import json
import shutil
import random
from pathlib import Path

# Configuración de rutas
JSON_PATH = Path("raw_coco/annotations.json")
# Si tu archivo se llama Tuna_Scanner.json en raw_coco, cámbialo aquí:
if not JSON_PATH.exists() and Path("raw_coco/Tuna_Scanner.json").exists():
    JSON_PATH = Path("raw_coco/Tuna_Scanner.json")

SRC_IMAGES_DIR = Path("raw_coco/images")
OUTPUT_DIR = Path("dataset")

VAL_RATIO = 0.2
SEED = 42

def create_dirs():
    # Limpiar y recrear carpetas para no mezclar etiquetas antiguas
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def main():
    if not JSON_PATH.exists():
        print(f"Error: No se encuentra el archivo {JSON_PATH}")
        return

    create_dirs()

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_info = {img["id"]: img for img in data["images"]}

    img_to_anns = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    valid_img_ids = [img_id for img_id in images_info.keys() if img_id in img_to_anns]
    
    random.seed(SEED)
    random.shuffle(valid_img_ids)

    n_val = int(len(valid_img_ids) * VAL_RATIO)
    val_ids = set(valid_img_ids[:n_val])

    processed_count = 0

    for img_id in valid_img_ids:
        img_info = images_info[img_id]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        split = "val" if img_id in val_ids else "train"

        src_img = SRC_IMAGES_DIR / file_name
        if not src_img.exists():
            src_img = SRC_IMAGES_DIR / Path(file_name).name
        if not src_img.exists():
            continue

        dst_img = OUTPUT_DIR / "images" / split / src_img.name
        shutil.copy(src_img, dst_img)

        label_file = OUTPUT_DIR / "labels" / split / f"{src_img.stem}.txt"
        lines = []

        for ann in img_to_anns[img_id]:
            keypoints = ann.get("keypoints", [])
            if not keypoints:
                continue

            # Extraer coordenadas de los keypoints válidos
            xs = [keypoints[i] for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0]
            ys = [keypoints[i+1] for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0]

            if not xs or not ys:
                continue

            # Comprobar si existe bbox válida en la anotación
            bbox = ann.get("bbox", [])
            if bbox and len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
                x_min, y_min, bw, bh = bbox
            else:
                # Auto-generar la bounding box rodeando los keypoints con un 10% de margen
                x_min_raw, x_max_raw = min(xs), max(xs)
                y_min_raw, y_max_raw = min(ys), max(ys)
                pad_x = (x_max_raw - x_min_raw) * 0.10
                pad_y = (y_max_raw - y_min_raw) * 0.15

                x_min = max(0, x_min_raw - pad_x)
                y_min = max(0, y_min_raw - pad_y)
                bw = min(width - x_min, (x_max_raw - x_min_raw) + 2 * pad_x)
                bh = min(height - y_min, (y_max_raw - y_min_raw) + 2 * pad_y)

            # Normalizar bbox
            x_center = (x_min + bw / 2.0) / width
            y_center = (y_min + bh / 2.0) / height
            w_norm = bw / width
            h_norm = bh / height

            # Normalizar keypoints
            kpts_str = []
            for i in range(0, len(keypoints), 3):
                kx = keypoints[i] / width
                ky = keypoints[i+1] / height
                kv = keypoints[i+2]
                kpts_str.append(f"{kx:.6f} {ky:.6f} {int(kv)}")

            line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(kpts_str)
            lines.append(line)

        if lines:
            with open(label_file, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines) + "\n")
            processed_count += 1

    print(f"¡Conversión corregida! Se procesaron {processed_count} imágenes con Bounding Boxes calculadas.")

if __name__ == "__main__":
    main()