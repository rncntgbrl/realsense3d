import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO  # Pour YOLOv8

def main():
    # -----------------------------
    # 1. Configuration RealSense
    # -----------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    
    # On active le flux de profondeur et le flux couleur
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # On démarre le pipeline
    pipeline_profile = pipeline.start(config)

    # Récupération de l'alignement pour aligner l'image de profondeur sur l'image couleur
    align_to = rs.stream.color
    align = rs.align(align_to)

    # -----------------------------
    # 2. Chargement du modèle YOLO
    # -----------------------------
    # Exemple : chargement d’un modèle YOLOv8 pré-entrainé sur COCO
    model = YOLO("yolov8n.pt")  # ou yolov8s.pt, yolov8m.pt, etc.

    # Optionnel : configuration du seuil de confiance minimal
    # model.conf = 0.5  # paramètre possible si vous utilisez ultralytics>=8.0.74

    try:
        while True:
            # -----------------------------
            # 3. Acquisition des images
            # -----------------------------
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Conversion en arrays numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # -----------------------------
            # 4. Détection des objets (IA)
            # -----------------------------
            # On utilise le modèle YOLOv8
            results = model(color_image, verbose=False)  # verbose=False pour moins d’affichage
            # results est une liste contenant l'ensemble des détections pour chaque frame
            
            # On récupère la liste des détections
            # Chaque élément "det" dans results[0].boxes est une bounding box détectée
            # Ex : det.xyxy, det.conf, det.cls
            detections = results[0].boxes if len(results) > 0 else []

            # -----------------------------
            # 5. Dessin des bounding boxes
            # -----------------------------
            # Couleurs aléatoires pour chaque classe (ici, on utilise un simple random).
            # On peut utiliser un dictionnaire de classes -> couleurs fixes si besoin.
            height, width, _ = color_image.shape
            
            for det in detections:
                # Récupération des coordonnées de la bounding box
                # Les coordonnées sont [x_min, y_min, x_max, y_max]
                box = det.xyxy[0].cpu().numpy().astype(int)  # Conversion en int
                x_min, y_min, x_max, y_max = box

                # Confiance et classe
                conf = float(det.conf[0].cpu().numpy())
                cls_id = int(det.cls[0].cpu().numpy())  # ID de classe
                class_name = model.names[cls_id] if model.names and cls_id in model.names else f"cls_{cls_id}"

                # Calcul distance
                # On prend la distance au centre de la bounding box (ou un autre point)
                cx = int((x_min + x_max) / 2)
                cy = int((y_min + y_max) / 2)

                # Récupération de la profondeur au point central
                depth_value = depth_frame.get_distance(cx, cy)  # en mètres
                distance_text = f"{depth_value:.2f} m"

                # Dessiner la bounding box sur l'image couleur
                color = (0, 255, 0)  # Vert par défaut
                cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), color, 2)

                # Afficher le label : classe, confiance et distance
                label = f"{class_name} {conf:.2f}"
                cv2.putText(color_image, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(color_image, distance_text, (x_min, y_max + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # -----------------------------
            # 6. Affichage
            # -----------------------------
            cv2.imshow("RealSense - Couleur", color_image)

            # Pour quitter, on attend la touche 'q' ou 'ESC'
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    except Exception as e:
        print(f"Erreur : {e}")

    finally:
        # On arrête le pipeline et ferme les fenêtres
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
