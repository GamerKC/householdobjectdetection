import cv2
from ultralytics import YOLO

def detect_objects_and_annotate(image_path, model_path="yolo11m.pt", output_path="output.jpg"):
    # Load model
    model = YOLO(model_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Inference
    results = model(image)[0]

    # Collect object names and draw annotations
    print("Detected objects:")
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        print(f" - {class_name}: {conf:.2f}")

        # Draw box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Save annotated image
    cv2.imwrite(output_path, image)
    print(f"\nAnnotated image saved to {output_path}")

    # Optional: Display image
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run from command line
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model_path", default="yolo11m.pt", help="Path to YOLO model file")
    parser.add_argument("--output_path", default="output.jpg", help="Path to save annotated image")
    args = parser.parse_args()

    detect_objects_and_annotate(args.image_path, args.model_path, args.output_path)
