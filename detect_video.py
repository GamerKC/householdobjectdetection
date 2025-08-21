import cv2
from ultralytics import YOLO

def detect_objects_in_video(video_path, model_path="best.pt", output_path="output_annotated.mp4"):
    # Load model
    model = YOLO(model_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    # Get video properties
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0  # fallback if 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # Define VideoWriter to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Failed to initialize VideoWriter. Check codec/ffmpeg.")

    print("Processing video...")
    detected_classes = set()  # Unique object names across the whole video

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 2 == 0:  # Process every other frame
            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                class_id = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detected_classes.add(class_name)

                # Draw box and label (class name only, no probability)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"\nAnnotated video saved to {output_path}")

    print("\nUnique detected objects:")
    print(sorted(detected_classes))

    # Optional: Show first frame of result
    cap2 = cv2.VideoCapture(output_path)
    ret, frame = cap2.read()
    if ret:
        cv2.imshow("Annotated Video (first frame)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap2.release()

# Run from command line
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--model_path", default="best.pt", help="Path to YOLO model file")
    parser.add_argument("--output_path", default="output_annotated.mp4", help="Path to save annotated video")
    args = parser.parse_args()

    detect_objects_in_video(args.video_path, args.model_path, args.output_path)
