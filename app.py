import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# ============== Load YOLO once ==============
# Replace with your model path if needed
model = YOLO("best.pt")


# ============== Image inference ==============
def detect_and_annotate_image(image):
    """Annotate an uploaded image and return the annotated image + unique classes."""
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(image_bgr, verbose=False)[0]

    unique_classes = set()
    for box in results.boxes:
        # robust tensor -> int
        cls_val = box.cls[0]
        class_id = int(cls_val.item() if hasattr(cls_val, "item") else cls_val)
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        unique_classes.add(class_name)

        # draw class name only (no probability)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr, class_name, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, "\n".join(sorted(unique_classes))


# ============== Helpers for Video ==============
def _normalize_video_input(video):
    """Return a usable filepath from Gradio Video input (str or dict)."""
    if isinstance(video, str) and os.path.exists(video):
        return video
    if isinstance(video, dict):
        for k in ("name", "path"):
            if k in video and isinstance(video[k], str) and os.path.exists(video[k]):
                return video[k]
    raise ValueError("Could not resolve video path from input.")


# ============== Video inference ==============
def detect_and_annotate_video(video):
    """Annotate an uploaded video and return path to annotated MP4 + unique classes."""
    video_path = _normalize_video_input(video)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open the uploaded video. Check the file format.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0  # fallback avoids VideoWriter errors

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # temp mp4 file for Gradio to display
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output_path = temp_output.name
    temp_output.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            "Failed to initialize VideoWriter. Install ffmpeg or try a different codec."
        )

    detected_classes = set()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # optional: process every other frame
            if frame_count % 2 == 0:
                # ensure writer size match
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                results = model(frame, verbose=False)[0]
                for box in results.boxes:
                    cls_val = box.cls[0]
                    class_id = int(cls_val.item() if hasattr(cls_val, "item") else cls_val)
                    class_name = model.names[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detected_classes.add(class_name)

                    # draw class name only (no probability)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, class_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

            out.write(frame)
            frame_count += 1
    finally:
        cap.release()
        out.release()

    return temp_output_path, "\n".join(sorted(detected_classes))


# ============== Gradio UI ==============
image_tab = gr.Interface(
    fn=detect_and_annotate_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Annotated Image"),
        gr.Textbox(label="Detected Objects (unique)")
    ],
    title="Scanventory - Image Object Detection"
)

video_tab = gr.Interface(
    fn=detect_and_annotate_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Video(label="Annotated Video"),
        gr.Textbox(label="Detected Objects (unique)")
    ],
    title="Scanventory - Video Object Detection"
)

demo = gr.TabbedInterface(
    interface_list=[image_tab, video_tab],
    tab_names=["Image Detection", "Video Detection"]
)


if __name__ == "__main__":
    demo.launch()
