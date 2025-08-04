import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model once
model = YOLO("yolo11m.pt")  # Change to your model path

def detect_and_annotate(image):
    # Convert PIL image to OpenCV format (numpy array)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(image)[0]

    objects = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{class_name} {conf:.2f}"
        objects.append(label)

        # Draw box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Convert back to RGB for Gradio display
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return annotated_image, "\n".join(objects)

# Launch Gradio Interface
demo = gr.Interface(
    fn=detect_and_annotate,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Annotated Image"),
        gr.Textbox(label="Detected Objects")
    ],
    title="YOLO Object Detector",
    description="Upload an image to detect and annotate objects using YOLOv8",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
