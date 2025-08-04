import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os

class ObjectDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detector")

        self.model = YOLO("yolo11m.pt")  # Change path if needed
        self.image_path = None

        # Buttons
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect Objects", command=self.detect_objects)
        self.detect_button.pack(pady=10)

        # Canvas to display image
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        # Label for detected object list
        self.output_label = tk.Label(root, text="", justify="left")
        self.output_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        self.image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((800, 600))
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(400, 300, image=self.tk_image)
        self.output_label.config(text="")

    def detect_objects(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please upload an image first.")
            return

        image = cv2.imread(self.image_path)
        results = self.model(image)[0]

        objects_text = "Detected objects:\n"
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{class_name} {conf:.2f}"
            objects_text += f" - {label}\n"

            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Save and reload annotated image
        annotated_path = "annotated_output.jpg"
        cv2.imwrite(annotated_path, image)

        img = Image.open(annotated_path)
        img.thumbnail((800, 600))
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(400, 300, image=self.tk_image)

        self.output_label.config(text=objects_text)

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorGUI(root)
    root.mainloop()
