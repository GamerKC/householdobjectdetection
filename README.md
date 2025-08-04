# 🧠 YOLO Object Detection with Gradio GUI

This project is a simple web app built with [Gradio](https://www.gradio.app/) that allows users to upload an image and see object detections using a YOLOv8 model (`yolo11m.pt` by default). The app annotates the image with bounding boxes and labels and displays the detected object list.

---

## 🚀 Features

- Upload any image via browser
- Detect and annotate objects using YOLOv8
- Visualize the output with bounding boxes and confidence scores
- See a list of all detected objects
- No need to write any frontend code!

---

## 🖼️ Example

![Example Screenshot](example.jpg)

---

## 🛠️ Requirements

- Python 3.8+
- `ultralytics` (YOLOv8)
- `gradio`
- `opencv-python`
- `numpy`

Install dependencies:

```bash
pip install ultralytics gradio opencv-python numpy
