# 🧠 Scanventory: Auto-inventory household items from images or video, aimed at simplifying insurance documentation and personal property tracking

This project is a simple web app built with [Gradio](https://www.gradio.app/) that allows users to upload an image and see object detections using a custom model. The app annotates the image with bounding boxes and labels and displays the detected object list.

---

## 🚀 Features

- Upload any image via browser
- Detect and annotate objects using YOLO based custom vision model
- Visualize the output with bounding boxes and confidence scores
- See a list of all detected objects
- No need to write any frontend code!

---

## Online Version

An online version is available at https://huggingface.co/spaces/chaparalak/Scanventory.

---

## 🛠️ Requirements

- Python 3.8+
- `ultralytics`
- `gradio`
- `opencv-python`
- `numpy`

Install dependencies:

```bash
pip install ultralytics gradio opencv-python numpy
