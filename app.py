import gradio as gr
from pathlib import Path
import PIL
from webapp import settings, helper

def object_detection(image, confidence):
    model_path = Path(settings.DETECTION_MODEL)
    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
        results = model(image, conf=confidence)
        boxes = results[0].boxes
        try:
            print(results[0])
        except Exception as e:
            print(e, "Wrong get attribute")
        res_plotted = results[0].plot()[:, :, ::-1]
        return res_plotted, [box.data for box in boxes]
    except Exception as ex:
        print(f"Unable to load model. Check the specified path: {model_path}\n{ex}")


iface = gr.Interface(
    fn=object_detection,
    inputs=[
        gr.components.Image(type="pil", label="Choose an image..."),
        gr.components.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Select Model Confidence")
    ],
    outputs=[
        gr.components.Image(type="pil", label="Detected Image"),
        gr.components.Textbox(type="text", label="Detection Results")
    ],
    title="Object Detection using YOLOv8",
    description="A Gradio interface for object detection using YOLOv8."
)


if __name__ == "__main__":
    iface.launch()

