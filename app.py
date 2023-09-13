import gradio as gr
from pathlib import Path
import PIL
from webapp import settings, helper

classes = {0.0: 'Person', 1.0: 'Car', 2.0: 'Bicycle', 3.0: 'OtherVehicle', 4.0: 'DontCare'}


def object_detection(image, confidence):
    model_path = Path(settings.DETECTION_MODEL)
    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
        results = model(image, conf=confidence)
        boxes = results[0].boxes
        res_plotted = results[0].plot()[:, :, ::-1]
        print(res_plotted)
        return res_plotted, [classes[box.cls.item()] for box in boxes]
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
    title="HUMAN Detection using YOLOv8 with Thermal SENSOR",
    description="A Gradio interface for object detection using YOLOv8. model on thermal image snapshots"
)


if __name__ == "__main__":
    iface.launch()

