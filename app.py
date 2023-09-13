import gradio as gr
from pathlib import Path
import PIL
from webapp import settings, helper

# YoLo Model Classes
classes = {0.0: 'Person', 1.0: 'Car', 2.0: 'Bicycle', 3.0: 'OtherVehicle', 4.0: 'DontCare'}


def object_detection(video, confidence):
    """
        Parameters:
            video (pil image array): snaps of the video 
            consfidence (int): confidence value for the model

        Retures:
            res_plotted (pil image array): masked image from YoLo model
            class_labels: prediction labels from YoLo
    """

    model_path = Path(settings.DETECTION_MODEL) 

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
        results = model(video, conf=confidence)
        boxes = results[0].boxes
        res_plotted = results[0].plot()[:, :, ::-1]
        class_labels = [classes[box.cls.item()] for box in boxes]
        return res_plotted, len(class_labels)
    except Exception as ex:
        print(f"Unable to load model. Check the specified path: {model_path}\n{ex}")


# Create a Gradio interface
iface = gr.Interface(
    fn=object_detection,
    inputs=[
        gr.components.Image(source="webcam", streaming=True),
        gr.components.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Select Model Confidence"),
    ],
    outputs=[
        gr.components.Image(type="pil", label="Detected Image"),
        gr.components.Textbox(type="text", label="Detection Results"),
    ],
    title="HUMAN Detection using YOLOv8 with Thermal SENSOR",
    description="A Gradio interface for object detection using YOLOv8. model on thermal image snapshots",
    live=True
)

if __name__ == "__main__":
    iface.launch()

