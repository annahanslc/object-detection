from ultralytics import YOLO
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.yolo import YOLOv5DatasetExporter
# Use this line to get access to export types
from fiftyone import types

# # Download only detection-labeled images for squirrels
# dataset = foz.load_zoo_dataset(
#     "open-images-v7",
#     split="train",
#     label_types=["detections"],
#     classes = ["Squirrel"],
#     max_samples=1000,
# )
#
# dataset.export(
#     export_dir="squirrel_yolo",
#     dataset_type=types.YOLOv5Dataset,
#     label_field="ground_truth",  # default field from Open Images
# )

# Load a model
model = YOLO("yolov8n.yaml") # build a new model from scratch

# Use the model
results = model.train(data='config.yaml', epochs=1)