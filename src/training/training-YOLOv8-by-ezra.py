import yaml
import wandb
from ultralytics import YOLO

# PATH
CFG = "/home/ezra/Desktop/computer-vision/saauvc/configs/yolov8-auv-vision-tracking.yaml"
DATA_YAML = "/home/ezra/Desktop/computer-vision/saauvc/data/roboflow1.yaml"
DEVICE = 0

ENTITY = "visionAmarine"
PROJECT = "YOLOv8-ezra"

# LOAD RUN NAME FROM YAML
with open(CFG, "r") as f:
    cfg_data = yaml.safe_load(f)

RUN = cfg_data.get("name", "experiment-unnamed")

# INIT WANDB
wandb.init(
    entity=ENTITY,
    project=PROJECT,
    name=RUN,
    job_type="training",
    config=cfg_data  
)

# LOAD MODEL
model = YOLO("yolov8n.pt")

# TRAINING
model.train(
    data=DATA_YAML,
    cfg=CFG,
    device=DEVICE,
    project=PROJECT,
    name=RUN
)

# VALIDATION
metrics = model.val(
    data=DATA_YAML,
    device=DEVICE
)

# LOG METRICS
wandb.log({
    "mAP50": float(metrics.box.map50),
    "mAP50-95": float(metrics.box.map),
    "Precision": float(metrics.box.mp),
    "Recall": float(metrics.box.mr)
})

wandb.finish()
