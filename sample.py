# depthai-sdk 1.15.1  ·  OAK-D S2  ·  spatial YOLOv8-Nano
# --------------------------------------------------------
from depthai_sdk import OakCamera


def print_xyz(pkt):
    for det in pkt.detections:
        xyz = det.img_detection.spatialCoordinates          # present when spatial=True
        print(f"{det.label_str:<12} {det.confidence:.2f}  "
              f"X:{xyz.x/1000:.2f} m  "
              f"Y:{xyz.y/1000:.2f} m  "
              f"Z:{xyz.z/1000:.2f} m")


with OakCamera() as oak:
    # 1️⃣  cameras
    color  = oak.create_camera('color',  resolution='1080p', fps=5)

    # 2️⃣  spatial YOLO
    nn = oak.create_nn(
        model='yolov8n_coco_640x352',
        input=color,
        nn_type='yolo',
        spatial=True               # <- **the only flag you need**
    )
    nn.config_nn(conf_threshold=0.8)            # set after creation

    # 3️⃣  outputs
    oak.visualize(nn, fps=True, scale=2/3)             # live video window
    oak.callback(nn, print_xyz)                        # console XYZ print-out

    oak.start(blocking=True)                           # press Q to quit
