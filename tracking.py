import os, time, math
os.environ["DEPTHAI_DISABLE_O3D"] = "1"      # skip Open3D if you haven’t installed it

from depthai_sdk import OakCamera
import depthai as dai

MOVE_THRESH = 200         #  • print when object moves ≥20 cm
COOLDOWN    = 1.0         #  • but not more often than 1 s
COCO_LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


last_xyz   : dict[int, dai.SpatialLocationCalculatorConfigData] = {}
last_print : dict[int, float] = {}

# ────── callback ────────────────────────────────────────────────────
def on_tracks(pkt):
    now = time.time()

    for tracklet in pkt.daiTracklets.tracklets:          # ← correct iteration
        trk  = tracklet
        xyz  = trk.spatialCoordinates
        tid  = trk.id
        stat = trk.status.name            # 'NEW' | 'TRACKED' | 'LOST'

        if stat == 'NEW':
            print(f"🆕 id {tid}  {COCO_LABELS[trk.label]}  "
                  f"X:{xyz.x/1000:+.2f}  Y:{xyz.y/1000:+.2f}  Z:{xyz.z/1000:.2f} m")
            last_xyz[tid]   = xyz
            last_print[tid] = now

        elif stat == 'TRACKED':
            prev = last_xyz.get(tid)
            if prev:
                moved = math.dist(
                    (prev.x, prev.y, prev.z),
                    ( xyz.x,  xyz.y,  xyz.z)
                )
                if moved > MOVE_THRESH and now - last_print[tid] > COOLDOWN:
                    print(f"↔ id {tid:<3} moved {moved/10:.1f} cm  "
                          f"X:{xyz.x/1000:+.2f}  Y:{xyz.y/1000:+.2f}  Z:{xyz.z/1000:.2f} m")
                    last_xyz[tid]   = xyz
                    last_print[tid] = now

        elif stat == 'LOST':
            print(f"❌ id {tid} lost  "
                  f"last X:{xyz.x/1000:+.2f}  Y:{xyz.y/1000:+.2f}  Z:{xyz.z/1000:.2f} m")
            last_xyz.pop(tid, None)
            last_print.pop(tid, None)

# ────── pipeline ────────────────────────────────────────────────────
with OakCamera() as oak:
    rgb = oak.create_camera('color', fps=1)
    nn  = oak.create_nn(
        'yolov6nr3_coco_640x352',
        rgb,
        tracker=True,
        spatial=True
    )

    nn.config_nn(resize_mode='stretch', conf_threshold=0.6)
    nn.config_tracker(
        tracker_type=dai.TrackerType.ZERO_TERM_IMAGELESS,
        # track_labels=[0],                      # persons only
        threshold=0.10,                       # low IOU keeps IDs stable
        forget_after_n_frames=3,             # ~3 s at 1 fps
        assignment_policy=dai.TrackerIdAssignmentPolicy.UNIQUE_ID,
        max_obj=20
    )

    oak.visualize(nn)           # uncomment for a live view
    oak.callback(nn.out.tracker, on_tracks)   # listen to tracklets only
    oak.start(blocking=True)
