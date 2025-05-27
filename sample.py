import depthai as dai

p = dai.Pipeline()

# Same stereo + YOLO graph you built before
monoL = p.createMonoCamera(); monoR = p.createMonoCamera()
for m,s in [(monoL, dai.CameraBoardSocket.LEFT),
            (monoR, dai.CameraBoardSocket.RIGHT)]:
    m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    m.setBoardSocket(s)

stereo = p.createStereoDepth()
monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

yolo = p.createYoloSpatialDetectionNetwork()
yolo.setBlobPath("yolov8n_openvino.blob")
stereo.depth.link(yolo.inputDepth)

# Send results to host
xout = p.createXLinkOut(); xout.setStreamName("detections")
yolo.out.link(xout.input)

with dai.Device(p) as dev:
    q = dev.getOutputQueue("detections")
    while True:
        for det in q.get().detections:
            print(f"{det.label} at {det.spatialCoordinates.z/1000:.2f} m")
