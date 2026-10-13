"""
Vehicle gate counter using YOUR Roboflow workflow (Freight Gate Vehicle Deduplication).
Detects + dedups vehicles via the workflow, tracks them, counts line crossings,
and logs every crossing to crossings.csv.

Setup (one time):
    pip install inference-sdk "supervision[assets]"
    export ROBOFLOW_API_KEY=<your NEW key>      # never paste it into the file
Run:
    python count_vehicles_workflow.py
"""
import csv
import os
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from supervision.assets import download_assets, VideoAssets

WORKSPACE = "shreya-palit"
WORKFLOW_ID = "freight-gate-vehicle-deduplication-1791846124550"
FRAME_STRIDE = 3      # send every 3rd frame 
MAX_FRAMES = 300      # stop after this many source frames for a quick first run; None = full video
# One counting line per side of the road (fractions of frame width/height).
# Toward camera (left side): vehicles enter at the top, so the line sits low.
# Away from camera (right side): vehicles enter at the bottom and are only detected
# once farther away, so the line sits higher.
MEDIAN_X = 0.48
TOWARD_Y = 0.75
AWAY_Y = 0.60

# Connecting to Roboflow
SOURCE = download_assets(VideoAssets.VEHICLES)
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=os.environ["ROBOFLOW_API_KEY"])
client.configure(InferenceConfiguration(api_key_transport="header"))

info = sv.VideoInfo.from_video_path(SOURCE)
out_info = sv.VideoInfo(width=info.width, height=info.height, fps=info.fps / FRAME_STRIDE)
tracker = sv.ByteTrack(frame_rate=int(out_info.fps))
anchors = [sv.Position.BOTTOM_CENTER]  # count when the vehicle's bottom-center crosses
W, H = info.width, info.height
lines = {
    "toward_camera": sv.LineZone(start=sv.Point(0, int(H * TOWARD_Y)), end=sv.Point(int(W * MEDIAN_X), int(H * TOWARD_Y)), triggering_anchors=anchors),
    "away_from_camera": sv.LineZone(start=sv.Point(int(W * MEDIAN_X), int(H * AWAY_Y)), end=sv.Point(W, int(H * AWAY_Y)), triggering_anchors=anchors),
}
box_ann, label_ann, line_ann = sv.BoxAnnotator(), sv.LabelAnnotator(), sv.LineZoneAnnotator()

frames = sv.get_video_frames_generator(SOURCE, stride=FRAME_STRIDE, end=MAX_FRAMES)
with open("crossings.csv", "w", newline="") as f, sv.VideoSink("gate_annotated.mp4", out_info) as sink:
    log = csv.writer(f)
    log.writerow(["video_time_s", "tracker_id", "vehicle_type", "line"])
    for i, frame in enumerate(frames):
        result = client.run_workflow(workspace_name=WORKSPACE, workflow_id=WORKFLOW_ID,
                                     images={"image": frame})[0]
        if i == 0:
            print("Workflow outputs:", list(result.keys()))
        detections = sv.Detections.from_inference(result["predictions"])
        detections = tracker.update_with_detections(detections)
        t = round(i * FRAME_STRIDE / info.fps, 2)
        for name, zone in lines.items():
            crossed_in, crossed_out = zone.trigger(detections)
            for idx in range(len(detections)):
                if crossed_in[idx] or crossed_out[idx]:
                    log.writerow([t, detections.tracker_id[idx], detections.data["class_name"][idx], name])

        labels = [f"#{tid} {c}" for tid, c in zip(detections.tracker_id, detections.data["class_name"])]
        frame = box_ann.annotate(frame.copy(), detections)
        frame = label_ann.annotate(frame, detections, labels)
        for zone in lines.values():
            frame = line_ann.annotate(frame, line_counter=zone)
        sink.write_frame(frame)

for name, zone in lines.items():
    print(f"{name}: {zone.in_count + zone.out_count} vehicles")
print("Saved gate_annotated.mp4 and crossings.csv")
