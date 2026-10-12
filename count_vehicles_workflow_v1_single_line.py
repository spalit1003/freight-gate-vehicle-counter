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
FRAME_STRIDE = 3      # send every 3rd frame (saves API credits); 1 = every frame
MAX_FRAMES = 300      # stop after this many source frames for a quick first run; None = full video
LINE_Y = 0.75         # counting line position (fraction of frame height from top)

SOURCE = download_assets(VideoAssets.VEHICLES)
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com",
                             api_key=os.environ["ROBOFLOW_API_KEY"])
client.configure(InferenceConfiguration(api_key_transport="header"))

info = sv.VideoInfo.from_video_path(SOURCE)
out_info = sv.VideoInfo(width=info.width, height=info.height, fps=info.fps / FRAME_STRIDE)
tracker = sv.ByteTrack(frame_rate=int(out_info.fps))
line = sv.LineZone(start=sv.Point(0, int(info.height * LINE_Y)),
                   end=sv.Point(info.width, int(info.height * LINE_Y)),
                   triggering_anchors=[sv.Position.BOTTOM_CENTER])  # count when the vehicle's bottom-center crosses
box_ann, label_ann, line_ann = sv.BoxAnnotator(), sv.LabelAnnotator(), sv.LineZoneAnnotator()

frames = sv.get_video_frames_generator(SOURCE, stride=FRAME_STRIDE, end=MAX_FRAMES)
with open("crossings.csv", "w", newline="") as f, sv.VideoSink("gate_annotated.mp4", out_info) as sink:
    log = csv.writer(f)
    log.writerow(["video_time_s", "tracker_id", "vehicle_type", "direction"])
    for i, frame in enumerate(frames):
        result = client.run_workflow(workspace_name=WORKSPACE, workflow_id=WORKFLOW_ID,
                                     images={"image": frame})[0]
        if i == 0:
            print("Workflow outputs:", list(result.keys()))
        detections = sv.Detections.from_inference(result["predictions"])
        detections = tracker.update_with_detections(detections)
        crossed_in, crossed_out = line.trigger(detections)

        t = round(i * FRAME_STRIDE / info.fps, 2)
        for idx in range(len(detections)):
            if crossed_in[idx] or crossed_out[idx]:
                log.writerow([t, detections.tracker_id[idx], detections.data["class_name"][idx],
                              "in" if crossed_in[idx] else "out"])

        labels = [f"#{tid} {c}" for tid, c in zip(detections.tracker_id, detections.data["class_name"])]
        frame = box_ann.annotate(frame.copy(), detections)
        frame = label_ann.annotate(frame, detections, labels)
        sink.write_frame(line_ann.annotate(frame, line_counter=line))

print(f"In: {line.in_count}  Out: {line.out_count}")
print("Saved gate_annotated.mp4 and crossings.csv")
