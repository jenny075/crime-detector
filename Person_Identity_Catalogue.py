import argparse
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
import torch.nn.functional as F
import random
import numpy as np
from collections import defaultdict
import json

# ----------------------------
# --- Set seed for reproducibility
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------

# YOLOv8 model
model = YOLO('yolov8s.pt')  # pretrained on COCO

# --- Step 3: ReID model ---
reid_model = resnet18(pretrained=True)
reid_model.fc = torch.nn.Identity()
reid_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
reid_model.to(device)

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def extract_embedding(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return None
    img = preprocess(person_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(img)
        feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu()

def assign_global_ids(all_detections, threshold=0.5):
    global_id_counter = 0
    global_features = []
    for det in all_detections:
        feat = det.get("embedding")
        if feat is None:
            det["global_id"] = -1
            continue
        assigned = False
        for gf in global_features:
            sim = F.cosine_similarity(feat, gf["feature"]).item()
            if sim > threshold:
                det["global_id"] = gf["global_id"]
                assigned = True
                break
        if not assigned:
            det["global_id"] = global_id_counter
            global_features.append({"global_id": global_id_counter, "feature": feat})
            global_id_counter += 1
    return all_detections

# --- Detection and tracking ---
from ultralytics.yolo.utils.trackers.byte_tracker import BYTETracker

def detect_people_in_clip(video_path, conf_thres=0.5, draw_boxes=False, output_dir=None, label_type='local'):
    clip_id = Path(video_path).stem
    results = model.predict(source=str(video_path), conf=conf_thres, classes=[0], stream=True)

    all_detections = []
    out = None
    if draw_boxes and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_dir / f"{clip_id}_annotated.mp4"), fourcc, fps, (width, height))
        cap.release()

    tracker = BYTETracker(frame_rate=30)

    for frame_idx, frame_result in enumerate(results):
        frame = frame_result.orig_img.copy()
        dets_for_tracker = []
        for box in frame_result.boxes:
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            dets_for_tracker.append(bbox + [conf])

        tracks = tracker.update(dets_for_tracker, frame.shape[:2])

        for track in tracks:
            x1, y1, x2, y2, track_id = track[0], track[1], track[2], track[3], int(track[4])
            det_entry = {
                'clip_id': clip_id,
                'frame_id': frame_idx,
                'bbox': [x1, y1, x2, y2],
                'confidence': float(track[5]) if len(track) > 5 else 1.0,
                'local_person_id': track_id
            }

            emb = extract_embedding(frame, det_entry['bbox'])
            det_entry["embedding"] = emb

            all_detections.append(det_entry)

            if draw_boxes and out:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                if label_type == 'local':
                    label = f"Local {track_id}"
                else:
                    # show global ID if available
                    gid = det_entry.get("global_id", -1)
                    label = f"Global {gid}" if gid >= 0 else f"Local {track_id}"
                cv2.putText(frame, label, (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if draw_boxes and out:
            out.write(frame)

    if draw_boxes and out:
        out.release()
        print(f"✅ Annotated video saved to {output_dir / f'{clip_id}_annotated.mp4'}")

    return all_detections

def process_path(path, draw_boxes=False, output_dir=None, label_type='local'):
    path = Path(path)
    video_paths = []

    if path.is_dir():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_paths.extend(path.glob(ext))
        if not video_paths:
            raise ValueError(f"No video clips found in directory {path}")
    elif path.is_file():
        video_paths.append(path)
    else:
        raise ValueError(f"Path {path} is invalid")

    all_results = []
    for video_path in video_paths:
        print(f"Processing {video_path} ...")
        detections = detect_people_in_clip(video_path, draw_boxes=draw_boxes, output_dir=output_dir, label_type=label_type)
        all_results.extend(detections)

    all_results = assign_global_ids(all_results, threshold=0.5)
    return all_results

# --- Step 4: hierarchical person catalogue ---
def hierarchical_person_catalogue(df):
    catalogue = defaultdict(list)
    for gid, group in df.groupby('global_id'):
        clips_list = []
        for clip_id, clip_group in group.groupby('clip_id'):
            frame_ranges = []
            local_ids = []
            sorted_frames = sorted(clip_group['frame_id'])
            start = prev = sorted_frames[0]
            for f in sorted_frames[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    frame_ranges.append((start, prev))
                    start = prev = f
            frame_ranges.append((start, prev))
            local_ids = list(clip_group['local_person_id'].unique())
            clips_list.append({
                'clip_id': clip_id,
                'frame_ranges': frame_ranges,
                'local_person_ids': local_ids
            })
        catalogue[gid] = clips_list
    return catalogue

def main():
    parser = argparse.ArgumentParser(description="Detect, track, and assign global IDs to people in videos")
    parser.add_argument('--path', type=str, required=True, help='Path to video clip or directory')
    parser.add_argument('--draw', action='store_true', help='Draw bounding boxes and save annotated videos')
    parser.add_argument('--output_dir', type=str, default='annotated_clips', help='Directory to save annotated videos')
    parser.add_argument('--label_type', type=str, choices=['local','global'], default='local', help='Type of label to draw on boxes')
    args = parser.parse_args()

    all_detections = process_path(args.path, draw_boxes=args.draw, output_dir=args.output_dir if args.draw else None, label_type=args.label_type)

    for det in all_detections:
        det.pop("embedding", None)

    df = pd.DataFrame(all_detections)

    # Step 4: hierarchical catalogue
    catalogue = hierarchical_person_catalogue(df)

    # Print JSON to console
    print(json.dumps(catalogue, indent=2))

if __name__ == "__main__":
    main()