from collections import defaultdict
from scipy.spatial import distance

import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

'''
CLASSNAMES

{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 
48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 
76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
'''

def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)

def normalize_centroids(tracks, frame_shape):
    frame_height, frame_width = frame_shape

    for track_id, centroids in tracks.items():
        normalized_centroids = [(round(x / frame_width, 3), round(y / frame_height, 3)) for x, y in centroids]
        tracks[track_id] = normalized_centroids

    return tracks

def extend_tracks(tracks, tiredness_factor=0.05):
    # Find the length of the longest list
    max_length = max(len(centroids) for centroids in tracks.values())

    for track_id, centroids in tracks.items():
        current_length = len(centroids)

        if current_length < max_length:
            # Calculate variations in x and y coordinates
            x_coords = [coord[0] for coord in centroids]
            y_coords = [coord[1] for coord in centroids]

            if current_length > 1:
                x_variation = (x_coords[-1] - x_coords[0]) / (current_length - 1)
                y_variation = (y_coords[-1] - y_coords[0]) / (current_length - 1)
            else:
                x_variation = 0
                y_variation = 0

            # Determine the pattern of y movement
            y_diff = np.diff(y_coords)
            up_frames = np.sum(y_diff > 0)
            down_frames = np.sum(y_diff < 0)

            # Generate new x, y values to fill the empty positions
            for i in range(current_length, max_length):
                new_x = x_coords[-1] + x_variation

                # Determine if the y value should increase or decrease
                if up_frames > down_frames:
                    new_y = y_coords[-1] + y_variation * (1 + tiredness_factor * (i / max_length))
                    up_frames -= 1  # Use one up-frame
                else:
                    new_y = y_coords[-1] - y_variation * (1 + tiredness_factor * (i / max_length))
                    down_frames -= 1  # Use one down-frame

                centroids.append((new_x, new_y))
                x_coords.append(new_x)
                y_coords.append(new_y)

    return tracks

def create_race_dataset(tracks, frame_shape):
    data = []

    # Exnend all tracks that have short length
    tracks = extend_tracks(tracks)

    # Normalize the centroids
    tracks = normalize_centroids(tracks, frame_shape)

    for track_id, centroids in tracks.items():
        row = {
            'id': track_id,
            'winner': 0,
            'centroids': centroids
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df

def calculate_centroid(bbox):
    x, y, w, h = bbox.xywh[0]
    return (int(x + w / 2), int(y + h / 2))

def initialize_tracks(detections, frame_shape):
    tracks = {}
    track_id = 0
    boxes = detections.boxes
    for box in boxes:
        centroid = calculate_centroid(box)
        tracks[track_id] = [centroid]
        track_id += 1
    return tracks

def update_tracks(tracks, detections, frame_shape):
    new_tracks = defaultdict(list)
    used_ids = set()
    
    boxes = detections.boxes
    for box in boxes:
        centroid = calculate_centroid(box)
        min_distance = float('inf')
        assigned_id = None

        # Find the closest existing track
        for track_id, centroids in tracks.items():
            if track_id in used_ids:
                continue  # Skip already used tracks

            dist = distance.euclidean(centroid, centroids[-1])
            if dist < min_distance:
                min_distance = dist
                assigned_id = track_id

        if assigned_id is not None:
            new_tracks[assigned_id] = tracks[assigned_id] + [centroid]
            used_ids.add(assigned_id)
        else:
            # Create a new track ID
            new_id = max(tracks.keys(), default=-1) + 1
            new_tracks[new_id] = [centroid]
            used_ids.add(new_id)

    # Merge with existing tracks
    for track_id, centroids in tracks.items():
        if track_id not in new_tracks:
            new_tracks[track_id] = centroids

    return new_tracks

def main() -> None:

    # Load the YOLOv8 model
    model = YOLO('yolov8m.pt')

    cap = cv2.VideoCapture('corrida.mp4')
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"# of Frames: {num_frames}")
    tracks = {}

    # Store the track history
    track_history = defaultdict(lambda: [])
    new_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Get only class 17 (horse)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", show_conf=True, classes=[0, 17])

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywhn.cpu()

        if results[0].boxes.id is None: break
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Lambda function to generate list of 1024 None values if ID is not in dictionary
        generate_none_list = lambda x: new_history.setdefault(x, [None] * num_frames)

        # Apply the lambda function to each ID in the list
        for id in track_ids:
            generate_none_list(id)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            new_history[track_id][curr_frame] = (float(x), float(y))

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
                
            break

        for result in results:
            if not tracks:
                tracks = initialize_tracks(result, (width, height))
            else:
                tracks = update_tracks(tracks, result, (width, height))

        cv2.imshow("Projeto Final IA", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    race_df = create_race_dataset(tracks, (width, height))

    # Save the dataset to a CSV file
    save_to_csv(race_df, 'race_dataset.csv')

if __name__ == '__main__':
    main()