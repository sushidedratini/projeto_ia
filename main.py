import ast
import sys
from typing import List, Dict
import cv2
import torch
import torchvision
import pandas as pd
import numpy as np

from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

from utils import create_race_dataset, plot_rider_tracks, save_to_csv # type: ignore
from post_operations import clean_track_history, fill_centroid_lists
from video_utils import prepare_dict_trimmed_videos
from winner import add_direction_to_track_history, determine_winner


def detect_camera_change(frame1, frame2, threshold=0.5):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute Structural Similarity Index (SSIM)
    score, _ = ssim(gray1, gray2, full=True)
    print(score)

    # Check if the SSIM score is below the threshold
    if score < threshold:
        return True  # Camera change detected
    return False  # No significant change


def calculate_centroid(bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min.item() + x_max.item()) / 2
    y_center = (y_min.item() + y_max.item()) / 2
    return (x_center, y_center)


def normalize_centroids(tracks: List[Dict], width, height):

    for track_dict in tracks:
        for _, values in track_dict.items():
            normalized_centroids = [
                (round(x / width, 3), round(y / height, 3)) for x, y in values['CENTROIDS']]
            values['CENTROIDS'] = normalized_centroids

    return tracks


def calculate_iou(bbox1, bbox2):
    box_1_xyxy = bbox1.xyxy.cpu()
    box_2_xyxy = bbox2.xyxy.cpu()

    iou = torchvision.ops.box_iou(box_1_xyxy, box_2_xyxy)

    return iou


def best_iou_found(obj_bbox, iter_bboxes, threshold) -> bool:
    best_obj = None
    best_iou = 0

    for bbox in iter_bboxes:
        iou = calculate_iou(obj_bbox, bbox)
        if iou > best_iou:
            best_iou = iou
            best_obj = bbox

    if best_obj and best_iou >= threshold:
        return True
    return False


def track_riders(herd, people, current_track):
    for person in people:

        person_bbox_xyxy = person.xyxy.cpu()
        person_bbox_centroid = calculate_centroid(person_bbox_xyxy[0])

        # Person and Horse verification
        # If false, skip as person is not a horse rider
        if best_iou_found(person, herd, threshold=0.4) and person.id is not None:
            person_id: int = person.id.item()
            if person_id in current_track:
                current_track[person_id]['CENTROIDS'].append(
                    person_bbox_centroid)
                current_track[person_id]['LAST_BBOX'] = person
                current_track[person_id]['WINNER'] = 0
            else:
                best_iou = 0
                best_id = None
                for key, value in current_track.items():
                    temp_bbox = value['LAST_BBOX']
                    iou = calculate_iou(person, temp_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = key

                if best_id and best_iou >= 0.5:
                    current_track[int(best_id)]['CENTROIDS'].append(
                        person_bbox_centroid)
                    current_track[int(best_id)]['LAST_BBOX'] = person
                    current_track[int(best_id)]['WINNER'] = 0

                # New person?
                else:
                    current_track[person_id] = {
                        'CENTROIDS': [person_bbox_centroid],
                        'LAST_BBOX': person,
                        'WINNER': 0
                    }

    return current_track


def plot_frame(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # confidence = box.conf.item()  # Confidence score
        if not box.id:
            continue
        box_id = int(box.id.item())

        label = f'{box_id}'  # {confidence:.2f}
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (255, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1 - text_height - baseline),
                      (x1 + text_width, y1), (255, 0, 0), thickness=cv2.FILLED)
        cv2.putText(frame, f'{box_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def get_track_history(model, file_name, test=False):
    cap = cv2.VideoCapture(file_name)
    ret, prev_frame = cap.read()

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    track_history = []
    current_track = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True,
                              show_conf=True, classes=[0, 17], device='cuda')

        if detect_camera_change(prev_frame, frame) and current_track:
            track_history.append(current_track)
            current_track = {}

        herd = [box for box in results[0].boxes if box.cls == 17]  # Horse
        people = [box for box in results[0].boxes if box.cls == 0]  # Person

        current_track = track_riders(herd, people, current_track)

        prev_frame = frame
        annotated_frame = None
        if test is not False:
            annotated_frame = plot_frame(frame, people)
        else:
            annotated_frame = frame

        cv2.imshow("Projeto Final IA", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if current_track:
        track_history.append(current_track)

    cap.release()
    cv2.destroyAllWindows()

    # Determine Winner for all scenes
    add_direction_to_track_history(track_history, num_points=10)
    if test is not False:
        determine_winner(track_history, width, height)

    # Post Operations
    final_list = clean_track_history(track_history)
    final_list = fill_centroid_lists(final_list)
    final_list = normalize_centroids(final_list, width, height)

    return final_list

def convert_to_numeric_list(y_values):
    if isinstance(y_values, str):
        y_values = ast.literal_eval(y_values)
    return [float(value) for value in y_values]

def extract_features_from_sequences(df):
    df['y_values'] = df['y_values'].apply(convert_to_numeric_list)
    df['y_mean'] = df['y_values'].apply(np.mean)
    df['y_std'] = df['y_values'].apply(np.std)
    df['y_min'] = df['y_values'].apply(np.min)
    df['y_max'] = df['y_values'].apply(np.max)
    return df[['y_mean', 'y_std', 'y_min', 'y_max']], df['winner']

def train_and_test(train_file_name, test_file_name) -> None:
    df = pd.read_csv(train_file_name)
    new_video_df = pd.read_csv(test_file_name)

    X, y = extract_features_from_sequences(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training accuracy: {accuracy:.2f}")

    X_new, _ = extract_features_from_sequences(new_video_df)

    new_predictions = model.predict(X_new)

    new_video_df['predicted_winner'] = new_predictions
    new_video_df.to_csv('new_video_predictions.csv', index=False)

def main() -> None:

    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit()

    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO('yolov8m.pt')
    model.to(device=device)

    # Download and trim videos
    video_list: List[Dict] = prepare_dict_trimmed_videos()

    all_track_histories = []
    for dict_video in video_list:
        for _, video_path in dict_video.items():
            track_history = get_track_history(model, video_path, True)
            all_track_histories.append(track_history)

    csv_file_name = 'race_dataset_new.csv'
    race_df = create_race_dataset(all_track_histories)
    # plot_rider_tracks(race_df)
    save_to_csv(race_df, csv_file_name)

    test_track_history = get_track_history(
        model, r"D:\Workspace\IA\ProjetoFinal\test_video_trimmed.mp4")
    test_race_df = create_race_dataset([test_track_history])
    csv_test_file_name = 'test_dataset.csv'
    save_to_csv(test_race_df, csv_test_file_name)

    train_and_test(csv_file_name, csv_test_file_name)


if __name__ == '__main__':
    main()
