from typing import List, Dict

import numpy as np
import cv2
import torch
import torchvision
import sys

from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

from src.camera_operations import calculate_background_motion, calculate_optical_flow, save_optical_flow_image
from src.post_operations import clean_track_history, fill_centroid_lists, fill_speed_lists
from src.train_test import train_and_test
from src.utils import create_race_dataset, plot_roc_curve_graph, save_to_csv
from src.video_utils import generate_subclip, prepare_dict_trimmed_videos
from src.winner import determine_winner_in_scene


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


def track_riders(herd: list, people: list, current_track: dict):
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
    '''This function plots the bounding boxes for the giving bounding boxes'''
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


def calculate_competitor_speed(flow, competitor_coords, background_motion):
    """
    Calcula a velocidade dos competidores subtraindo o movimento do background.
    """
    speeds = []
    for (x, y) in competitor_coords:
        flow_at_competitor = flow[int(y), int(x)]
        relative_motion = flow_at_competitor - background_motion
        speed = np.linalg.norm(relative_motion)
        speeds.append(speed)
    return speeds


def estimate_direction(flow, competitor_coords):
    """
    Estima a direção de movimento dos competidores com base no optical flow.
    """
    directions = []
    for (x, y) in competitor_coords:
        flow_at_competitor = flow[int(y), int(x)]
        directions.append(flow_at_competitor)
    return directions


def filter_track_history_by_speed(track_history):
    """
    Remove entradas no track_history que não possuem a chave 'SPEED'.
    """
    filtered_track_history = []

    for scene in track_history:
        filtered_scene = {}
        for rider_id, info in scene.items():
            # Verifica se o item possui a chave 'SPEED'
            if 'SPEEDS' in info:
                filtered_scene[rider_id] = info

        # Somente adiciona a cena se houver pelo menos um competidor com 'SPEED'
        if filtered_scene:
            filtered_track_history.append(filtered_scene)

    return filtered_track_history


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

        # current_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        results = model.track(frame, persist=True,
                              show_conf=True, classes=[0, 17], device='cuda')

        if detect_camera_change(prev_frame, frame) and current_track:
            track_history.append(current_track)
            current_track = {}

        herd = [box for box in results[0].boxes if box.cls == 17]  # Horse
        people = [box for box in results[0].boxes if box.cls == 0]  # Person

        # Calcular Optical Flow para o cálculo de velocidade e direção
        if prev_frame is not None:
            flow, _ = calculate_optical_flow(prev_frame, frame)
            # new_img = f'data/flow/{index}_{current_frame_number}.png'
            # save_optical_flow_image(flow_img, new_img)
            competitor_coords = [info['CENTROIDS'][-1]
                                 for info in current_track.values() if 'CENTROIDS' in info]

            # Calcular o movimento do background
            background_motion = calculate_background_motion(flow)

            # Calcular a velocidade dos competidores
            speeds = calculate_competitor_speed(
                flow, competitor_coords, background_motion)

            # Estimar a direção dos competidores
            directions = estimate_direction(flow, competitor_coords)

            # Atualizar as velocidades e direções no histórico de track
            for idx, (_, info) in enumerate(current_track.items()):
                if 'CENTROIDS' in info:
                    # Se o campo de velocidade não existir, inicializar com [0]
                    if 'SPEEDS' not in info:
                        info['SPEEDS'] = [0]
                    # Adicionar a nova velocidade ao array de velocidades
                    info['SPEEDS'].append(speeds[idx])

                    # Se o campo de direção não existir, inicializar com a primeira direção
                    if 'DIRECTION_VECTOR' not in info:
                        info['DIRECTION_VECTOR'] = []

                    # Adicionar a nova direção ao array de direção
                    info['DIRECTION_VECTOR'].append(directions[idx])

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
    # add_direction_to_track_history(track_history, num_points=10)

    # Post Operations
    final_list = clean_track_history(track_history)
    final_list = fill_centroid_lists(final_list)
    final_list = normalize_centroids(final_list, width, height)

    filtered_track_history = filter_track_history_by_speed(final_list)
    final_list = fill_speed_lists(filtered_track_history)
    determine_winner_in_scene(final_list)

    return final_list


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
    print(video_list)

    all_track_histories = []
    for dict_video in video_list:
        for _, video_path in dict_video.items():
            track_history = get_track_history(model, video_path, True)
            all_track_histories.append(track_history)

    csv_file_name = 'data/train_dataset.csv'
    race_df = create_race_dataset(all_track_histories)
    # plot_rider_tracks(race_df)
    save_to_csv(race_df, csv_file_name)

    test_track_history = get_track_history(
        model, r"D:\Workspace\IA\ProjetoFinal\data\test_video_2_trimmed.mp4", True)
    test_race_df = create_race_dataset([test_track_history])
    csv_test_file_name = 'data/test_dataset.csv'
    save_to_csv(test_race_df, csv_test_file_name)

    train_and_test(csv_file_name, csv_test_file_name)


if __name__ == '__main__':
    main()
