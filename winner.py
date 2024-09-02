import numpy as np


def determine_direction(centroids, num_points=5):
    if len(centroids) < num_points:
        num_points = len(centroids)

    recent_points = np.array(centroids[-num_points:])
    delta = recent_points[-1] - recent_points[0]  # Vetor de movimento

    angle = np.arctan2(delta[1], delta[0]) * 180 / \
        np.pi  # Calcula o ângulo em graus

    # Determina a direção com base no ângulo
    if -22.5 <= angle < 22.5:
        direction = 'right'
    elif 22.5 <= angle < 67.5:
        direction = 'right-up'
    elif 67.5 <= angle < 112.5:
        direction = 'up'
    elif 112.5 <= angle < 157.5:
        direction = 'left-up'
    elif -67.5 <= angle < -22.5:
        direction = 'right-down'
    elif -112.5 <= angle < -67.5:
        direction = 'down'
    elif -157.5 <= angle < -112.5:
        direction = 'left-down'
    else:
        direction = 'left'

    return direction


def add_direction_to_track_history(track_history, num_points=5):
    for scene in track_history:
        for _, info in scene.items():
            direction = determine_direction(
                info['CENTROIDS'], num_points=num_points)
            info['DIRECTION'] = direction


def calculate_distance_to_margin(centroid, direction, frame_width, frame_height):
    x, y = centroid
    if direction == 'right':
        target = (frame_width, y)
    elif direction == 'right-up':
        target = (frame_width, 0)
    elif direction == 'up':
        target = (x, 0)
    elif direction == 'left-up':
        target = (0, 0)
    elif direction == 'left':
        target = (0, y)
    elif direction == 'left-down':
        target = (0, frame_height)
    elif direction == 'down':
        target = (x, frame_height)
    elif direction == 'right-down':
        target = (frame_width, frame_height)
    else:
        return float('inf')

    return np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)


def determine_winner(track_history, frame_width, frame_height):
    for scene in track_history:
        min_distance = float('inf')
        winner_id = None

        for rider_id, info in scene.items():
            last_centroid = info['CENTROIDS'][-1]
            direction = info['DIRECTION']
            distance = calculate_distance_to_margin(
                last_centroid, direction, frame_width, frame_height)

            if distance < min_distance:
                min_distance = distance
                winner_id = rider_id

        if winner_id is not None:
            scene[winner_id]['WINNER'] = 1
