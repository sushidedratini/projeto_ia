import numpy as np


def find_max_list(dict_track):
    list_len = [len(inner_dict['CENTROIDS'])
                for inner_dict in dict_track.values()]
    return max(list_len)


def clean_track_history(track_history):
    cleaned_history = []

    for scene in track_history:
        cleaned_scene = {}
        max_size = find_max_list(scene)

        for rider_id, data in scene.items():
            min_size = int(0.5 * max_size)

            if len(data['CENTROIDS']) > min_size:
                cleaned_scene[rider_id] = data

        cleaned_history.append(cleaned_scene)

    return cleaned_history


def simulate_fatigue(x, y, fatigue_factor=0.1, noise_chance=0.2, noise_magnitude=0.5):
    y_variation = np.random.uniform(-fatigue_factor, fatigue_factor)

    if np.random.rand() < noise_chance:
        y_variation += np.random.uniform(-noise_magnitude, noise_magnitude)

    new_y = y + y_variation
    return (x, new_y)


def calculate_movement_statistics(centroids):
    centroids_array = np.array(centroids)
    mean_x = np.mean(centroids_array[:, 0])
    mean_y = np.mean(centroids_array[:, 1])

    # Cálculo das variações em X e Y
    delta_x = np.diff(centroids_array[:, 0]).mean() if len(
        centroids) > 1 else 0
    delta_y = np.diff(centroids_array[:, 1]).mean() if len(
        centroids) > 1 else 0

    return mean_x, mean_y, delta_x, delta_y


def fill_centroid_lists(track_history):
    max_len = max(
        len(data['CENTROIDS']) for scene in track_history for data in scene.values()
    )

    for scene in track_history:
        for _, data in scene.items():
            centroids = data['CENTROIDS']
            if len(centroids) < max_len:
                mean_x, mean_y, delta_x, delta_y = calculate_movement_statistics(
                    centroids)

                for _ in range(len(centroids), max_len):
                    new_x = mean_x + delta_x + np.random.uniform(-1, 1)
                    new_y = simulate_fatigue(mean_x, mean_y + delta_y)[1]
                    centroids.append((new_x, new_y))
                data['CENTROIDS'] = centroids

    return track_history
