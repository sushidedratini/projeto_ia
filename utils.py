import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_rider_tracks(race_df) -> None:
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    unique_races = race_df['scene_id'].unique()

    for race_id in unique_races:
        race_data = race_df[race_df['scene_id'] == race_id]
        rider_ids = race_data['rider_id'].unique()
        video_id = race_data['video_id'].iloc[0]

        # # Plotting the X graph
        # plt.figure(figsize=(12, 6))
        # for rider_id in rider_ids:
        #     rider_data = race_data[race_data['rider_id'] == rider_id]
        #     plt.plot(range(len(rider_data)),
        #              rider_data['x'], label=f'Rider {rider_id}')

        # plt.title(f'X-Axis Movement for Race {race_id} in Video {video_id}')
        # plt.xlabel('Element Number (Frame)')
        # plt.ylabel('X Coordinate')
        # plt.legend()
        # plt.grid(True)

        # # Save the X graph
        # x_graph_filename = f"graphs/{video_id}_{race_id}_x.png"
        # plt.savefig(x_graph_filename)
        # plt.close()

        # Plotting the Y graph
        plt.figure(figsize=(12, 6))
        for rider_id in rider_ids:
            rider_data = race_data[race_data['rider_id'] == rider_id]
            plt.plot(range(len(rider_data)),
                     rider_data['y_values'], label=f'Rider {rider_id}')

        plt.title(f'Y-Axis Movement for Race {race_id} in Video {video_id}')
        plt.xlabel('Element Number (Frame)')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)

        # Save the Y graph
        y_graph_filename = f"graphs/{video_id}_{race_id}_y.png"
        plt.savefig(y_graph_filename)
        plt.close()


def save_to_csv(df, file_path) -> None:
    df.to_csv(file_path, index=False)


def create_race_dataset(all_track_histories) -> pd.DataFrame:
    data = []

    for video_id, track_history in enumerate(all_track_histories):
        for scene_id, scene in enumerate(track_history):
            for rider_id, info in scene.items():
                y_values = list(list(zip(*info['CENTROIDS']))[1]) # Extract Y values as a list
                winner = info['WINNER']
                data.append({
                    'video_id': video_id,
                    'scene_id': scene_id,
                    'rider_id': rider_id,
                    'y_values': y_values,
                    'winner': winner
                })

    race_df = pd.DataFrame(data)
    return race_df

