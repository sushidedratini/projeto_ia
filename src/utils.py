import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_rider_tracks(race_df) -> None:
    """
    Plots the Y movement for each rider in a scene. 
    Saves the graphs in a folder named 'graphs', with filenames based on video_id and scene_id.
    """
    # Create 'graphs' folder if it doesn't exist
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    # Get the unique scenes by video_id and scene_id
    unique_scenes = race_df[['video_id', 'scene_id']].drop_duplicates()

    # Loop through each unique scene (video_id, scene_id)
    for _, row in unique_scenes.iterrows():
        video_id = row['video_id']
        scene_id = row['scene_id']

        # Filter data for the current scene
        scene_data = race_df[(race_df['video_id'] == video_id) & (race_df['scene_id'] == scene_id)]

        # Start plotting
        plt.figure(figsize=(12, 6))

        for _, rider_row in scene_data.iterrows():
            rider_id = rider_row['rider_id']
            y_values = rider_row['y_values']

            # Plot the y_values over time (index serves as the time/sequence)
            plt.plot(range(len(y_values)), y_values, label=f'Rider {rider_id}')
        
        plt.title(f'Y-Axis Movement for Scene {scene_id} in Video {video_id}')
        plt.xlabel('Time (Frame Index)')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)

        # Save the plot
        filename = f"graphs/{video_id}_{scene_id}_y.png"
        plt.savefig(filename)
        plt.close()

def plot_roc_curve_graph(fpr, tpr, auc_score):
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save the plot
    filename = "graphs/roc_curve_score_graph.png"
    plt.savefig(filename)
    plt.close()


def save_to_csv(df, file_path) -> None:
    df.to_csv(file_path, index=False)


def create_race_dataset(all_track_histories) -> pd.DataFrame:
    data = []

    for video_id, track_history in enumerate(all_track_histories):
        for scene_id, scene in enumerate(track_history):
            for rider_id, info in scene.items():
                y_values = list(list(zip(*info['CENTROIDS']))[1]) # Extract Y values as a list
                data.append({
                    'video_id': video_id,
                    'scene_id': scene_id,
                    'rider_id': rider_id,
                    'y_values': y_values,
                    'speeds': info['SPEEDS'],  # Lista de velocidades
                    'direction_vector': info['DIRECTION_VECTOR'],  # Lista de vetores de direção
                    'winner': info['WINNER']  # Se este corredor foi o vencedor
                })

    race_df = pd.DataFrame(data)
    return race_df

