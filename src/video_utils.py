import os
import glob
import uuid
from typing import List, Dict
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.editor import VideoFileClip

VIDEOS_TO_DOWNLOAD = [
    "https://www.youtube.com/watch?v=259yqIiW4iE",
    "https://www.youtube.com/watch?v=ywwkgZIam0U",
    "https://www.youtube.com/watch?v=SW7BYOhq6sQ",
    "https://www.youtube.com/watch?v=c-Ayhzav98M",
    "https://www.youtube.com/watch?v=6S46iHMqDPo",
    "https://www.youtube.com/watch?v=kd2d2PIwyN0",
    "https://www.youtube.com/watch?v=ZodCN9zhdq0",
    "https://www.youtube.com/watch?v=PWIn9lWFNZ4",
    "https://www.youtube.com/watch?v=DMUUV4BSoBI",
    "https://www.youtube.com/watch?v=pzyf4576Mzs",
    "https://www.youtube.com/watch?v=m0jPQ1saWYw"
]

def generate_subclip(path) -> None:
    folder_path = os.path.abspath("data/trimmed_videos")

    # Load the video
    video = VideoFileClip(path)
    resized_video = video.resize((640, 360))
    no_audio_video = resized_video.without_audio()
    duration = no_audio_video.duration
    trimmed_video = no_audio_video.subclip(5, duration - 5)

    # Generate a random UUID for the filename
    file_name = f"{uuid.uuid4()}.mp4"
    file_path_name = os.path.join(folder_path, file_name)

    # Save the trimmed video
    trimmed_video.write_videofile(file_path_name, codec="libx264")

def prepare_and_download_videos() -> List[str]:
    folder_name = "data/downloaded_videos"
    folder_path = os.path.abspath(folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    video_paths = []
    for url in VIDEOS_TO_DOWNLOAD:
        try:
            # Create a YouTube object
            yt = YouTube(url, on_progress_callback = on_progress)

            # Get the title of the video and construct the expected file path
            video_title = yt.title
            expected_filename = f"{video_title}.mp4"
            expected_file_path = os.path.join(folder_path, expected_filename)

            # Check if the file already exists
            if os.path.exists(expected_file_path):
                print(f"Video '{video_title}' already downloaded. Skipping...")
                video_paths.append(expected_file_path)
                continue

            # Download the highest resolution video available
            stream = yt.streams.get_highest_resolution()
            # Save the video in the specified folder
            video_path = stream.download(output_path=folder_path)
            # Append the full path of the downloaded video to the list
            video_paths.append(video_path)
        except Exception as e:
            print(f"Failed to download video from {url}. Error: {e}")

    return video_paths

def prepare_dict_trimmed_videos() -> List[Dict]:
    folder_name = "data/trimmed_videos"
    folder_path = os.path.abspath(folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if len(os.listdir(folder_path)) > 0:
        files = glob.glob(f'{folder_path}/*')
        for f in files:
            os.remove(f)

    video_paths = prepare_and_download_videos()
    video_paths = glob.glob("data/downloaded_videos/*.mp4")

    for path in video_paths:
        generate_subclip(path)

    videos_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(folder_path, filename)
            videos_list.append({filename: video_path})
    return videos_list
