import os
import cv2
import numpy as np
import pandas as pd


def load_tracks_from_folder(folder_path):
    """
    Loads tracks from all CSV files in the specified folder.
    Each CSV represents a single track and is assigned a unique track ID.
    Each CSV file must have columns: ['x_topleft', 'y_topleft', 'width', 'height'].

    Args:
        folder_path (str): Path to the folder containing the CSV files.

    Returns:
        tracks (list): A list of tracks, where each track is a dictionary with
                       'track_id' and 'centers' (list of points).
                       Example: [{'track_id': 1, 'centers': [(x1, y1), (x2, y2), ...]}, ...]
    """
    all_tracks = []

    # Iterate over all files in the folder in alphabetical order (to maintain consistent track IDs)
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file (skip the first row)
            track_df = pd.read_csv(file_path, skiprows=1,
                                   names=["track_id", "class_id", "score", "x_topleft", "y_topleft", "width", "height"])

            # Ensure the CSV file has the required columns
            if not {'x_topleft', 'y_topleft', 'width', 'height'}.issubset(track_df.columns):
                print(f"Invalid CSV format in {file_name}, skipping...")
                continue

            # Compute centers for all bounding boxes in this track
            centers = []
            for _, row in track_df.iterrows():
                track_id = row['track_id']
                center_x = int(row['x_topleft'] + row['width'] // 2)
                center_y = int(row['y_topleft'] + row['height'] // 2)
                centers.append((center_x, center_y))

            # Add the track (along with its track ID) to all_tracks
            if centers:
                all_tracks.append({'track_id': track_id, 'centers': centers})


    return all_tracks


def plot_tracks_with_lines(video_path, tracks, output_path=None):
    """
    Plots centers of bounding boxes as circles on video frames and connects points within a track with a line.
    Displays the track ID on the first point of each track.

    Args:
        video_path (str): Path to the input video file.
        tracks (list): A list of tracks, where each track is represented as a dictionary:
                       {'track_id': int, 'centers': [(x1, y1), (x2, y2), ...]}.
        output_path (str, optional): Path to save the processed video. Displays
                                     the video if not provided.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare writer if output_path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames to read

        # Overlay all track points and lines onto the frame
        for track in tracks:
            track_id = track['track_id']
            centers = track['centers']
            prev_point = None

            for idx, point in enumerate(centers):
                # Draw the current point as a circle
                cv2.circle(frame, point, radius=1, color=(0, 255, 255), thickness=-1)  # Yellow filled circle

                # Draw a line connecting the previous point to the current point
                # if prev_point is not None:
                #     cv2.line(frame, prev_point, point, color=(255, 0, 0), thickness=1)  # Blue line
                #
                # # Add the track ID as text on the first point of the track
                # if idx == 0:
                #     cv2.putText(frame, f"Track {track_id}", (point[0] + 5, point[1] - 5),
                #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                #                 color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)  # Green text
                #
                # prev_point = point

        # Display the frame in a window
        cv2.imshow("Processed Video", frame)

        # Save a sample frame for debugging purposes
        if frame_no == 60:
            cv2.imwrite("mock_frame.jpg", frame)

        frame_no += 1

        # Listen for 'q' key press to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write frame to output video if specified
        if output_path:
            writer.write(frame)

    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# Main function
if __name__ == "__main__":
    # Folder containing CSV track files
    folder_path = "C:/transmetric/dev/python/AI_camera/trial/faster-R-CNN-model/tracks/067-00007_Wed_Thur_27hrs_1500/2024_1204_154045_002A/2024_1204_154045_002A_2024_1218_104501_detection"
    # Path to the input video
    video_path = "C:/transmetric/trafficdata/video/067-00007_Wed_Thur_27hrs_1500/2024_1204_154045_002A.MP4"
    # Output video path
    output_path = "output_video.mp4"

    # Load all tracks (list of track points with IDs)
    all_tracks = load_tracks_from_folder(folder_path)

    # Plot tracks onto the video
    plot_tracks_with_lines(video_path, all_tracks, output_path)