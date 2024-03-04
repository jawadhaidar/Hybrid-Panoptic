import cv2
import os

def extract_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    frame_paths = []
    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        # Append the path of the saved image to the list
        frame_paths.append(frame_path)

        frame_count += 1

    # Release the video capture object
    cap.release()

    return frame_paths

if __name__ == "__main__":
    # Path to the input video file
    video_path = "/home/aub/ficosadatasetvideos/2_video/video.mp4"

    # Output directory to save the frames
    output_dir = "frames"

    # Extract frames from the video
    frame_paths = extract_frames(video_path, output_dir)
    print("\n".join(frame_paths))
