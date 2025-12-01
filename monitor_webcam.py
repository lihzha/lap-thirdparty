import cv2
import os

def main():
    # Change this if your camera is not at index 0
    cam_index = 9

    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {cam_index}")
        return

    # Optional: set resolution (you can tweak/remove these)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    save_dir = "captures"
    os.makedirs(save_dir, exist_ok=True)

    print("Press 'y' to save current frame, 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame.")
            break

        cv2.imshow("Webcam (press 'y' to save, 'q' to quit)", frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit on 'q' or ESC
        if key == ord('q') or key == 27:
            break

        # Save on 'y'
        if key == ord('y'):
            # Prompt in the terminal for a filename (without extension)
            filename = input("Enter image name (no extension): ").strip()
            if not filename:
                print("Empty name, skipping save.")
                continue

            # Basic cleanup: replace spaces with underscores
            filename = filename.replace(" ", "_")

            save_path = os.path.join(save_dir, f"{filename}.png")
            success = cv2.imwrite(save_path, frame)
            if success:
                print(f"Saved: {save_path}")
            else:
                print("Error: Failed to save image.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
