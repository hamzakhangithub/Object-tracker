import cv2
import numpy as np
import argparse

# Parse 
parser = argparse.ArgumentParser(description='Object Tracking using Optical Flow')
parser.add_argument('video_path', help='Path to the input video file')
args = parser.parse_args()


cap = cv2.VideoCapture(args.video_path)


width = int(cap.get(3))
height = int(cap.get(4))

output = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width, height))


feature_params = dict(
    maxCorners=100,
    qualityLevel=0.5,
    minDistance=7,
    blockSize=7
)

lucas_kanade_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

color = np.random.randint(0, 255, (100, 3))

# Initial frame and corners
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

mask = np.zeros_like(prev_frame)
tracked_objects = []

while True:
    ret, frame = cap.read()

    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                                frame_gray,
                                                                prev_corners,
                                                                None,
                                                                **lucas_kanade_params)

        # Check if new_corners is not None
        if new_corners is not None:
            # Select and store good points
            good_new = new_corners[status == 1]
            good_old = prev_corners[status == 1]

            # compute velocity and direction
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = map(int, new.ravel())
                c, d = map(int, old.ravel())

                velocity = np.sqrt((a - c)**2 + (b - d)**2)
                direction = np.arctan2(b - d, a - c)

                tracked_objects.append({
                    'position': (a, b),
                    'velocity': velocity,
                    'direction': direction
                })

                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

                text = f"Vel: {velocity:.2f}, Dir: {np.degrees(direction):.2f}"
                frame = cv2.putText(frame, text, (int(a), int(b)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            img = cv2.add(frame, mask)

            # Save Video
            output.write(img)

            # Now update the previous frame and previous points
            prev_gray = frame_gray.copy()
            prev_corners = good_new.reshape(-1, 1, 2)
        else:
            # Reinitialize feature points if not found
            prev_corners = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)

    else:
        break


cap.release()
output.release()
cv2.destroyAllWindows()


