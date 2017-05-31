import extract
import cv2

# Get OpenCV capture of the video
cap = extract.get_capture('https://www.youtube.com/watch?v=ynMYE64IEKs', '1080p')

# Exit if cannot get capture
if cap is None:
    exit(1)

# Move capture to launch. If live this line does nothing.
extract.skip_to_launch(cap)

# Read the first frame
_, frame = cap.read()

# While the video hasn't finished
while frame is not None:
    # Calculate the time, velocity and alitutde values from the frame
    # If can't calculate values, returns (None, None, None)
    time, velocity, altitude = extract.extract_telemetry(frame)

    # If values are valid, print them
    if time is not None:
        print(time, velocity, altitude)
        
    # Read the next frame
    _, frame = cap.read()