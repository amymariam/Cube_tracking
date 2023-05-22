import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

# Input video
cap = cv2.VideoCapture('C://Users//Admin//Desktop//muku//newnewnew.mp4')

# Define the lower and upper purple color thresholds
lower_purple = np.array([120, 50, 50])
upper_purple = np.array([150, 255, 255])

# Set up initial values for cube position
cube_position = None
last_cube_position = None

# Set up initial values for trajectory points
trajectory_points = []

while True:
    # Read frame from video capture device
    ret, frame = cap.read()

    # Break the loop if the frame is not successfully read
    if not ret:
        break

    # Convert frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply color segmentation to detect purple cube
    mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)

    # Perform morphological operations to remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize cube contour
    cube_contour = None

    # Find the largest contour (the purple cube)
    if len(contours) > 0:
        cube_contour = max(contours, key=cv2.contourArea)

        # Calculate the bounding box of the cube contour
        #x and y represent the top-left corner coordinates of the bounding rectangle. 
        #w and h represent the width and height of the bounding rectangle.
        x, y, w, h = cv2.boundingRect(cube_contour)

        # Calculate the centroid of the cube
        cube_position = (int(x + w / 2), int(y + h / 2))

    # If we have a cube position, draw it on the frame and update the last cube position
    if cube_position:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        last_cube_position = cube_position

    # If we have a last cube position, add it to the trajectory points
    if last_cube_position:
        trajectory_points.append(last_cube_position)

    # Draw the trajectory points on the frame
    for point in trajectory_points:
        cv2.circle(frame, point, 1, (0, 0, 255), -1)

    # Show the frame
    frame = imutils.resize(frame,height=150, width=150)
    cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Convert trajectory points to numpy array
trajectory_points = np.array(trajectory_points)

# Extract x and y coordinates from trajectory points
x_coordinates = trajectory_points[:, 0]
y_coordinates = trajectory_points[:, 1]

# Calculate horizontal distance from the starting position
horizontal_distance = x_coordinates - x_coordinates[0]

# Calculate vertical distance from the starting position
vertical_distance = -(y_coordinates - y_coordinates[0])

# Plot the graph of vertical distance vs horizontal distance
# plt.figure(figsize=(6, 4))
plt.plot(horizontal_distance, vertical_distance)
plt.xlabel('Horizontal Distance')
plt.ylabel('Vertical Distance')
plt.title('Vertical Distance vs Horizontal Distance')

# Set the vertical axis limits
plt.ylim(0, 400)

# Set the aspect ratio to compress the plot horizontally
plt.gca().set_aspect(0.9)

# Display the plot
plt.show()

# Release video capture device and destroy all windows
cap.release()
cv2.destroyAllWindows()



# Calculate total vertical distance traveled by the cube
total_vertical_distance = abs(vertical_distance[-1])

# Calculate total time taken (10 seconds for the video)
total_time = 9/4

# Calculate the speed of the cube
speed = total_vertical_distance / total_time

print("Speed of the cube:", speed, "units per second\n")