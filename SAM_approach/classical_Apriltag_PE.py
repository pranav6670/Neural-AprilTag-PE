import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# Camera calibration parameters (example values)
fx, fy = 600, 600  # Focal lengths
cx, cy = 320, 240  # Principal point
camera_params = (fx, fy, cx, cy)

# Size of the AprilTag in meters
tag_size = 0.16

# Load the image
image = cv2.imread('tags6.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the AprilTag detector
detector = Detector(families='tag36h11')

# Detect tags in the image
results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

# Overlay pose vectors and output rotation/translation
for result in results:
    # Pose estimation
    t_vector = result.pose_t  # Translation vector
    r_matrix = result.pose_R  # Rotation matrix

    # Convert rotation matrix to Euler angles
    rotation = R.from_matrix(r_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)

    # Print the six pose parameters
    print("Rotation (Euler angles):", euler_angles)
    print("Translation (x, y, z):", t_vector.ravel())

    # Overlay pose on image
    # Define points for the axes in 3D space
    axis_length = tag_size / 2  # Length of the overlayed axis
    axis_points = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length]
    ]).reshape(-1, 3)

    # Project the 3D points to the 2D image plane
    r_vec, _ = cv2.Rodrigues(r_matrix)
    img_points, _ = cv2.projectPoints(axis_points, r_vec, t_vector, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), None)

    # Draw axes on the image
    img_points = img_points.astype(int)
    cv2.line(image, tuple(img_points[0].ravel()), tuple(img_points[1].ravel()), (0, 0, 255), 3)  # X axis - Red
    cv2.line(image, tuple(img_points[0].ravel()), tuple(img_points[2].ravel()), (0, 255, 0), 3)  # Y axis - Green
    cv2.line(image, tuple(img_points[0].ravel()), tuple(img_points[3].ravel()), (255, 0, 0), 3)  # Z axis - Blue

    # Draw the center of the tag
    (cX, cY) = (int(result.center[0]), int(result.center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

# Display the result
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
