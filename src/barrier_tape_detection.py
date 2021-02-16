import cv2
import numpy as np
from math import radians, sqrt
from scipy.spatial.transform import Rotation as R

class Camera(object):
    def __init__(self):
        self.resolution = (640, 480)
        self.intrinsic_matrix = np.array([[643.336890787299, 0, 303.201848937713],[0, 643.532951456850, 240.391572527453],[0, 0, 1]])
        self.distortion_vector = np.array([0.0791643096656636, -0.518750253004116, -0.00264733385610876, -0.00200184149743238, 1.17271667584054])
        self.new_intrinsic_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.intrinsic_matrix, self.distortion_vector, self.resolution, 1, self.resolution)

    def worldToImage(self, translation, rotation, world_point):
        """Converts world coordinates to pixel coordinates

        Args:
            translation (list): [xc, yc, zc]: Camera position wrt world in meters
            rotation (list): [rx, ry, rz]: Camera rotation wrt world as Euler angles in radians
            world_point (list): [x, y, z]: World point in meters to convert to pixel coordinates

        Returns:
            [list]: [x, y]: Pixel location (origin at top left corner)
        """        ""
        translation_vector = np.array([[translation[0]],  [translation[1]],  [translation[2]]]) # Center of camera wrt world [xc, yc, zc]
        rotation_vector = np.array(rotation) # Rotation of camera wrt world [roll, pitch, yaw]
        rotation_matrix = R.from_euler('xyz', rotation_vector, degrees=False).as_matrix() # Rotation matrix of camera wrt world
        transposed_rotation_matrix = np.transpose(rotation_matrix)
        extrinsic_matrix = np.append(transposed_rotation_matrix, -np.dot(transposed_rotation_matrix, translation_vector), axis=1) # Rotation matrix of world wrt camera
        camera_matrix = np.dot(self.new_intrinsic_matrix, extrinsic_matrix)
        world_point = np.array([[world_point[0]], [world_point[1]], [world_point[2]], [1]])
        image_point_scaled = np.dot(camera_matrix, world_point)
        image_point = image_point_scaled/image_point_scaled[2]
        image_point = np.delete(image_point, 2)
        return image_point

    def imageToGroundPlane(self, translation, rotation, pixel_point):
        """Converts pixel location to location on ground plane

        Args:
            translation (list): [xc, yc, zc]: Camera position wrt world in meters
            rotation (list): [rx, ry, rz]: Camera rotation wrt world as Euler angles in radians
            pixel_point (list): [x, y]: Pixel location in pixel to convert to x, y on ground plane

        Returns:
            [list]: x,y positon on ground plane wrt origin
        """        ""
        pixel_vector = np.array([[pixel_point[0]], [pixel_point[1]], [1]])
        translation_vector = np.array([[translation[0]],  [translation[1]],  [translation[2]]]) # Center of camera wrt world [xc, yc, zc]
        rotation_vector = np.array(rotation) # Rotation of camera wrt world [roll, pitch, yaw]
        rotation_matrix = R.from_euler('xyz', rotation_vector, degrees=False).as_dcm() # Rotation matrix of camera wrt world
        transposed_rotation_matrix = np.transpose(rotation_matrix)
        extrinsic_matrix = np.append(transposed_rotation_matrix, -np.dot(transposed_rotation_matrix, translation_vector), axis=1) # Rotation matrix of world wrt camera
        camera_matrix = np.dot(self.new_intrinsic_matrix, extrinsic_matrix)
        normalized_camera_matrix = camera_matrix/camera_matrix[2,3]
        homography_matrix = np.delete(normalized_camera_matrix, 2, 1)
        inversed_homography_matrix = np.linalg.inv(homography_matrix)
        world_point_scaled = np.dot(inversed_homography_matrix, pixel_vector)
        world_point = world_point_scaled/world_point_scaled[2]
        return world_point

    def initializeVideoCapture(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.resolution[0])
        self.cap.set(4, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        #cv2.namedWindow("Trackbars")
        #cv2.createTrackbar("L-H", "Trackbars", 18, 180, nothing)
        #cv2.createTrackbar("L-S", "Trackbars", 100, 255, nothing)
        #cv2.createTrackbar("L-V", "Trackbars", 100, 255, nothing)
        #cv2.createTrackbar("U-H", "Trackbars", 30, 180, nothing)
        #cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
        #cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

    def detectLinePoints(self):
        global frame
        _, frame = self.cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = 18 #cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = 100 #cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = 100 #cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = 30 #cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = 255 #cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = 255 #cv2.getTrackbarPos("U-V", "Trackbars")

        lower_yellow = np.array([l_h, l_s, l_v])
        upper_yellow = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 0
        center_points = []
        distances = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)

            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                distances.append(sqrt(cx**2+cy**2))
                center_points.append([cx, cy])
                cv2.circle(frame, (cx,cy), radius=4, color=(0, 0, 255), thickness=-1)
            except (ZeroDivisionError, IndexError) as e:
                pass

            i += 1
            if area > 400:
                cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                
        if not distances:
            line_detected = False
        else:
            line_detected = True
            biggest_index = distances.index(max(distances)) # First start/end point of line
            smallest_index = distances.index(min(distances)) # Second start/end point of line
            start_point = center_points[smallest_index]
            end_point = center_points[biggest_index]
            cv2.line(frame,(start_point[0],start_point[1]),(end_point[0],end_point[1]),(255,0,0),5)
            
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        if line_detected:
            return np.array([start_point, end_point])
        else:
            return None

def nothing(x):
    pass

if __name__ == "__main__":
    cam = Camera()
    cam.initializeVideoCapture()
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        line = cam.detectLinePoints()
        if line is not None:
            p1 = cam.imageToGroundPlane([0, 0, 0.295], [radians(-29-90), 0, 0], line[0])
            p2 = cam.imageToGroundPlane([0, 0, 0.295], [radians(-29-90), 0, 0], line[1])
            print(p1)
            print(p2)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()