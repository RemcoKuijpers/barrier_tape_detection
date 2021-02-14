import cv2
import numpy as np
from math import radians
from scipy.spatial.transform import Rotation as R

class Camera(object):
    def __init__(self, name):
        self.intrinsic_matrix = np.array([[642.990857314726, 0, 306.635520400467],[0, 643.486121738129, 244.561566828635],[0, 0, 1]])

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
        camera_matrix = np.dot(self.intrinsic_matrix, extrinsic_matrix)
        world_point = np.array([[world_point[0]], [world_point[1]], [world_point[2]], [1]])
        image_point_scaled = np.dot(camera_matrix, world_point)
        image_point = image_point_scaled/image_point_scaled[2]
        image_point = np.delete(image_point, 2)
        return image_point

    def ImageToGroundPlane(self, translation, rotation, pixel_point):
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
        rotation_matrix = R.from_euler('xyz', rotation_vector, degrees=False).as_matrix() # Rotation matrix of camera wrt world
        transposed_rotation_matrix = np.transpose(rotation_matrix)
        extrinsic_matrix = np.append(transposed_rotation_matrix, -np.dot(transposed_rotation_matrix, translation_vector), axis=1) # Rotation matrix of world wrt camera
        camera_matrix = np.dot(self.intrinsic_matrix, extrinsic_matrix)
        normalized_camera_matrix = camera_matrix/camera_matrix[2,3]
        homography_matrix = np.delete(normalized_camera_matrix, 2, 1)
        inversed_homography_matrix = np.linalg.inv(homography_matrix)
        world_point_scaled = np.dot(inversed_homography_matrix, pixel_vector)
        world_point = world_point_scaled/world_point_scaled[2]
        return world_point

if __name__ == "__main__":
    cam = Camera('C920')
    cm_30 = cam.worldToImage([0, 0, 0.295], [radians(-29-90), 0, 0], [0, 0.34, 0])
    cm_40 = cam.worldToImage([0, 0, 0.295], [radians(-29-90), 0, 0], [0, 0.44, 0])
    cm_50 = cam.worldToImage([0, 0, 0.295], [radians(-29-90), 0, 0], [0, 0.54, 0])

    image = cv2.imread(r'C:\Users\20203316\Documents\Projects\line_detection\images\WIN_20210214_19_24_49_Pro.jpg')
    image = cv2.circle(image, (int(cm_30[0]),int(cm_30[1])), radius=4, color=(0, 0, 255), thickness=-1)
    image = cv2.circle(image, (int(cm_40[0]),int(cm_40[1])), radius=4, color=(0, 0, 255), thickness=-1)
    image = cv2.circle(image, (int(cm_50[0]),int(cm_50[1])), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.imshow("image", image) 
    cv2.waitKey(0)  
    cv2.destroyAllWindows() 