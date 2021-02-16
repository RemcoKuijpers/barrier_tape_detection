#!/usr/bin/env python
import sys
import rospy
import rospkg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from math import radians

rospack = rospkg.RosPack()
project_path = rospack.get_path('barrier_tape_detection')
sys.path.insert(1, project_path+"/src")
from barrier_tape_detection import Camera, cv2

rospy.init_node("line_publisher")
line_pub = rospy.Publisher("line", Marker, queue_size=1)

cam = Camera()
cam.initializeVideoCapture()

while not rospy.is_shutdown():
    line = cam.detectLinePoints()
    if line is not None:
        p1 = cam.imageToGroundPlane([0, 0, 0.295], [radians(-29-90), 0, 0], line[0])
        p2 = cam.imageToGroundPlane([0, 0, 0.295], [radians(-29-90), 0, 0], line[1])

        key = cv2.waitKey(1)
        if key == 27:
            break

    line = Marker()
    line.header.frame_id = "world"
    line.type = line.ARROW
    line.action = line.ADD
    line.scale.x = 0.01
    line.scale.y = 0.01
    line.scale.z = 0.01
    line.pose.orientation.w = 1.0
    line.color.r = 1.0
    line.color.g = 1.0
    line.color.b = 0.0
    line.color.a = 1.0

    point1 = Point()
    point1.x = p1[0]
    point1.y = p1[1]

    point2 = Point()
    point2.x = p2[0]
    point2.y = p2[1]

    points = []
    points.append(point1)
    points.append(point2)

    line.points = points

    line_pub.publish(line)