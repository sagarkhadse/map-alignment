#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid

import numpy as np
import cv2 as cv

class map_align:

    def __init__(self):
        self.cvbr = CvBridge()
        rospy.init_node('map_align')
        rospy.Subscriber('/rtabmap/grid_map', OccupancyGrid, self.callback)
        self.pub = rospy.Publisher('map_aligned_2d', Image, queue_size=10)

    @classmethod
    def map_to_img(cls, data):
        w = data.info.width
        h = data.info.height
        r = data.info.resolution
        l = len(data.data)

        img = np.zeros((w, h, 1), np.uint8)
        for y in range(0, h):
            for x in range(0, w):
                val = data.data[w * y + x]
                img[x, y, 0] = 255 if val > 0 else 0
        img = cv.Mat(img)
        return img

    def callback(self, data):
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        img = map_align.map_to_img(data)
        if img is not None:
            self.pub.publish(self.cvbr.cv2_to_imgmsg(img))
        
    def run(self):
        rospy.spin()
  
if __name__ == '__main__':
    node = map_align()
    node.run()