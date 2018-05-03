#!/usr/bin/env python
import numpy as np
# Plotting stuff
from matplotlib import pyplot as plt
from matplotlib import cm
# image stuff
import PIL
import cv2
from cv_bridge import CvBridge, CvBridgeError
#ros stuff
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import oct_15_demo.srv
from IPython import embed

class RasterScan:
    def __init__(self, visualize=True):

        self.domain = {'L1':30, 'L2':30}
        self.points = self.generateGrid()
        self.safePoints = self.cropGrid(self.points)
        self.stiffness = np.zeros((self.points.shape[0],))
        self.visualize=True
        self.probedPoints = []
        self.searching = False

        self.min = np.inf

        print(len(self.stiffness[self.safePoints]))
        
        rospy.wait_for_service('/stereo/probe2D')
        self.probe2D = rospy.ServiceProxy('/stereo/probe2D', oct_15_demo.srv.Probe2D)

        self.pub = rospy.Publisher('/stereo/stiffness_map', Image, queue_size=10)
        self.sub = rospy.Subscriber('/stereo/get_stiffness', Bool, self.searchingCB)    

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if not self.searching:
                rate.sleep()
                continue
            for idx in self.safePoints:
                if not self.searching:
                    break
                point = oct_15_demo.srv.Probe2DRequest()
                x_probed = self.points[idx]/self.domain['L1']
                print x_probed
                point.Target.x, point.Target.y= (x_probed[0], x_probed[1])
                stiffness = self.probe2D(point).Stiffness
                if stiffness < 0:
                    print("Invalid stiffness ignored")
                    continue
                self.stiffness[idx] = stiffness
                if stiffness < self.min:
                    self.min = stiffness
                self.visualize_map(title="Raster Scan", figure=2)
                # self.probedPoints.append(point)
                rate.sleep()
            self.searching = False
        embed()

    def searchingCB(self, msg):
        print("HELLO")
        if msg.data and not self.searching:
            self.stiffness = np.zeros((self.points.shape[0],))
            self.probedPoints = []
        self.searching = msg.data;

    def generateGrid(self,res=1):
        x = np.linspace(0, self.domain['L1'], self.domain['L1'] // res)
        y = np.linspace(0, self.domain['L2'], self.domain['L2'] // res)
        Xg,Yg = np.meshgrid(x,y)
        grid = np.array([Xg.flatten(), Yg.flatten()]).T
        return grid

    def cropGrid(self, grid):
        newGrid = []
        for index, point in enumerate(grid):
            dx = point[0] - self.domain['L1']/2
            dy = point[1] - self.domain['L2']/2
            r_squared = dx*dx+dy*dy
            safety = 0.01 # some safety region away from boundary
            r_max = self.domain['L1']/2 * (1 - safety) ## maximum allowed radius of the region to palpate in
            if r_squared < r_max*r_max: 
                newGrid.append(index)
        return newGrid

    def visualize_map(self, title, figure):
        map = self.stiffness
        self.stiffness[self.stiffness == 0] = self.min
        map = map-self.min
        map = map / np.max(map)
        map= map.reshape(int(np.sqrt(map.size)),int(np.sqrt(map.size)))
        im = PIL.Image.fromarray(np.uint8(cm.hot(map)*255))
        cv_im = np.array(im)

        msg_frame = CvBridge().cv2_to_imgmsg(cv_im,'rgba8')
        self.pub.publish(msg_frame)

        if self.visualize:
            plt.figure(figure)
            plt.clf()
            fig=plt.imshow(map,origin='lower',cmap=cm.hot)
            plt.title(title)
            plt.colorbar()
            if len(self.probedPoints) != 0:
                plt.scatter(self.probedPoints[:,0],
                            self.probedPoints[:,1])
            # plt.xlim((0,int(np.sqrt(map.size))))
            # plt.ylim((0,int(np.sqrt(map.size))))
            # plt.tight_layout()
            # plt.show()
            plt.pause(0.01)



if __name__ == '__main__':
    rospy.init_node('raster_scanner', anonymous=True)
    RasterScan()