#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from IPython import embed
from scipy.stats import multivariate_normal, norm
from scipy.interpolate import griddata
import PIL

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge, CvBridgeError
import oct_15_demo.srv
from IPython import embed
np.random.seed(1)

class aquisition_algorithm(object):
    """Abstract class for the different algorithms"""
    def __init__(self, estimated_map):
        self.estimated_map = estimated_map

    def aquisitionFuncitons(self):
        raise Exception("NotImplementedException")

class EI(aquisition_algorithm):
        """Expected improvement class"""
        def __init__(self, estimated_map, stiffnessCollected):
            super(EI, self).__init__(estimated_map)
            self.stiffnessCollected=stiffnessCollected

        def aquisitionFunciton(self):
            eps = 0.1
            ymu = self.estimated_map['mean']
            ys2 = self.estimated_map['variance']

            minStiffnessCollected = np.min(self.stiffnessCollected)

            ys=np.atleast_2d(np.sqrt(ys2)).T;
            yEI=np.max(self.stiffnessCollected) - minStiffnessCollected

            ymuMax=np.max(ymu)
            ymu=ymu/ymuMax
            yEI=yEI/ymuMax
            ind_zero = (ys<=0.01)
            aquisitionFunciton = (ymu-yEI-eps)*norm.cdf((ymu-yEI-eps)/ys) + ys*norm.pdf((ymu-yEI-eps)/ys);
            aquisitionFunciton[ind_zero]=0;
            aquisitionFunciton/=np.sum(aquisitionFunciton)
            return aquisitionFunciton

class UCB(aquisition_algorithm):
    """Upper confidence bound class"""
    def __init__(self, estimated_map,_):
        super(UCB, self).__init__(estimated_map)

    def aquisitionFunciton(self):
        beta=4.35
        ymu = self.estimated_map['mean']
        ys2 = self.estimated_map['variance']
        ymu=ymu/np.max(ymu)

        ys=np.atleast_2d(np.sqrt(ys2)).T;
        aquisitionFunciton = ymu + beta*ys;
        aquisitionFunciton/=np.sum(aquisitionFunciton)
        return aquisitionFunciton

class LSE(aquisition_algorithm):
    """docstring for LSE"""
    def __init__(self, estimated_map,_):
        super(LSE, self).__init__(estimated_map)
        self.C = None
        self.a = None #ambiguity
        self.h = .8
        self.beta=1.35

    def aquisitionFunciton(self):
        ymu = self.estimated_map['mean']
        ymuMax=np.max(ymu)
        if ymuMax != 0:
            ymu=ymu/ymuMax
       
        self.h=0.6
        ys2 = self.estimated_map['variance']
        ys=np.atleast_2d(np.sqrt(ys2)).T;

        Q_min = ymu - self.beta*ys;
        Q_max = ymu + self.beta*ys;
       
        if self.C is None:
            self.C = {'min':Q_min,'max':Q_max}
        else:
            self.C['min']=np.max(np.array([Q_min,self.C['min']]),axis = 0)
            self.C['max']=np.min(np.array([Q_max,self.C['max']]),axis = 0)

        temp1=self.C['max']-self.h*np.ones(self.C['max'].shape)
        temp2=self.h*np.ones(self.C['max'].shape)-self.C['min']
        self.a=np.min(np.array([temp1.T[0],temp2.T[0]]),axis=0)
        return self.a

class gpr_palpation():
    def __init__(self, algorithm_name, visualize=True,  simulation=True, wait_for_searching_signal = True):
        self.searching = not wait_for_searching_signal
        self.simulation = simulation
        self.domain = {'L1':100, 'L2':100}
        self.grid = self.generateGrid()
        self.groundTruth = self.generateStiffnessMap()
        self.gp = self.gp_init()
       
        self.estimated_map={'mean':None,'variance':None}
        self.probedPoints=[] # saves all the probed points so far
        self.stiffnessCollected=[] # saves all the probed stiffnesses
       
        self.algorithm_class = self.chooseAlgorithm(algorithm_name)

        # ROS
        self.pub = rospy.Publisher('/stereo/stiffness_map', Image, queue_size=10)       
        self.sub = rospy.Subscriber('/stereo/get_stiffness', Bool,self.searchingCB)

        self.rate = rospy.Rate(1000)
        self.visualize = visualize

        if not simulation:
            rospy.wait_for_service('/stereo/probe2D')
            self.probe2D = rospy.ServiceProxy('/stereo/probe2D', oct_15_demo.srv.Probe2D)

        # self.ind = np.random.randint(0,self.domain['L1'])
        self.ind = 7050

    def searchingCB(self, msg):
        if msg.data and not self.searching:
            # self.ind = np.random.randint(0,self.domain['L1'])
            self.ind = 7050
            self.estimated_map['mean'] = None
            self.estimated_map['variance'] = None
            self.probedPoints[:] = [] # saves all the probed points so far
            self.stiffnessCollected[:] = [] # saves all the probed stiffnesses
        self.searching = msg.data
       

    def chooseAlgorithm(self, algorithm_name):
        if algorithm_name == 'EI':
            return EI(self.estimated_map,self.stiffnessCollected)
        if algorithm_name == 'UCB':
            return UCB(self.estimated_map,self.stiffnessCollected)
        if algorithm_name == 'LSE':
            return LSE(self.estimated_map,None)

    def gp_init(self):
        kernel = C(1.0, (1e-3, 1e3))*RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, n_restarts_optimizer=9)
        return gp

    def generateGrid(self,res=1):
        x = np.linspace(0, self.domain['L1'], self.domain['L1']/res)
        y = np.linspace(0, self.domain['L2'], self.domain['L2']/res)
        Xg,Yg = np.meshgrid(x,y)
        grid = np.array([Xg.flatten(), Yg.flatten()]).T
        return grid

    def generateStiffnessMap(self):
        '''
        generates random ground truth(stiffness map) of an organ for simulation purposes
        '''
        m1=[20.0,20.0]
        s1=60.0*np.identity(2)
        m2=[30.0,60.0]
        s2=60.0*np.identity(2)
        m3=[60.0,60.0]
        s3=150.0*np.identity(2)
        m4=[70.0,20.0]
        s4=60*np.identity(2);

        grid = self.grid

        mvn1 = multivariate_normal(m1,s1)
        G1 = mvn1.pdf(grid)
        mvn2 = multivariate_normal(m2,s2)
        G2 = mvn2.pdf(grid)
        mvn3 = multivariate_normal(m3,s3)
        G3 = mvn3.pdf(grid)
        mvn4 = multivariate_normal(m4,s4)
        G4 = mvn4.pdf(grid)

        G=G1+G2+2*G3+G4
        # G=np.max(G,0); #crop below 0
        G[G<0.0]=0.0
        G=1000*G + 9000; #normalize
        G=G/np.max(G)
        # print np.max(G)
        # print np.min(G)
        return G

    def visualize_map(self, title, figure, map=None, probed_points=None):
        if map is None:
            dense_grid=self.generateGrid(res=1)

            ymu = self.gp.predict(self.grid, return_std=False)
            ymu[ymu<0]=0
            map = ymu

        map=map.reshape(self.domain['L1'],self.domain['L2'])
        normalized_map = map/np.max(map)

        im = PIL.Image.fromarray(np.uint8(cm.hot(normalized_map)*255))
        cv_im=np.array(im)
        # cv_im = cv2.flip(cv_im,0)

        msg_frame = CvBridge().cv2_to_imgmsg(cv_im,'rgba8')
        self.pub.publish(msg_frame)
       
        if self.visualize:
            plt.figure(figure)
            plt.clf()
            fig=plt.imshow(normalized_map,origin='lower',cmap=cm.hot)
            plt.title(title)
            plt.colorbar()
            if not (probed_points is None):
                plt.scatter(probed_points[:,0],probed_points[:,1])
            plt.xlim((0,100))
            plt.ylim((0,100))
            # plt.tight_layout()
            # plt.show()
            plt.pause(0.01)

        return msg_frame
       
    def evaluateStiffness(self, X_query):
        return griddata( self.grid, self.groundTruth, X_query)

    def probe(self, x_probed):
        self.probedPoints.append(x_probed.tolist())
        if self.simulation:        
        #evaluate the stiffness at that point
            yind = self.evaluateStiffness(X_query=x_probed)

        else:
            point = oct_15_demo.srv.Probe2DRequest()
            x_probed = x_probed/self.domain['L1']
            point.Target.x, point.Target.y= (x_probed[0],x_probed[1])
            yind = self.probe2D(point)
            print yind
            if yind.Stiffness < 0 or yind.Stiffness == None:
                rospy.logwarn("Got invalid stiffness. Ignoring value")
                self.probedPoints = self.probedPoints[:len(self.probedPoints)-1]
                return
            yind = np.array([yind.Stiffness])

        self.stiffnessCollected.append(yind.tolist())
        probedPoints_array = np.asarray(self.probedPoints)
        stiffnessCollected_array = np.asarray(self.stiffnessCollected)
        minStiffness = np.min(stiffnessCollected_array)
        stiffnessCollected_array = stiffnessCollected_array - minStiffness
        self.gp.fit(probedPoints_array, stiffnessCollected_array)
        self.estimated_map['mean'], self.estimated_map['variance'] = self.gp.predict(self.grid, return_std=True)
        self.estimated_map['mean'][self.estimated_map['mean']<0]=0
       
        # shows the animation of the stiffness estimation
        self.visualize_map(title='Estimated map', figure=2, map=self.estimated_map['mean'], probed_points=probedPoints_array)

    def nextBestPoint(self, alg):
        '''
        Takes in a funciton to estimate next best point accordingly. For now it supports 'EI' and 'UCB'
        '''
        self.aquisitionFunciton = alg.aquisitionFunciton()
        indices = sorted(range(len(self.aquisitionFunciton)),reverse=True, key=lambda x: self.aquisitionFunciton[x])
        found_safe_point = False
       
        # i=0
        # while not found_safe_point:
        #     ind = indices[i]
        #     x_probe = self.grid[ind,:]
        #     dx = x_probe[0] - self.domain['L1']/2
        #     dy = x_probe[1] - self.domain['L1']/2
        #     r_squared = dx*dx+dy*dy
        #     safety = 10 # some safety region away from boundary
        #     r_max = self.domain['L1']/2 - safety ## maximum allowed radius of the region to palpate in
        #     if r_squared < r_max*r_max:
        #         found_safe_point = True
        #     i=i+1
        ind = indices[0]
        return ind
   
    def autoPalpation(self, num_of_probes=-1):

        if num_of_probes == -1:
            print self.searching
            rate = rospy.Rate(100)
            i = 0
            while not rospy.is_shutdown():
                if not self.searching:
                    i = 0
                    rate.sleep()
                    continue
                i = i + 1
                print('Probing point: '+str(i))
                x_probe = self.grid[self.ind,:]
                self.probe(x_probe)
                self.ind = self.nextBestPoint(self.algorithm_class) #Here change which algorithm you want
        else:
            for i in range(num_of_probes):
                print('Probing point: '+str(i))
                x_probe = self.grid[ind,:]
                self.probe(x_probe)
                ind = self.nextBestPoint(self.algorithm_class) #Here change which algorithm you want

if __name__ == "__main__":
    rospy.init_node('gpr_python', anonymous=True)
    gpr = gpr_palpation(algorithm_name='UCB', visualize=False, simulation=False, wait_for_searching_signal = True) # 'LSE', 'EI', 'UCB'

    # visualize ground truth
    ##gpr.visualize_map(map=gpr.groundTruth,title='Ground Truth', figure=1)

    gpr.autoPalpation()
# plt.show()