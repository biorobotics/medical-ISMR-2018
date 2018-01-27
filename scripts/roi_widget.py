#!/usr/bin/env python
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
from PyQt5 import QtGui, QtCore, QtOpenGL, QtWidgets, uic
from sensor_msgs.msg import RegionOfInterest
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from clean_resource_path import cleanResourcePath
import message_filters
import rospy
import numpy as np

class ROIWidget(QtWidgets.QWidget):
    bridge = CvBridge()
    def __init__(self, stiffnessTopic, texturePath, masterWidget=None, parent=None):
        super(ROIWidget, self).__init__(parent)
        # Load UI
        uiPath = cleanResourcePath("package://oct_15_demo/scripts/roi_widget.ui")
        uic.loadUi(uiPath, self)
        # Load texture
        image = cv2.imread(cleanResourcePath(texturePath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.originalImage = image
        if image.shape[0] > 512 or image.shape[1] > 512:
            height, width = min(image.shape[0], 512), min(image.shape[1], 512)
            image = cv2.resize(image, (width,height))
        self.cvWidget = CvWidget(image, stiffnessTopic, parent=self)

        self.m = masterWidget
        if type(self.m) != type(None):
            self.cvWidget.mouseMoveEvent = self.m.cvWidget.mouseMoveEvent
            self.cvWidget.mousePressEvent = self.m.cvWidget.mousePressEvent
            self.cvWidget.mouseReleaseEvent = self.m.cvWidget.mouseReleaseEvent
            self.cvWidget.timerEvent = self.childTimerEvent

            self.getStiffness.clicked.connect(self.m.cvWidget.overlayStiffness)
            self.clearStiffness.clicked.connect(self.m.cvWidget.clearStiffness)
            self.stop.clicked.connect(self.m.cvWidget.stopOverlayStiffness)
        else:
            rospy.Subscriber(stiffnessTopic,
                             Image,
                             self.stiffness_map_callback,
                             queue_size=1)
            self.imgPub = rospy.Publisher('stiffness_texture', Image, queue_size=10)
            self.getStiffness.clicked.connect(self.cvWidget.overlayStiffness)
            self.clearStiffness.clicked.connect(self.cvWidget.clearStiffness)
            self.stop.clicked.connect(self.cvWidget.stopOverlayStiffness)

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addWidget(self.cvWidget)
        # mainLayout.addWidget(self.button)

        self.cvWindow.setLayout(mainLayout)
        self.resize(600, 600)

    def childTimerEvent(self, event):
        self.cvWidget.stiffActive = self.m.cvWidget.stiffActive
        self.cvWidget.center = self.m.cvWidget.center
        self.cvWidget.majorAxis = self.m.cvWidget.majorAxis
        self.cvWidget.minorAxis = self.m.cvWidget.minorAxis
        self.cvWidget.cvImage = self.m.cvWidget.cvImage
        self.cvWidget.update()

    def stiffness_map_callback(self, msg):
        image = self.originalImage.copy()
        width, height = image.shape[1], image.shape[0]
        centerX = int(self.cvWidget.center[0] * width)
        centerY = int((1-self.cvWidget.center[1]) * height)
        majorAxis = int(self.cvWidget.majorAxis * width * 2)
        minorAxis = int(self.cvWidget.minorAxis * height * 2)
        if self.cvWidget.stiffActive and majorAxis > 0 and minorAxis > 0:
            try:
                # Convert your ROS Image message to OpenCV2
                stiffnessImg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                stiffnessImg = cv2.cvtColor(stiffnessImg, cv2.COLOR_BGR2RGB)
                stiffnessShape = (majorAxis, minorAxis)
                stiffnessImg = cv2.resize(stiffnessImg,
                                          stiffnessShape,
                                          interpolation = cv2.INTER_LINEAR)
            except CvBridgeError, e:
                print(e)
                return

            xStart = centerX - majorAxis / 2
            xEnd = xStart + majorAxis
            yStart = centerY - minorAxis / 2
            yEnd = yStart + minorAxis

            mask = np.zeros(stiffnessImg.shape, dtype = np.uint8)
            center = tuple(np.array((mask.shape[1], mask.shape[0])) / 2)
            cv2.ellipse(mask, center, center, 0, 0, 360,
                        color=(1,1,1), thickness=-1)
            mask = mask[:,:,0] > 0

            cv2.subtract(image[yStart:yEnd, xStart:xEnd],
                         stiffnessImg,
                         dst = stiffnessImg)

            image[yStart:yEnd, xStart:xEnd][mask] = stiffnessImg[mask]

        self.cvWidget.cvImage = image

        msg = self.bridge.cv2_to_imgmsg(image, 'rgb8')
        self.imgPub.publish(msg)

class CvWidget(QtOpenGL.QGLWidget):

    def __init__(self, image, stiffnessTopic, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.cvImage = image
        self.aspectRatio = image.shape[1] / float(image.shape[0])
        self.size = (400, 400)
        # Ellipse variables
        self.center = (0,0)
        self.majorAxis = 0
        self.minorAxis = 0
        self.startTimer(67)
        self.roiPub = rospy.Publisher('stiffness_roi',
                                      RegionOfInterest,
                                      queue_size=1)
        self.getStiffnessPub = rospy.Publisher('get_stiffness',
                                               Bool,
                                               queue_size=1)
        self.overlayImgSize = 1000 # in pixels
        self.overlayImage = None
        self.stiffActive = False

    def clearStiffness(self):
        self.getStiffnessPub.publish(False)
        self.stiffActive = False
        self.overlayImage = None

    def overlayStiffness(self):
        if self.majorAxis > 0 and self.minorAxis > 0:
            self.getStiffnessPub.publish(True)
            self.stiffActive = True

    def stopOverlayStiffness(self):
        self.getStiffnessPub.publish(False)

    def resizeEvent(self,event):
        QtWidgets.QWidget.resizeEvent(self,event)
        self.size = (self.width(), self.height())

    def getMousePos(self, event):
        size = min(self.height(),self.width())
        offset = ((self.width() - size) / 2.0,
                  (self.height() - size) / 2.0)
        x = (event.pos().x() - offset[0]) / float(size)
        y = 1 - (event.pos().y() - offset[1]) / float(size)
        return (x,y)

    def clampMousePos(self, mousePos):
        # Clamps mouse position to positive values resulting in a valid ellipse
        x = abs(mousePos[0] - self.center[0])
        y = abs(mousePos[1] - self.center[1])
        xMax = min(self.center[0], 1 - self.center[0])
        yMax = min(self.center[1], 1 - self.center[1])
        x, y = min(x, xMax) + self.center[0], min(y, yMax) + self.center[1]
        return (x , y)

    def mouseMoveEvent(self, QMouseEvent):
        mouse_pos = self.getMousePos(QMouseEvent)
        mouse_pos = self.clampMousePos(mouse_pos)
        self.majorAxis = mouse_pos[0] - self.center[0]
        self.minorAxis = mouse_pos[1] - self.center[1]

    def mousePressEvent(self, QMouseEvent):
        self.stiffActive = False
        self.getStiffnessPub.publish(self.stiffActive)
        self.center = self.getMousePos(QMouseEvent)
        self.mouse_pos = self.getMousePos(QMouseEvent)
        self.majorAxis = 0
        self.minorAxis = 0

    def mouseReleaseEvent(self, QMouseEvent):
        if self.majorAxis > 0 and self.minorAxis > 0:
            self.publish_roi()
        
    def timerEvent(self, event):
        if self.isVisible():
            self.update()

    def publish_roi(self):
        msg = RegionOfInterest()
        msg.x_offset = (self.center[0] - self.majorAxis) * self.overlayImgSize
        msg.y_offset = (self.center[1] - self.minorAxis) * self.overlayImgSize
        msg.width = int(self.majorAxis * 2 * self.overlayImgSize)
        msg.height = int(self.minorAxis * 2 * self.overlayImgSize)
        self.roiPub.publish(msg)

    def paintGL(self):
        if not self.isVisible():
            return
        img = self.cvImage
        if type(img) == type(None):
            return
        # Resize image to fit screen
        newHeight = int(self.height()*self.aspectRatio)
        size = min(newHeight,self.width())
        offset = ((self.width() - size) / 2,
                  (self.height() - int(size / self.aspectRatio)) / 2)
        width, height = (size,int(size / self.aspectRatio))
        img = cv2.resize(img, (width, height))
        # Draw selcetion ellipse
        if(self.majorAxis > 0 and self.minorAxis > 0):
            centerX = int(self.center[0] * width)
            centerY = int((1-self.center[1]) * height)
            majorAxis = int(self.majorAxis * width * 2)
            minorAxis = int(self.minorAxis * height * 2)
            box = ((centerX, centerY),(majorAxis, minorAxis),0)
            cv2.ellipse(img, box, color=(0,255,0), thickness=size/100)
        # Need to flip image because GL buffer has 0 at bottom
        img = cv2.flip(img, flipCode = 0)
        fmt = GL_RGB
        t = GL_UNSIGNED_BYTE
        glViewport(offset[0],offset[1],width,height)
        glClearColor(0.0,1.0,1.0,1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glEnable(GL_ALPHA_TEST)
        glAlphaFunc(GL_GREATER,0)

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glMatrixMode(GL_PROJECTION);
        matrix = glGetDouble( GL_PROJECTION_MATRIX )
        glLoadIdentity();
        glOrtho(0.0, width, 0.0, height, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix()
        glLoadIdentity()
        glRasterPos2i(0,0)
        glDrawPixels(width, height, fmt, t, img)
        glPopMatrix()
        glFlush()

    def initializeGL(self):

        glClearColor(0,0,0, 1.0)

        glClearDepth(1.0)              
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()                    
        gluPerspective(45.0,1.33,0.1, 100.0) 
        glMatrixMode(GL_MODELVIEW)

if __name__ == '__main__':
    app = QtWidgets.QApplication(['Yo'])  
    rospy.init_node('stiffness_overlay', anonymous=False)
    # texturePath = rospy.get_param("~texture_path")
    texturePath = "package://oct_15_demo/resources/goofy-face2.png"
    w = ROIWidget('/stereo/left/image_rect', texturePath)
    # w = ROIWidget(None, 'stiffness_map', texturePath)
    w.show()
    app.exec_()