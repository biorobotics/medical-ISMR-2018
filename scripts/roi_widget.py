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

class ROIWidget(QtWidgets.QWidget):
    def __init__(self, stiffnessTopic, texturePath, masterWidget=None, parent=None):
        super(ROIWidget, self).__init__(parent)
        image = cv2.imread(cleanResourcePath(texturePath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.cvWidget = CvWidget(image, parent=self)
        self.m = masterWidget
        if type(masterWidget) != type(None):
            self.cvWidget.mouseMoveEvent = self.m.cvWidget.mouseMoveEvent
            self.cvWidget.mousePressEvent = self.m.cvWidget.mousePressEvent
            self.cvWidget.mouseReleaseEvent = self.m.cvWidget.mouseReleaseEvent
            self.cvWidget.timerEvent = self.childTimerEvent

        # self.button = QtWidgets.QPushButton('Test', self)
        uiPath = cleanResourcePath("package://oct_15_demo/scripts/roi_widget.ui")
        uic.loadUi(uiPath, self)
        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addWidget(self.cvWidget)
        # mainLayout.addWidget(self.button)

        self.cvWindow.setLayout(mainLayout)
        self.resize(600, 600)

        self.getStiffness.clicked.connect(self.cvWidget.overlayStiffness)
        self.clearStiffness.clicked.connect(self.cvWidget.clearStiffness)
        self.stop.clicked.connect(self.cvWidget.stopOverlayStiffness)

    def childTimerEvent(self, event):
        self.cvWidget.center = self.m.cvWidget.center
        self.cvWidget.majorAxis = self.m.cvWidget.majorAxis
        self.cvWidget.minorAxis = self.m.cvWidget.minorAxis
        self.cvWidget.update()


class CvWidget(QtOpenGL.QGLWidget):

    def __init__(self, image, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.cvImage = image
        self.aspectRatio = self.cvImage.shape[1] / float(self.cvImage.shape[0])
        self.size = (400, 400)
        # Ellipse variables
        self.center = (0,0)
        self.majorAxis = 0
        self.minorAxis = 0
        self.startTimer(67)
        self.roiPub = rospy.Publisher('/stiffness_roi', RegionOfInterest, queue_size=1)
        self.getStiffnessPub = rospy.Publisher('/get_stiffness', Bool, queue_size=1)
        # rospy.Subscriber(stiffnessTopic, Image, self.stiffness_map_callback, queue_size=1)
        self.overlayImgSize = 1000 # in pixels
        self.overlayImage = None

    def clearStiffness(self):
        self.getStiffnessPub.publish(False)
        self.overlayImage = None

    def overlayStiffness(self):
        if self.majorAxis > 0 and self.minorAxis > 0:
            self.getStiffnessPub.publish(True)

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
            cv2.ellipse(img, box, color=(0,255,0), thickness=img.shape[1]/200)
        # Need to flip image because GL buffer has 0 at bottom
        img = cv2.flip(img, flipCode = 0)
        fmt = GL_RGB
        t = GL_UNSIGNED_BYTE
        glViewport(offset[0],offset[1],width,height)
        glClearColor(1.0,1.0,1.0,1.0)
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

#     def stiffness_map_callback(self, img):
#         if self.majorAxis == 0 or self.minorAxis == 0:
#             return 
#         try:
#             # Convert your ROS Image message to OpenCV2
#             stiffnessImg = self.bridge.imgmsg_to_cv2(img, "bgr8")
#             stiffnessShape = (2 * abs(self.majorAxis), 2 * abs(self.minorAxis))
#             stiffnessImg = cv2.resize(stiffnessImg,
#                                       stiffnessShape,
#                                       interpolation = cv2.INTER_LINEAR)
#             h, w, byteV = stiffnessImg.shape
#             byteV = byteV * w
#         except CvBridgeError, e:
#             print(e)

#         if self.stiffActive:
#             xRatio = float(self.cvImage.shape[0]) / self.cvWindow.pixmap().width()
#             height = int(stiffnessImg.shape[0] * xRatio)
#             width = int(stiffnessImg.shape[1] * xRatio)
            
#             stiffnessImg=cv2.resize(stiffnessImg, (width,height))
#             mask = np.zeros(stiffnessImg.shape, dtype = np.uint8)
#             center = tuple(np.array((mask.shape[1], mask.shape[0])) / 2)
#             cv2.ellipse(mask, center, center, 0, 0, 360,
#                         color=(1,1,1), thickness=-1)
#             mask = mask[:,:,0] > 0
#             xStart = int(self.ellipseCorner.x() * xRatio)
#             xEnd = xStart + stiffnessImg.shape[1]
#             yStart = int(self.ellipseCorner.y()* xRatio)
#             yEnd = yStart + stiffnessImg.shape[0]
#             cv2.subtract(self.cvImage[yStart:yEnd, xStart:xEnd],
#                          stiffnessImg, 
#                          dst = stiffnessImg)
#             self.imgWithOverlay[yStart:yEnd, xStart:xEnd][mask] = stiffnessImg[mask]
#             img = cv2.cvtColor(self.imgWithOverlay, cv2.COLOR_BGR2RGB)
#             msg = self.bridge.cv2_to_imgmsg(img,'rgb8')
#             self.imgPub.publish(msg)

if __name__ == '__main__':
    app = QtWidgets.QApplication(['Yo'])  
    rospy.init_node('stiffness_overlay', anonymous=False)
    texturePath = rospy.get_param("~texture_path")
    w = ROIWidget(None, '/stiffness_map', texturePath)
    w.show()
    app.exec_()

# import numpy as np
# import cv2
# import rospy
# import roslib
# # ROS Image message
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError

# import vtk
# from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from clean_resource_path import cleanResourcePath

# from sensor_msgs.msg import RegionOfInterest
# from std_msgs.msg import Bool

# class ROIWidget(QtWidgets.QWidget):
#     def __init__(self, stiffnessTopic, texturePath, masterWidget=None):
#         # Setup widget
#         super(ROIWidget, self).__init__()
#         uiPath = cleanResourcePath("package://oct_15_demo/scripts/roi_widget.ui")
#         # Get CV image from path
#         uic.loadUi(uiPath, self)
#         cvImage = cv2.imread(cleanResourcePath(texturePath))
#         self.cvImage = cvImage.copy()
#         self.imgWithOverlay = cvImage.copy()
#         #Convert cv image into a QT image for GUI
#         self.cvWindow.setAlignment(QtCore.Qt.AlignCenter);
#         # Set up 
#         self.ellipseCorner = None
#         self.stiffActive = False
#         self.mouse_pos = QtCore.QPoint(0,0)
#         self.center = QtCore.QPoint(0,0)
#         self.majorAxis = 0
#         self.minorAxis = 0
#         self.painter = QtGui.QPainter()
#         self.painter.setBrush(QtCore.Qt.NoBrush);
#         # Set up your subscriber and defne its callback
#         self.bridge = CvBridge()
#         # rospy.Subscriber("/stereo/left/image_rect", Image, self.stiffness_map_callback, queue_size=1)
#         rospy.Subscriber(stiffnessTopic, Image, self.stiffness_map_callback, queue_size=1)
#         self.cvWindow.mousePressEvent = self.mousePressEvent
#         self.cvWindow.mouseMoveEvent = self.mouseMoveEvent
#         self.cvWindow.mouseReleaseEvent = self.mouseReleaseEvent
#         self.getStiffness.clicked.connect(self.overlayStiffness)
#         self.clearStiffness.clicked.connect(self.clearStiffness)
#         self.stop.clicked.connect(self.stopOverlayStiffness)
#         self.roiPub = rospy.Publisher('/stiffness_roi', RegionOfInterest, queue_size=1)
#         self.getStiffnessPub = rospy.Publisher('/get_stiffness', Bool, queue_size=1)
#         self.stiffness_img_size = 1000 # in pixels

#         self.imgPub = rospy.Publisher('/stiffness_overlay', Image, queue_size=10)

#     def paintEvent(self, QPaintEvent):
#         # Make a QT image out of the image + overlay
#         overlayImage = cv2.cvtColor(self.imgWithOverlay, cv2.COLOR_BGR2RGB)
#         ratio = self.cvWindow.height() / float(overlayImage.shape[1])
#         width, height = (int(overlayImage.shape[0]*ratio) - 1,
#                          int(overlayImage.shape[1]*ratio) - 1)
#         overlayImage = cv2.resize(overlayImage, (width, height))
#         box = ((self.center.x(), self.center.y()),
#                (self.majorAxis * 2, self.minorAxis * 2),0)
#         cv2.ellipse(overlayImage, box, color=(0,255,0), thickness=overlayImage.shape[1]/200)
#         height, width, byteValue = overlayImage.shape
#         byteValue = byteValue * width
#         img = QtGui.QImage(overlayImage, width, height, byteValue,
#                            QtGui.QImage.Format_RGB888)
#         pixMapItem = QtGui.QPixmap.fromImage(img)
#         # Add pixel map to window
#         self.cvWindow.setPixmap(pixMapItem)

#         img = cv2.cvtColor(self.imgWithOverlay, cv2.COLOR_BGR2RGB)
#         msg = self.bridge.cv2_to_imgmsg(img,'rgb8')
#         self.imgPub.publish(msg)

#     def clearStiffness(self):
#         self.stiffActive = False
#         self.getStiffnessPub.publish(self.stiffActive)
#         self.imgWithOverlay = self.cvImage.copy()

#     def overlayStiffness(self):
#         if self.majorAxis > 0 and self.minorAxis > 0:
#             self.stiffActive = True
#             self.getStiffnessPub.publish(self.stiffActive)

#     def stopOverlayStiffness(self):
#         self.stiffActive = False
#         self.getStiffnessPub.publish(self.stiffActive)

#     def calculate_uv_position(self,QPoint):
#         # Returns a UV position in range [0,1] relative to a position on pixMap
#         xRatio = float(1.0) / self.cvWindow.pixmap().width()
#         uv = np.array((QPoint.x(), QPoint.y()), float) * xRatio
#         return uv.tolist()

#     def publish_roi(self):
#         msg = RegionOfInterest()
#         offset = self.calculate_uv_position(self.ellipseCorner)
#         msg.x_offset = int(offset[0] * self.stiffness_img_size)
#         msg.y_offset = int(offset[1] * self.stiffness_img_size)
#         widthHeight = self.calculate_uv_position(QtCore.QPoint(self.majorAxis,
#                                                                self.minorAxis))
#         msg.width = int(abs(widthHeight[0]) * 2 * self.stiffness_img_size)
#         msg.height = int(abs(widthHeight[1]) * 2 * self.stiffness_img_size)
#         self.roiPub.publish(msg)

#     def getMousePos(self, event):
#         # Returns mouse position relative to the pixMap
#         widthDiff = (self.cvWindow.width() - self.cvWindow.pixmap().width())/2
#         heightDiff = (self.cvWindow.height() - self.cvWindow.pixmap().height())/2
#         x = event.pos().x() - widthDiff
#         y = event.pos().y() - heightDiff
#         return QtCore.QPoint(x,y)

#     def clampMousePos(self, mousePos):
#         # Clamps mouse position to positive values resulting in a valid ellipse
#         x = abs(mousePos.x() - self.center.x())
#         y = abs(mousePos.y() - self.center.y())
#         xMax = min(self.center.x(), self.cvWindow.pixmap().width() - self.center.x())
#         yMax = min(self.center.y(), self.cvWindow.pixmap().height() - self.center.y())
#         x, y = min(x, xMax), min(y, yMax)
#         return QtCore.QPoint(x + self.center.x(), y + self.center.y())

#     def mouseMoveEvent(self, QMouseEvent):
#         self.mouse_pos = self.getMousePos(QMouseEvent)
#         self.mouse_pos = self.clampMousePos(self.mouse_pos)
#         self.majorAxis = (self.mouse_pos.x() - self.center.x())
#         self.minorAxis = (self.mouse_pos.y()- self.center.y())
#         self.ellipseCorner = self.center * 2 - self.mouse_pos

#     def mousePressEvent(self, QMouseEvent):
#         self.stiffActive = False
#         self.getStiffnessPub.publish(self.stiffActive)
#         self.center = self.getMousePos(QMouseEvent)
#         self.mouse_pos = self.getMousePos(QMouseEvent)
#         self.majorAxis = 0
#         self.minorAxis = 0

#     def mouseReleaseEvent(self, QMouseEvent):
#         if self.majorAxis > 0 and self.minorAxis > 0:
#             self.publish_roi()



# if __name__=="__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     #self.scn = QtGui.QGraphicsScene()
#     rospy.init_node('stiffness_overlay', anonymous=False)
#     texturePath = rospy.get_param("~texture_path")
#     w = ROIWidget('/stiffness_map', texturePath)
#     w.resize(600, 400)
#     w.show()
#     app.exec_()