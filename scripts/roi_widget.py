import numpy as np
import cv2
import rospy
import roslib
# ROS Image message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import vtk
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from clean_resource_path import cleanResourcePath

from sensor_msgs.msg import RegionOfInterest
from std_msgs.msg import Bool

class ROIWidget(QtWidgets.QWidget):
    def __init__(self, stiffnessTopic, texturePath, parentWindow=None):
        # Setup widget
        super(ROIWidget, self).__init__()
        uiPath = cleanResourcePath("package://oct_15_demo/scripts/roi_widget.ui")
        # Get CV image from path
        uic.loadUi(uiPath, self)
        cvImage = cv2.imread(cleanResourcePath(texturePath))
        self.cvImage = cvImage.copy()
        self.imgWithOverlay = cvImage.copy()
        #Convert cv image into a QT image for GUI
        self.cvWindow.setAlignment(QtCore.Qt.AlignCenter);
        # Set up 
        self.ellipseCorner = None
        self.stiffActive = False
        self.mouse_pos = QtCore.QPoint(0,0)
        self.center = QtCore.QPoint(0,0)
        self.majoraxis = 0
        self.minoraxis = 0
        self.painter = QtGui.QPainter()
        self.painter.setBrush(QtCore.Qt.NoBrush);
        # Set up your subscriber and defne its callback
        self.bridge = CvBridge()
        # rospy.Subscriber("/stereo/left/image_rect", Image, self.stiffness_map_callback, queue_size=1)
        rospy.Subscriber(stiffnessTopic, Image, self.stiffness_map_callback, queue_size=1)
        self.cvWindow.mousePressEvent = self.mousePressEvent
        self.cvWindow.mouseMoveEvent = self.mouseMoveEvent
        self.cvWindow.mouseReleaseEvent = self.mouseReleaseEvent
        self.getStiffness.clicked.connect(self.overlay_stiffness)
        self.clearStiffness.clicked.connect(self.clear_stiffness)
        self.stop.clicked.connect(self.stop_overlay_stiffness)
        self.roiPub = rospy.Publisher('/stiffness_roi', RegionOfInterest, queue_size=1)
        self.getStiffnessPub = rospy.Publisher('/get_stiffness', Bool, queue_size=1)
        self.stiffness_img_size = 1000 # in pixels

        self.imgPub = rospy.Publisher('/stiffness_overlay', Image, queue_size=10)

    def paintEvent(self, QPaintEvent):
        # Make a QT image out of the image + overlay
        overlayImage = cv2.cvtColor(self.imgWithOverlay, cv2.COLOR_BGR2RGB)
        height, width, byteValue = overlayImage.shape
        byteValue = byteValue * width
        mQImage = QtGui.QImage(overlayImage, width, height, byteValue,
                               QtGui.QImage.Format_RGB888)
        # Scale image to fit window size
        windowW = self.cvWindow.width()
        windowH = self.cvWindow.height()
        # Need to subtract one from height and width to take into account border
        img = mQImage.scaled(windowW - 1, windowH -1, QtCore.Qt.KeepAspectRatio)
        pixMapItem = QtGui.QPixmap.fromImage(img)
        self.painter.begin(pixMapItem)
        # Draw ellipse on image
        pen = QtGui.QPen(QtCore.Qt.green, 3)
        self.painter.setPen(pen);
        self.painter.scale(1, 1)
        rect = self.painter.viewport()
        self.painter.drawEllipse(self.center, self.majoraxis, self.minoraxis)
        self.painter.end()
        # Add pixel map to window
        self.cvWindow.setPixmap(pixMapItem)

        img = cv2.cvtColor(self.imgWithOverlay, cv2.COLOR_BGR2RGB)
        msg = self.bridge.cv2_to_imgmsg(img,'rgb8')
        self.imgPub.publish(msg)

    def clear_stiffness(self):
        self.stiffActive = False
        self.getStiffnessPub.publish(self.stiffActive)
        self.imgWithOverlay = self.cvImage.copy()

    def overlay_stiffness(self):
        if self.majoraxis > 0 and self.minoraxis > 0:
            self.stiffActive = True
            self.getStiffnessPub.publish(self.stiffActive)

    def stop_overlay_stiffness(self):
        self.stiffActive = False
        self.getStiffnessPub.publish(self.stiffActive)

    def calculate_uv_position(self,QPoint):
        # Returns a UV position in range [0,1] relative to a position on pixMap
        xRatio = float(1.0) / self.cvWindow.pixmap().width()
        uv = np.array((QPoint.x(), QPoint.y()), float) * xRatio
        return uv.tolist()

    def publish_roi(self):
        msg = RegionOfInterest()
        offset = self.calculate_uv_position(self.ellipseCorner)
        msg.x_offset = int(offset[0] * self.stiffness_img_size)
        msg.y_offset = int(offset[1] * self.stiffness_img_size)
        widthHeight = self.calculate_uv_position(QtCore.QPoint(self.majoraxis,
                                                               self.minoraxis))
        msg.width = int(abs(widthHeight[0]) * 2 * self.stiffness_img_size)
        msg.height = int(abs(widthHeight[1]) * 2 * self.stiffness_img_size)
        self.roiPub.publish(msg)

    def getMousePos(self, event):
        # Returns mouse position relative to the pixMap
        widthDiff = (self.cvWindow.width() - self.cvWindow.pixmap().width())/2
        heightDiff = (self.cvWindow.height() - self.cvWindow.pixmap().height())/2
        x = event.pos().x() - widthDiff
        y = event.pos().y() - heightDiff
        return QtCore.QPoint(x,y)

    def clampMousePos(self, mousePos):
        # Clamps mouse position to positive values resulting in a valid ellipse
        x = abs(mousePos.x() - self.center.x())
        y = abs(mousePos.y() - self.center.y())
        xMax = min(self.center.x(), self.cvWindow.pixmap().width() - self.center.x())
        yMax = min(self.center.y(), self.cvWindow.pixmap().height() - self.center.y())
        x, y = min(x, xMax), min(y, yMax)
        return QtCore.QPoint(x + self.center.x(), y + self.center.y())

    def mouseMoveEvent(self, QMouseEvent):
        self.mouse_pos = self.getMousePos(QMouseEvent)
        self.mouse_pos = self.clampMousePos(self.mouse_pos)
        self.majoraxis = (self.mouse_pos.x() - self.center.x())
        self.minoraxis = (self.mouse_pos.y()- self.center.y())
        self.ellipseCorner = self.center * 2 - self.mouse_pos

    def mousePressEvent(self, QMouseEvent):
        self.stiffActive = False
        self.getStiffnessPub.publish(self.stiffActive)
        self.center = self.getMousePos(QMouseEvent)
        self.mouse_pos = self.getMousePos(QMouseEvent)
        self.majoraxis = 0
        self.minoraxis = 0

    def mouseReleaseEvent(self, QMouseEvent):
        if self.majoraxis > 0 and self.minoraxis > 0:
            self.publish_roi()

    def stiffness_map_callback(self, img):
        if self.majoraxis == 0 or self.minoraxis == 0:
            return 
        try:
            # Convert your ROS Image message to OpenCV2
            stiffnessImg = self.bridge.imgmsg_to_cv2(img, "bgr8")
            stiffnessShape = (2 * abs(self.majoraxis), 2 * abs(self.minoraxis))
            stiffnessImg = cv2.resize(stiffnessImg,
                                      stiffnessShape,
                                      interpolation = cv2.INTER_LINEAR)
            h, w, byteV = stiffnessImg.shape
            byteV = byteV * w
        except CvBridgeError, e:
            print(e)

        if self.stiffActive:
            xRatio = float(self.cvImage.shape[0]) / self.cvWindow.pixmap().width()
            height = int(stiffnessImg.shape[0] * xRatio)
            width = int(stiffnessImg.shape[1] * xRatio)
            
            stiffnessImg=cv2.resize(stiffnessImg, (width,height))
            mask = np.zeros(stiffnessImg.shape, dtype = np.uint8)
            center = tuple(np.array((mask.shape[1], mask.shape[0])) / 2)
            cv2.ellipse(mask, center, center, 0, 0, 360,
                        color=(1,1,1), thickness=-1)
            mask = mask[:,:,0] > 0
            xStart = int(self.ellipseCorner.x() * xRatio)
            xEnd = xStart + stiffnessImg.shape[1]
            yStart = int(self.ellipseCorner.y()* xRatio)
            yEnd = yStart + stiffnessImg.shape[0]
            cv2.subtract(self.cvImage[yStart:yEnd, xStart:xEnd],
                         stiffnessImg, 
                         dst = stiffnessImg)
            self.imgWithOverlay[yStart:yEnd, xStart:xEnd][mask] = stiffnessImg[mask]
            img = cv2.cvtColor(self.imgWithOverlay, cv2.COLOR_BGR2RGB)
            msg = self.bridge.cv2_to_imgmsg(img,'rgb8')
            self.imgPub.publish(msg)

if __name__=="__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    #self.scn = QtGui.QGraphicsScene()
    rospy.init_node('stiffness_overlay', anonymous=False)
    texturePath = rospy.get_param("~texture_path")
    w = ROIWidget('/stiffness_map', texturePath)
    w.resize(600, 400)
    w.show()
    app.exec_()