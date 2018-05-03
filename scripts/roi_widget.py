import cv2
from dvrk_vision.cv_widget import CvWidget
from PyQt5 import QtGui, QtCore, QtWidgets, uic
from sensor_msgs.msg import RegionOfInterest
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from dvrk_vision.clean_resource_path import cleanResourcePath
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
        self.cvWidget = OverlayInteractorWidget(image, stiffnessTopic, parent=self)

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
                # Flip because GL buffer has zero on bottom
                stiffnessImg = cv2.flip(stiffnessImg, flipCode = 0)
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


            # image[yStart:yEnd, xStart:xEnd][mask] = stiffnessImg[mask]
            image[yStart:yEnd, xStart:xEnd] = stiffnessImg

        self.cvWidget.cvImage = image

        msg = self.bridge.cv2_to_imgmsg(image, 'rgb8')
        self.imgPub.publish(msg)

class OverlayInteractorWidget(CvWidget):

    def __init__(self, image, stiffnessTopic, fps=15, parent=None):
        CvWidget.__init__(self, fps, parent);
        self.roiPub = rospy.Publisher('stiffness_roi',
                                      RegionOfInterest,
                                      queue_size=1)
        self.getStiffnessPub = rospy.Publisher('get_stiffness',
                                               Bool,
                                               queue_size=1)
        # Ellipse variables
        self.center = (0,0)
        self.majorAxis = 0
        self.minorAxis = 0
        self.overlayImgSize = 1000 # in pixels
        self.overlayImage = None
        self.setImage(image)
        self.stiffActive = False

    def stopOverlayStiffness(self):
        self.getStiffnessPub.publish(False)

    def clearStiffness(self):
        self.getStiffnessPub.publish(False)
        self.stiffActive = False
        self.overlayImage = None

    def overlayStiffness(self):
        if self.majorAxis > 0 and self.minorAxis > 0:
            self.getStiffnessPub.publish(True)
            self.stiffActive = True

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

    def publish_roi(self):
        msg = RegionOfInterest()
        msg.x_offset = (self.center[0] - self.majorAxis) * self.overlayImgSize
        msg.y_offset = (self.center[1] - self.minorAxis) * self.overlayImgSize
        msg.width = int(self.majorAxis * 2 * self.overlayImgSize)
        msg.height = int(self.minorAxis * 2 * self.overlayImgSize)
        self.roiPub.publish(msg)

    def imageProc(self, img):
        # Draw selcetion ellipse
        if(self.majorAxis > 0 and self.minorAxis > 0):
            width = img.shape[1]
            height = img.shape[0]
            thickness = (width + height) / 200
            centerX = int(self.center[0] * width)
            centerY = int((1-self.center[1]) * height)
            majorAxis = int(self.majorAxis * width * 2)
            minorAxis = int(self.minorAxis * height * 2)
            box = ((centerX, centerY),(majorAxis, minorAxis),0)
            cv2.ellipse(img, box, color=(0,255,0), thickness=thickness)

        return img

if __name__ == '__main__':
    app = QtWidgets.QApplication(['roi_widget'])  
    rospy.init_node('stiffness_overlay', anonymous=False)
    # texturePath = rospy.get_param("~texture_path")
    texturePath = "package://oct_15_demo/resources/Diffuse.png"
    w = ROIWidget('/stereo/left/image_rect', texturePath)
    # w = ROIWidget(None, 'stiffness_map', texturePath)
    w.show()
    app.exec_()