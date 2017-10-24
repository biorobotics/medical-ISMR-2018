#!/usr/bin/env python
import sys
import os
import vtk
import numpy as np
import rospy
import rospkg
import cv2
# Which PyQt we use depends on our vtk version. QT4 causes segfaults with vtk > 6
if(int(vtk.vtkVersion.GetVTKVersion()[0]) >= 6):
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
    from PyQt5 import uic
    from PyQt5.QtCore import QThread
    _QT_VERSION = 5
    from dvrk_vision.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
else:
    from PyQt4.QtGui import QWidget, QVBoxLayout, QApplication
    from PyQt4 import uic
    from PyQt4.QtCore import QThread
    _QT_VERSION = 4
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import dvrk_vision.vtktools as vtktools
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from tf import transformations

from IPython import embed

class vtkRosTextureActor(vtk.vtkActor):
    ''' Attaches texture to the actor. Texture is received by subscribing to a ROS topic and then converted to vtk image
        Input: vtk.Actor
        Output: Updates the input actor with the texture
    '''

    def __init__(self,topic, color = (1,0,0)):
        print(topic)
        self.bridge = CvBridge()
        self.vtkImage = None

        #Subscriber
        sub = rospy.Subscriber(topic, Image, self.imageCB, queue_size=1)
        self.texture = vtk.vtkTexture()
        self.texture.EdgeClampOff()
        self.color = color
        self.textureOnOff(False)

    #Subscriber callback function
    def imageCB(self, img):
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print(e)
        else:
            if type(self.vtkImage) == type(None):
                self.vtkImage = vtktools.makeVtkImage(cv2_img.shape)
                self.textureOnOff(True)
            # Convert numpy to vtk image
            vtktools.numpyToVtkImage(cv2_img, self.vtkImage)

        if vtk.VTK_MAJOR_VERSION <= 5:
            self.texture.SetInput(self.vtkImage)
        else:
            self.texture.SetInputData(self.vtkImage)

    def textureOnOff(self, data):
        if data:
            self.SetTexture(self.texture)
            self.GetProperty().SetColor(1, 1, 1)
            self.GetProperty().LightingOff()
        else:
            self.SetTexture(None)
            self.GetProperty().SetColor(self.color)
            self.GetProperty().LightingOn()



def cleanResourcePath(path):
    newPath = path
    if path.find("package://") == 0:
        newPath = newPath[len("package://"):]
        pos = newPath.find("/")
        if pos == -1:
            rospy.logfatal("%s Could not parse package:// format", path)
            quit(1)

        package = newPath[0:pos]
        newPath = newPath[pos:]
        package_path = rospkg.RosPack().get_path(package)

        if package_path == "":
            rospy.logfatal("%s Package [%s] does not exist",
                           path.c_str(),
                           package.c_str());
            quit(1)

        newPath = package_path + newPath;
    elif path.find("file://") == 0:
        newPath = newPath[len("file://"):]

    if not os.path.isfile(newPath):
        rospy.logfatal("%s file does not exist", newPath)
        quit(1)
    return newPath;

class OverlayWidget(QWidget):

    def __init__(self, meshPath, scale=1, namespace="/stereo", parentWindow=None):

        super(OverlayWidget, self).__init__()
        uiPath = cleanResourcePath("package://oct_15_demo/scripts/overlay_widget.ui")
        # Get CV image from path
        uic.loadUi(uiPath, self)

        # Check whether this is the left (primary) or the right (secondary) window
        self.isPrimaryWindow = parentWindow == None
        side = "left" if self.isPrimaryWindow else "right"

        self.otherWindows = []
        if not self.isPrimaryWindow:
            parentWindow.otherWindows.append(self)
            self.otherWindows.append(parentWindow) 

        # Set up subscriber for camera image
        self.bridge = CvBridge()
        imgSubTopic = namespace + "/"+side+"/image_rect"
        self.imageSub = rospy.Subscriber(imgSubTopic, Image, self.imageCallback)

        
        # Set up vtk background image
        msg = rospy.wait_for_message(imgSubTopic, Image)
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Add vtk widget
        self.vl = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.vtkFrame)
        self.vl.addWidget(self.vtkWidget)
        self.vtkFrame.setLayout(self.vl)

        # Set up vtk camera using camera info
        self.bgImage = vtktools.makeVtkImage(image.shape[0:2])
        self.renWin = self.vtkWidget.GetRenderWindow()
        camInfo = rospy.wait_for_message(namespace + "/" + side + "/camera_info",
                                         CameraInfo, timeout=2)
        intrinsicMatrix, extrinsicMatrix = vtktools.matrixFromCamInfo(camInfo)
        self.ren, self.bgRen = vtktools.setupRenWinForRegistration(self.renWin,
                                                                   self.bgImage,
                                                                   intrinsicMatrix)
        pos = extrinsicMatrix[0:3,3]
        self.ren.GetActiveCamera().SetPosition(pos)
        pos[2] = 1
        self.ren.GetActiveCamera().SetFocalPoint(pos)
        self.zBuff = vtktools.zBuff(self.renWin)

        # Set up 3D actor for organ
        meshPath = cleanResourcePath(meshPath)
        extension = os.path.splitext(meshPath)[1]
        if extension == ".stl" or extension == ".STL":
            meshReader = vtk.vtkSTLReader()
        elif extension == ".obj" or extension == ".OBJ":
            meshReader = vtk.vtkOBJReader()
        else:
            ROS_FATAL("Mesh file has invalid extension (" + extension + ")")
        meshReader.SetFileName(meshPath)
        # Scale STL
        transform = vtk.vtkTransform()
        transform.Scale(scale,scale,scale)
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(meshReader.GetOutputPort())
        transformFilter.Update()
        color = (0,0,1)
        self.actor_moving = vtkRosTextureActor("/stiffness_overlay", color = color)
        self.actor_moving.GetProperty().BackfaceCullingOn()
        self._updateActorPolydata(self.actor_moving,
                                  polydata=transformFilter.GetOutput(),
                                  color = color)
        # Hide actor
        self.actor_moving.VisibilityOff()
        self.ren.AddActor(self.actor_moving)

        # Set up subscriber for registered organ position
        poseSubTopic = namespace + "/registration_marker"
        self.poseSub = rospy.Subscriber(poseSubTopic, Marker, self.poseCallback)

        # Setup interactor
        self.iren = self.renWin.GetInteractor()
        self.iren.RemoveObservers('LeftButtonPressEvent')
        self.iren.RemoveObservers('LeftButtonReleaseEvent')
        self.iren.RemoveObservers('MouseMoveEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')
        self.iren.RemoveObservers('MiddleButtonPressEvent')

        # Set up QT slider for opacity
        self.opacitySlider.valueChanged.connect(self.sliderChanged) 
        self.textureCheckBox.stateChanged.connect(self.checkBoxChanged)
        
        # Set up timer to refresh render
        self.show()

    def sliderChanged(self):
        self.actor_moving.GetProperty().SetOpacity(self.opacitySlider.value() / 255.0)
        for window in self.otherWindows:
            window.opacitySlider.setValue(self.opacitySlider.value())

    def checkBoxChanged(self):
        self.actor_moving.textureOnOff(self.textureCheckBox.isChecked())
        for window in self.otherWindows:
            window.textureCheckBox.setChecked(self.textureCheckBox.isChecked())

    def imageCallback(self, data):
        # TODO no try catch
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            vtktools.numpyToVtkImage(image,self.bgImage)
            self.renWin.Render()
        except:
            rospy.logwarn("Overlay GUI: Error in image callback")
            pass

    def poseCallback(self, data):
        pos = data.pose.position
        rot = data.pose.orientation
        mat = transformations.quaternion_matrix([rot.x,rot.y,rot.z,rot.w])
        mat[0:3,3] = [pos.x,pos.y,pos.z]
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.SetMatrix(mat.ravel())
        self.actor_moving.SetUserTransform(transform)
        self.actor_moving.VisibilityOn()             

    def _updateActorPolydata(self,actor,polydata,color=None):
        # Modifies an actor with new polydata
        bounds = polydata.GetBounds()

        # Visualization
        mapper = actor.GetMapper()
        if mapper == None:
            mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(polydata)
        else:
            mapper.SetInputData(polydata)
        actor.SetMapper(mapper)
        if type(color) !=  type(None):
            actor.GetProperty().SetColor(color[0], color[1], color[2])
        else:
            actor.GetProperty().SetColor(1, 0, 0)
        self.sliderChanged()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # RosThread.update = self.update
    rosThread = vtktools.RosQThread()
    meshPath = rospy.get_param("~mesh_path")
    stlScale = rospy.get_param("~mesh_scale")
    windowL = RegistrationWindow(meshPath, scale=stlScale)
    windowR = RegistrationWindow(meshPath, scale=stlScale, parentWindow=windowL)
    rosThread.start()
    sys.exit(app.exec_())