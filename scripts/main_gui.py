#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
import rospkg

from PyQt5 import QtWidgets, QtGui, uic, QtCore
import bingham_registration
from dvrk_vision.registration_gui import RegistrationWidget
from roi_widget import ROIWidget
from dvrk_vision.overlay_gui import OverlayWidget
import dvrk_vision.vtktools as vtktools
from dvrk_vision.vtk_stereo_viewer import StereoCameras

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, camera, masterWidget = None):

        super(MainWindow, self).__init__()

        functionPath = os.path.dirname(os.path.realpath(__file__))
        uic.loadUi(os.path.join(functionPath,"main_gui.ui"), self)
        
        meshPath = rospy.get_param("~mesh_path")
        try:
            secondaryMeshPath = rospy.get_param("~secondary_mesh_path")
        except:
            rospy.logwarn("No secondary_mesh_path found. Using mesh_path for overlay")
            secondaryMeshPath = meshPath

        stlScale = rospy.get_param("~mesh_scale")

        # Set up parents
        regParent = None if masterWidget == None else masterWidget.reg
        overlayParent = None if masterWidget == None else masterWidget.overlay
        roiParent = None if masterWidget == None else masterWidget.roi

        self.reg = RegistrationWidget(camera,
                                      meshPath,
                                      scale=stlScale,
                                      masterWidget = regParent,
                                      parent = self)
        self.regLayout = QtWidgets.QVBoxLayout()
        self.regLayout.addWidget(self.reg)
        self.tabRegistration.setLayout(self.regLayout)

        texturePath = rospy.get_param("~texture_path")
        self.roi = ROIWidget("stiffness_map",
                             texturePath,
                             masterWidget = roiParent,
                             parent = self)
        self.roiLayout = QtWidgets.QVBoxLayout()
        self.roiLayout.addWidget(self.roi)
        self.tabROI.setLayout(self.roiLayout)
        
        self.overlay = OverlayWidget(camera,
                                     texturePath,
                                     secondaryMeshPath,
                                     scale=stlScale,
                                     masterWidget = overlayParent,
                                     parent = self)
        self.overlayLayout = QtWidgets.QVBoxLayout()
        self.overlayLayout.addWidget(self.overlay)
        self.tabOverlay.setLayout(self.overlayLayout)

        self.otherWindows = []
        if masterWidget != None:
            masterWidget.otherWindows.append(self)
            self.otherWindows.append(masterWidget)

        self.tabWidget.currentChanged.connect(self.tabChanged)
        self.show()

    def tabChanged(self):
        idx = self.tabWidget.currentIndex()
        for window in self.otherWindows:
            window.tabWidget.setCurrentIndex(idx)

    def closeEvent(self, qCloseEvent):
        for window in self.otherWindows:
            window.close()
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    rosThread = vtktools.QRosThread()
    rosThread.start()
    frameRate = 15
    slop = 1.0 / frameRate
    cams = StereoCameras("left/image_rect",
                      "right/image_rect",
                      "left/camera_info",
                      "right/camera_info",
                      slop = slop)
    mainWin = MainWindow(cams.camL)
    secondWin = MainWindow(cams.camR, masterWidget = mainWin)
    sys.exit(app.exec_())
