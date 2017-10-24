#!/usr/bin/env python
import sys
import os
import numpy as np
import rospy
import rospkg

from PyQt5 import QtWidgets, QtGui, uic, QtCore
import dvrk_vision
import bingham_registration
from dvrk_vision.registration_gui import RegistrationWidget
from roi_widget import ROIWidget
from overlay_gui import OverlayWidget
import dvrk_vision.vtktools as vtktools

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parentWindow = None):

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
        regParent = None if parentWindow == None else parentWindow.reg
        overlayParent = None if parentWindow == None else parentWindow.overlay
        roiParent = None if parentWindow == None else parentWindow.roi

        self.reg = RegistrationWidget(meshPath,
                                      scale=stlScale,
                                      parentWindow = regParent)
        self.regLayout = QtWidgets.QVBoxLayout()
        self.regLayout.addWidget(self.reg)
        self.tabRegistration.setLayout(self.regLayout)

        self.overlay = OverlayWidget(secondaryMeshPath,
                                     scale=stlScale,
                                     parentWindow = overlayParent)
        self.overlayLayout = QtWidgets.QVBoxLayout()
        self.overlayLayout.addWidget(self.overlay)
        self.tabOverlay.setLayout(self.overlayLayout)

        if parentWindow == None:
            texturePath = rospy.get_param("~texture_path")
            self.roi = ROIWidget('/stiffness_map',
                                 texturePath,
                                 parentWindow = roiParent)
            self.roiLayout = QtWidgets.QVBoxLayout()
            self.roiLayout.addWidget(self.roi)
            self.tabROI.setLayout(self.roiLayout)
        

        self.otherWindows = []
        if parentWindow != None:
            parentWindow.otherWindows.append(self)
            self.otherWindows.append(parentWindow)

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
    rosThread = vtktools.RosQThread()
    mainWin = MainWindow()
    secondWin = MainWindow(mainWin)
    rosThread.start()
    sys.exit(app.exec_())