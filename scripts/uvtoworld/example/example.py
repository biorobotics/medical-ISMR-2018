#!/usr/bin/env python
if __name__ == '__main__':
	import sys
	import os.path as path
	if __package__ is None:
		sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
	import vtk
	from vtk.util.colors import tomato, turquoise
	import sys
	import numpy as np
	from uvtoworld import UVToWorldConverter, makeTexturedObjData
	from uvtoworld import rendertools as rt
	cellIds = vtk.vtkIdList()

	def mouseMoveEvent(obj,event):
		mousePos = obj.GetEventPosition()
		size = renWin.GetSize()
		scale = imageCenter[0] * 2
		mousePos = (mousePos[0] / float(size[0]) , mousePos[1] / float(size[1]))
		bgDot.SetPosition(scale * mousePos[0], scale * mousePos[1], 0)
		targetPos, normal = uvConverter.toWorldSpace(mousePos)
		# d, i = tree.query(mousePos)
		# targetPos = tCoordsArray[i]
		normalPoint = np.add(targetPos,np.multiply(normal,0.1))
		normalDot.SetPosition(normalPoint)
		targetDot.SetPosition(targetPos)
		bgRen.Render()
		ren.Render()
		renWin.Render()
		return
	dirPath = path.dirname(path.abspath(__file__))
	objPath = path.join(dirPath, 'Jesus_Unity.obj')
	imgPath = path.join(dirPath, 'JesusDiffuse.png')
	# Create the graphics structure.
	renWin = vtk.vtkRenderWindow()
	ren, bgRen, imageCenter = rt.makeBG(renWin, imgPath)
	targetDot = rt.makeDotActor((0,2,0),tomato,.05)
	ren.AddActor(targetDot)
	normalDot = rt.makeDotActor((0,2,0),turquoise,.02)
	ren.AddActor(normalDot)
	bgDot = rt.makeDotActor(imageCenter,turquoise,25)
	bgRen.AddActor(bgDot)
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(renWin)
	iren.AddObserver('MouseMoveEvent', mouseMoveEvent, 1.0)
	# style = ActorFollowMouseStyle()
	iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

	drawnDots = []
	for i in range(10):
		dot = rt.makeDotActor((0,2,0),turquoise,.002)
		ren.AddActor(dot)
		drawnDots.append(dot)

	objActor = rt.makeObjActor(objPath, imgPath)
	ren.AddActor(objActor)

	start = [0.0, 1.5, 1.0]
	end = [0.0, 2.5, -1.0]
	startActor = rt.makeDotActor(start,turquoise,.02)

	polyData = objActor.GetMapper().GetInput()

	uvConverter = UVToWorldConverter(polyData)

	renWin.SetSize(720, 720)
	 
	# This allows the interactor to initalize itself. It has to be
	# called before an event loop.
	iren.Initialize()
	 
	# We'll zoom in a little by accessing the camera and invoking a "Zoom"
	# method on it.
	ren.ResetCamera()
	renWin.Render()

	# Start the event loop.
	iren.Start()
