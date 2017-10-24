""" This module contains helper functions for rendering in VTK"""
import numpy as np
import vtk
from uv_to_world import makeTexturedObjData

__all__ = ["CleanTexturedPolyData", "makeDot", "makeBG", "makeObjData", "makeObjActor"]

def makeDotActor(center, color, radius):
	""" Creates a sphere of specified color and radius at position 'center'."""
	# create source
	source = vtk.vtkSphereSource()
	source.SetCenter((0,0,0))
	source.SetRadius(radius)
	# mapper
	mapper = vtk.vtkPolyDataMapper()
	if vtk.VTK_MAJOR_VERSION <= 5:
		mapper.SetInput(source.GetOutput())
	else:
		mapper.SetInputConnection(source.GetOutputPort())
	# actor
	actor = vtk.vtkActor()
	actor.SetPosition(center)
	actor.SetMapper(mapper)
	actor.GetProperty().SetColor(color)
	actor.GetProperty().LightingOff()
	# assign actor to the renderer
	return actor

def makeBG(renderWindow, pngFilePath):
	""" Reads in a PNG image and turns it into a background renderer
	
	Args:
		renderWindow (vtk.vtkRenderWindow): Render window to which our output
			renderers will be attached.
		pngFilePath (string): String representing the path to the image file
			we would like to put in the background

	Returns:
		ren (vtk.vtkRenderer): Foreground renderer
		bgRen (vtk.vtkRenderer): Background renderer
		imageCenter (tuple len 3 of type float): (x,y,z) coordinates
			representing the center of image object in world space
	"""
	pngReader = vtk.vtkPNGReader()
	pngReader.SetFileName(pngFilePath)
	pngReader.Update()
	imgData = pngReader.GetOutput()
	# Create an image actor to display the image
	imgActor = vtk.vtkImageActor()
	imgActor.SetOpacity(.5)
 
	if vtk.VTK_MAJOR_VERSION <= 5:
		imgActor.SetInput(imgData)
	else:
		imgActor.SetInputData(imgData)
 
	# Create a renderer to display the image in the background
	bgRen = vtk.vtkRenderer()
	ren = vtk.vtkRenderer()

	bgRen.SetBackground(1,.5,1)

	# Set up the render window and renderers such that there is
	# a background layer and a foreground layer
	bgRen.SetLayer(0)
	bgRen.InteractiveOff()
	ren.SetLayer(1)
	renderWindow.SetNumberOfLayers(2)
	renderWindow.AddRenderer(bgRen)
	renderWindow.AddRenderer(ren)
	bgRen.AddActor(imgActor)

	# Set up the background camera to fill the renderer with the image
	origin = imgData.GetOrigin()
	spacing = imgData.GetSpacing()
	extent = imgData.GetExtent()
 
	camera = bgRen.GetActiveCamera()
	camera.ParallelProjectionOn()
 	
 	# Calculate image placement based on camera parameters
	xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
	yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
	yd = (extent[3] - extent[2] + 1) * spacing[1]
	d = camera.GetDistance()

	camera.SetParallelScale(0.5 * yd)
	camera.SetFocalPoint(xc, yc, 0.0)
	camera.SetPosition(xc, yc, d)
	imageCenter = (xc, yc, 0)

	return ren, bgRen, imageCenter

def makeObjActor(objPath, pngPath):
	"""	Create an actor from a specified .obj file and .png texture
	
	Notes:
		Make sure that the .obj loaded in includeds texture coordinates

	Args:
		objPath (string): File path to .obj file to load as object geometry
		pngPath (string): File path to .png file to load as object texture

	Returns:
		actor (vtk.vtkActor): VTK actor object with geometry and texture applied
	"""
	# Load geometry
	polyData = makeTexturedObjData(objPath)
	# Mapper
	mapper = vtk.vtkPolyDataMapper()
	if vtk.VTK_MAJOR_VERSION <= 5:
		mapper.SetInput(polyData)
	else:
		mapper.SetInputData(polyData)

	# Load texture
	pngReader = vtk.vtkPNGReader()
	pngReader.SetFileName(pngPath)
	texture = vtk.vtkTexture()
	texture.EdgeClampOff()
	if vtk.VTK_MAJOR_VERSION <= 5:
		texture.SetInput(pngReader.GetOutput())
	else:
		texture.SetInputConnection(pngReader.GetOutputPort())

	# Create actor
	actor = vtk.vtkActor()
	actor.SetMapper(mapper)
	actor.SetTexture(texture)
	actor.GetProperty().LightingOff()
	return actor

