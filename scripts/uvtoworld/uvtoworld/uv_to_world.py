""" UV to world conversion tools

This module contains tools to convert 2D UV coordinates to 3D world
coordinates using VTK

	TODO:
		* Make ROS node
"""
import vtk
import numpy as np
from vtk.util import numpy_support
import time

from IPython import embed

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

__all__ = ["makeTexturedObjData","CleanTexturedPolyData","pointToBarycentric", "UVToWorldConverter"]

def _mkVtkIdList(it):
	""" This function creates a VTK idList from a integer typed iterable"""
	vil = vtk.vtkIdList()
	for i in it:
		vil.InsertNextId(int(i))
	return vil

def _cellsToNumpy(cellArray):
	""" This function creates a numpy array from a VTK cellArray object"""
	cellArray.InitTraversal()
	ids = vtk.vtkIdList()
	polys = np.zeros((cellArray.GetNumberOfCells(),3),np.uint32)
	counter = 0
	while cellArray.GetNextCell(ids):
		for i in range(ids.GetNumberOfIds()):
			polys[counter,i] = ids.GetId(i)
		counter += 1
	return polys

class CleanTexturedPolyData(vtk.vtkProgrammableFilter):
	""" This class is a filter for cleaning VTK polyData based on texture data.
		
		Note: Unlike vtkCleanPolyData it merges vertices based on the UV 
			coordinates, not the 3D positions of points. Because of this, it
			will produce artifacts when the UV map overlaps itself.

		TODO:
			* Incorporate 3d point data to fix UV merging artifacts
	"""
	def __init__(self):
		self.triAlg = vtk.vtkTriangleFilter() # Triangulates mesh
		self.SetExecuteMethod(self._cleanData) # Filters mesh based on tCoords
		self.cleanAlg = vtk.vtkCleanPolyData() # Removes extra cell data
		self.cleanAlg.PointMergingOff() # Turn off since we already merge points
		self.polyData = vtk.vtkPolyData()
		if vtk.VTK_MAJOR_VERSION <= 5:
			self.cleanAlg.SetInput(self.polyData)
		else:
			self.cleanAlg.SetInputData(self.polyData)


		self.normalGenerator = vtk.vtkPolyDataNormals()
		self.normalGenerator.ComputePointNormalsOff()
		self.normalGenerator.ComputeCellNormalsOn()
		self.normalGenerator.AutoOrientNormalsOn()
		self.normalGenerator.ConsistencyOn()
		self.normalGenerator.SetSplitting(0)
		if vtk.VTK_MAJOR_VERSION <= 5:
			self.normalGenerator.SetInput(self.cleanAlg.GetOutput())
		else:
			self.normalGenerator.SetInputConnection(self.cleanAlg.GetOutputPort())

	# Filter input through triangle filter before passing it to our algorithm.
	if vtk.VTK_MAJOR_VERSION <= 5:
		def SetInput(self,inData):
			self.triAlg.SetInput(inData)
			vtk.vtkProgrammableFilter.SetInput(self,self.triAlg.GetOutput())
	else:
		def SetInputConnection(self,inData):
			self.triAlg.SetInputConnection(inData)
			vtk.vtkProgrammableFilter.SetInputConnection(self,self.triAlg.GetOutputPort())

	def _cleanData(self):
		""" This function merges cells and points based on UV Coordinates."""
		
		# Get data and convert texture coordinates to numpy
		data = self.GetPolyDataInput()
		tData = data.GetPointData().GetTCoords()
		tCoords = numpy_support.vtk_to_numpy(tData)

		# Find unique texture coordinates 
		idx = np.lexsort(tCoords.transpose()[::-1])
		diffs = np.abs(np.mean(np.diff(tCoords[idx], axis=0),axis=1))
		diffs = np.insert(diffs,0,1)
		uniqueIds = idx[np.where(diffs > 0.000001)]
		uniqueIds.sort()

		# Filter out points that don't have unique texture coordinates
		pointData = data.GetPoints().GetData()
		points = numpy_support.vtk_to_numpy(pointData)[uniqueIds]
		pointDataNew = vtk.vtkPoints()
		for i in range(points.shape[0]):
			pointDataNew.InsertPoint(uniqueIds[i], points[i])

		# Filter out polygons (cells) that don't have unique texture coordinates
		polys = _cellsToNumpy(data.GetPolys())
		uniqueId = 0
		for i in range(tCoords.shape[0]):
			if(diffs[i]>.000001):
				uniqueId = idx[i]
			else:
				polys[polys == idx[i]] = uniqueId
		vtkPolys = vtk.vtkCellArray()
		for i in range(polys.shape[0]):
			vtkPolys.InsertNextCell( _mkVtkIdList(polys[i]))
		
		# Fill temporary polydata with points and polys
		self.polyData.SetPoints(pointDataNew)
		self.polyData.SetPolys(vtkPolys)
		self.polyData.GetPointData().SetTCoords(data.GetPointData().GetTCoords())
		self.polyData.BuildCells()
		self.polyData.BuildLinks()

		# Use cleanPolyData object to get rid of extra information left over
		# including zero area triangles, edges, and cells 
		self.cleanAlg.Update()
		
		self.normalGenerator.Update()
		self.GetPolyDataOutput().ShallowCopy(self.normalGenerator.GetOutput())

def makeTexturedObjData(objPath, scale=1):
	""" Loads .obj into VTK polyData optimized for searching texture space. 
	
	Args:
		objPath (string): File path to .obj file to load

	Returns:
		polyData (vtk.vtkPolyData): VTK polyData object optimized for finding
			mapping from 2D texture coordinates to 3D world coordinates
	"""
	meshReader = vtk.vtkOBJReader()
	meshReader.SetFileName(objPath)

	cleanFilter = CleanTexturedPolyData()

	if vtk.VTK_MAJOR_VERSION <= 5:
		cleanFilter.SetInput(meshReader.GetOutput())
	else:
		cleanFilter.SetInputConnection(meshReader.GetOutputPort())
	cleanFilter.Update()
	transform = vtk.vtkTransform()
	transform.Scale(scale,scale,scale)
	transformFilter=vtk.vtkTransformPolyDataFilter()
	transformFilter.SetTransform(transform)
	transformFilter.SetInputConnection(cleanFilter.GetOutputPort())
	transformFilter.Update()
	polyData = transformFilter.GetOutput()
	return polyData

def pointToBarycentric(p, a, b, c):
	""" This funciton computes pointToBarycentric coordinates of a point in a triangle.

	Note:
		Transcribed from Christer Ericson's Real-Time Collision Detection
	
	Args:
		p (iterable of type float) : point for which to compute coordinates
		a, b, c (iterables of type float) : points forming a triangle
	"""
	v0 = np.subtract(b,a)
	v1 = np.subtract(c,a)
	v2 = np.subtract(p,a)
	d00 = np.dot(v0, v0)
	d01 = np.dot(v0, v1)
	d11 = np.dot(v1, v1)
	d20 = np.dot(v2, v0)
	d21 = np.dot(v2, v1)
	denom = d00 * d11 - d01 * d01
	v = (d11 * d20 - d01 * d21) / denom
	w = (d00 * d21 - d01 * d20) / denom
	u = 1.0 - v - w
	return u,v,w

class UVToWorldConverter:
	""" This class is used to convert 2D texture coordinates to 3D object space.

	Args:
		data (vtk.vtkPolyData): VTK object which contains geometry and
			texture coordinates
	"""
	def __init__(self, data):
		self.polyData = data
		self.polyData.BuildCells()
		self.tCoords = data.GetPointData().GetTCoords()
		self.points = data.GetPoints()

		self.npTCoords = np.empty((data.GetPolys().GetNumberOfCells(),6))
		self.npPolys = np.empty((data.GetPolys().GetNumberOfCells(),3), dtype=np.uint16)

		nTuples = self.tCoords.GetNumberOfTuples()
		tCoordPoints = vtk.vtkFloatArray()
		tCoordPoints.SetNumberOfComponents(3)
		# tCoordPoints.SetNumberOfTuples(3)
		tCoordPoints.Allocate(nTuples*3)
		tCoordPoints.SetNumberOfTuples(nTuples)
		tCoordPoints.CopyComponent(0, self.tCoords, 0)
		tCoordPoints.CopyComponent(1, self.tCoords, 1)
		tCoordPoints.FillComponent(2,0)
		polyData2D = vtk.vtkPolyData()
		points = vtk.vtkPoints()
		points.SetData(tCoordPoints)
		polyData2D.SetPoints(points)
		polyData2D.SetPolys(data.GetPolys())

		self.pointLocator = vtk.vtkPointLocator()
		self.pointLocator.SetDataSet(data)
		self.pointLocator.BuildLocator()

		self.pointLocator2D = vtk.vtkPointLocator()
		self.pointLocator2D.SetDataSet(polyData2D)
		self.pointLocator.BuildLocator()
		
		data.GetPolys().InitTraversal()
		idList = vtk.vtkIdList()
		counter = 0
		while(data.GetPolys().GetNextCell(idList)):
			for i in range(idList.GetNumberOfIds()):
				tup = self.tCoords.GetTuple(idList.GetId(i))
				self.npPolys[counter,i] = idList.GetId(i)
				self.npTCoords[counter, i*2] = tup[0]
				self.npTCoords[counter, i*2+1] = tup[1]
			counter += 1

	# @timing
	def toWorldSpace(self,p):
		""" This function converts a 2D texture coordinate to 3D object space.
		
		Args:
			p (iterable of type float): 2D texture coordinate in (x,y) format 
				with x and y values between 0 and 1 representing a relative
				position on the image texure.

		Returns:
			worldPoint(iterable of type float): 3D euclidian coordinate of the
				point corresponding to the 2D texture coordinate 'p' in the VTK
				polyData object's frame.
			normalVector(iterable of type float): 3D vector (x,y,z) representing
				the normal vector on the current face.

		TODO:
			* Return color data
		"""
		# bounds = .02
		# lowerBoundX = (self.npTCoords[:,0:2:5] > p[0]-bounds).any(axis=-1)
		# upperBoundX = (self.npTCoords[:,0:2:5] < p[0]+bounds).any(axis=-1)
		# lowerBoundY = (self.npTCoords[:,1:2:6] > p[1]-bounds).any(axis=-1)
		# upperBoundY = (self.npTCoords[:,1:2:6] < p[1]+bounds).any(axis=-1)

		# xBounds = np.logical_and(lowerBoundX,upperBoundX)
		# yBounds = np.logical_and(lowerBoundY,upperBoundY)

		# goodIdx = np.argwhere(np.logical_and(xBounds,yBounds))
		pointIds = vtk.vtkIdList()
		cellIds = vtk.vtkIdList()
		point = self.pointLocator2D.FindClosestNPoints(10,(p[0],p[1],0),pointIds)
		points = []
		for i in range(pointIds.GetNumberOfIds()):
			points += np.argwhere((self.npPolys == pointIds.GetId(i)).any(axis=-1)).transpose().tolist()[0]
		points = np.unique(points)

		for i in points:
			a = self.npTCoords[i,0:2].flatten()
			b = self.npTCoords[i,2:4].flatten()
			c = self.npTCoords[i,4:6].flatten()
			u, v, w = pointToBarycentric(p, a, b, c)
			if u >= 0 and v >=0 and w>=0:
				d = np.array(self.points.GetPoint(self.npPolys[i,0]))
				e = np.array(self.points.GetPoint(self.npPolys[i,1]))
				f = np.array(self.points.GetPoint(self.npPolys[i,2]))
				worldPoint = d*u + e*v + f*w
				normalVector = list(self.polyData.GetCellData().GetNormals().GetTuple(i))
				return worldPoint, normalVector
		return [0,0,0], [0,0,0]

if __name__ == '__main__':
	import sys, getopt
	objFile = ''
	texFile = ''
	# parse command line options
	try:
		opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
	except getopt.error, msg:
		print msg
		print "for help use --help"
		sys.exit(2)
	# process options
	for o, a in opts:
		if o in ("-h", "--help"):
			print __doc__
			sys.exit(0)
		elif o in ("-o", "--obj"):
			inputfile = arg
		elif o in ("-i", "--image"):
			inputfile = arg

	polyData = makeTexturedObjData('Jesus_Unity.obj')
	polyData.BuildCells()
	polyData.BuildLinks()
	uvConverter = UVToWorldConverter(polyData)
	print uvConverter.toWorldSpace((.3,.8))
	print uvConverter.toWorldSpace((-1,0))
