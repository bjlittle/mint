import numpy


class EdgeToVerts:

	def __init__(self,):
		pass


	def setGrid(self, grid):
		self.grid = grid
        self.numCells = self.grid.GetNumberOfCells()
        self.points = self.__getNumpyArrayFromVtkDoubleArray(self.numCells, 3, 
                                                             self.grid.GetPoints().GetData())
        self.computeJacobian()
        self.computeVectorBasesOnVerts()


	def setEdgeField(self, edgeData):
		self.edgeData = edgeData


	def saveToVTKFile(self, filename):
		pass


    def computeJacobian(self,):
        self.jac = numpy.zeros((self.numCells,), numpy.float64)
        for i in range(self.numCells):
            d10 = self.points[i, 1, :] - self.points[i, 0, :]
            d21 = self.points[i, 2, :] - self.points[i, 1, :]
            d32 = self.points[i, 3, :] - self.points[i, 2, :]
            d03 = self.points[i, 0, :] - self.points[i, 3, :]
            self.jac[i] = 0.5*(numpy.cross(d10, d21).dot(zHat) + numpy.cross(d32, d03).dot(zHat))


    def computeVectorBasesOnVerts(self):

        zHat = numpy.array([0., 0., 1.])

        self.vertVecBases = numpy.zeros((self.numCells, 4, 3))

        for i in range(self.numCells):
            d10 = self.points[i, 1, :] - self.points[i, 0, :]
            d21 = self.points[i, 2, :] - self.points[i, 1, :]
            d32 = self.points[i, 3, :] - self.points[i, 2, :]
            d03 = self.points[i, 0, :] - self.points[i, 3, :]
            jac = 0.5*(numpy.cross(d10, d21).dot(zHat) + numpy.cross(d32, d03).dot(zHat))

            drdXsi00 = d10
            drdEta00 = -d03

            drdXsi10 = d10
            drdEta10 = d21

            drdXsi11 = -d32
            drdEta11 = d21

            drdXsi01 = -d32
            drdEta01 = -d03

            # grad xsi = (dr/d eta x zHat) / jac
            gradXsi00 = numpy.cross(drdEta00, zHat) / jac
            gradXsi10 = numpy.cross(drdEta10, zHat) / jac
            gradXsi11 = numpy.cross(drdEta11, zHat) / jac
            gradXsi01 = numpy.cross(drdEta01, zHat) / jac

            # grad eta = (zHat x dr/d xsi) / jac
            gradEta00 = numpy.cross(zHat, drdXsi00) / jac
            gradEta10 = numpy.cross(zHat, drdXsi10) / jac
            gradEta11 = numpy.cross(zHat, drdXsi11) / jac
            gradEta01 = numpy.cross(zHat, drdXsi01) / jac

            self.vertVecBases[i, 0, :] = gradXsi00 + gradEta00
            self.vertVecBases[i, 1, :] = gradXsi10 + gradEta10
            self.vertVecBases[i, 2, :] = gradXsi11 + gradEta11
            self.vertVecBases[i, 3, :] = gradXsi01 + gradEta01




    def __getNumpyArrayFromVtkDoubleArray(self, numCells, numComponents, vtkArray):
        """
        Get a numpy array from a VTK array
        @param numCells number of cells
        @param numComponents number of components
        @param vtkArray instance of vtkDoubleArray
        @return numpy array
        """
        # vtkArray.GetVoidPointer(0) returns the address of the pointer as a string
        # "_X_void_p" where X is the hex address. Then convert X
        # to an int using base 16.
        addr = int(vtkArray.GetVoidPointer(0).split('_')[1], 16)
        ArrayType = ctypes.c_double * numCells * 4 * numComponents
        if numComponents == 1:
            return numpy.frombuffer(ArrayType.from_address(addr)).reshape((numCells, 4))
        else:
            return numpy.frombuffer(ArrayType.from_address(addr)).reshape((numCells, 4, numComponents))


