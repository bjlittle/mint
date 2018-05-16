import numpy


class EdgeToCells:

    def __init__(self,):
        """
        Constructor
        no args
        """
        pass


    def setGrid(self, grid):
        """
        Set the grid
        @param grid instance of vtkUnstructuredGrid
        """
        self.grid = grid
        self.numCells = self.grid.GetNumberOfCells()
        # returns the grid vertices as a numpy array
        self.points = self.__getNumpyArrayFromVtkDoubleArray(self.numCells, 3, 
                                                             self.grid.GetPoints().GetData())

    def setEdgeField(self, name, edgeData):
        """
        Set the edge field
        @param name will use this name to store the field in the VTK grid
        @param edgeData a numpy array of size (numcCells, 4)
        """

        zHat = numpy.array([0., 0., 1.])

        # store vector values at cell centres
        self.vectorValues = numpy.zeros((self.numCells, 3))

        # iterate over cells
        for i in range(self.numCells):

            # edge displacement, go anti-clockwise around the cell
            d10 = self.points[i, 1, :] - self.points[i, 0, :]
            d21 = self.points[i, 2, :] - self.points[i, 1, :]
            d32 = self.points[i, 3, :] - self.points[i, 2, :]
            d03 = self.points[i, 0, :] - self.points[i, 3, :]

            # Jacobian
            jac = 0.5*(numpy.cross(d10, d21).dot(zHat) + numpy.cross(d32, d03).dot(zHat))

            drdXsiLo = + d10
            drdXsiHi = - d32
            drdXsi = 0.5*(drdXsiLo + drdXsiHi)

            drdEtaLo = - d03
            drdEtaHi = + d21
            drdEta = 0.5*(drdEtaLo + drdEtaHi)

            gradXsi = numpy.cross(drdEta, zHat) / jac
            gradEta = numpy.cross(zHat, drdXsi) / jac

            self.vectorValues[i, :] = 0.5*(edgeData[i, 0] - edgeData[i, 2])*gradXsi \
                                    + 0.5*(edgeData[i, 1] - edgeData[i, 3])*gradEta

        self.vecData = vtk.vtkDoubleArray()
        self.vecData.SetName(name)
        self.vecData.SetNumberOfComponents(3)
        self.vecData.SetNumberOfTuples(self.numCells)
        self.vecData.SetVoidArray(self.vectorValues, self.numCells * 3, 1)

        # attach the cell centred vector field to the grid
        self.grid.GetCellData().AddArray(self.vecData)


    def saveToVTKFile(self, filename):
        """
        Save the grid to a VTK file
        @param filename VTK file
        """
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(self.vtk['grid'])
        writer.Update()


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


