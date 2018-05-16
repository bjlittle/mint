import numpy
import vtk
import ctypes


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
        self.points = self.getNumpyArrayFromVtkDoubleArray(self.numCells, 3, 
                                                           self.grid.GetPoints().GetData())

    def getEdgeDataByName(self, varname):
        """
        Get the edge field as a numpy array by name
        @param varname edge field name
        """
        data = self.grid.GetCellData().GetArray(varname)
        numCells = self.grid.GetNumberOfCells()
        return self.getNumpyArrayFromVtkDoubleArray(numCells, 1, data)


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

            # average the X and Y components
            self.vectorValues[i, :] = 0.5*(edgeData[i, 0] - edgeData[i, 2])*gradXsi \
                                    + 0.5*(edgeData[i, 1] - edgeData[i, 3])*gradEta

        self.vecData = vtk.vtkDoubleArray()
        self.vecData.SetName(name)
        self.vecData.SetNumberOfComponents(3)
        self.vecData.SetNumberOfTuples(self.numCells)
        self.vecData.SetVoidArray(self.vectorValues, self.numCells * 3, 1)

        # attach the cell centred vector field to the grid
        self.grid.GetCellData().AddArray(self.vecData)

        self.vecVarName = name


    def getCellVectorField(self):
        """
        Get the cell centred vector field
        @return numpy array
        """
        data = self.grid.GetCellData().GetArray(self.vecVarName)
        addr = int(data.GetVoidPointer(0).split('_')[1], 16)
        ArrayType = ctypes.c_double * self.numCells * 3
        return numpy.frombuffer(ArrayType.from_address(addr)).reshape((self.numCells, 3))


    def saveToVtkFile(self, filename):
        """
        Save the grid to a VTK file
        @param filename VTK file
        """
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(self.grid)
        writer.Update()


    def getNumpyArrayFromVtkDoubleArray(self, numCells, numComponents, vtkArray):
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

###############################################################################

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Read ugrid file')
    parser.add_argument('-i', dest='input', default='input.vtk', help='Specify input VTK file with edge field')
    parser.add_argument('-v', dest='varname', default='edge_integrated_velocity', help='Specify variable name of edge field')
    parser.add_argument('-o', dest='output', default='output.vtk', help='Save grid with cell centred vector field in VTK file')

   
    args = parser.parse_args()

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(args.input)
    reader.Update()

    grid = reader.GetOutput()

    e2c = EdgeToCells()
    e2c.setGrid(grid)
    edgeData = e2c.getEdgeDataByName(args.varname)
    e2c.setEdgeField('vector_field', edgeData)

    if args.output:
        e2c.saveToVtkFile(args.output)

if __name__ == '__main__':
    main()
