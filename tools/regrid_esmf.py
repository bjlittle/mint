from ugrid_reader import UgridReader
from latlon_reader import LatLonReader
from regrid_base import RegridBase
from edge_to_cells import EdgeToCells
import numpy 
import vtk
import ESMF


class RegridEsmf(RegridBase):


    def __init__(self):
        """
        Constructor
        no args
        """
        super(RegridEsmf, self).__init__()
        self.manager = ESMF.Manager(debug=True)


    def computeWeights(self):
        """
        Compute the interpolation weights
        """

        # construct src and dst meshes
        self.esmfSrcMesh = self.createEsmfMesh(self.srcGrid)
        self.esmfDstMesh = self.createEsmfMesh(self.dstGrid)

        # ESMF wants a field to compute the weights
        self.esmfSrcField = ESMF.Field(self.esmfSrcMesh, name="srcfield", meshloc=ESMF.MeshLoc.ELEMENT)
        self.esmfDstField = ESMF.Field(self.esmfDstMesh, name="dstfield", meshloc=ESMF.MeshLoc.ELEMENT)

        self.esmfSrcField.data[:] = 0.0
        self.esmfDstField.data[:] = 0.0

        # compute the weights
        self.regrid = ESMF.Regrid(self.esmfSrcField, self.esmfDstField,
                                  regrid_method=ESMF.RegridMethod.CONSERVE,
                                  unmapped_action=ESMF.UnmappedAction.IGNORE)


    def applyWeights(self, srcData):
        """
        Apply the interpolation weights to the field
        @param srcData line integrals on the source grid edges, dimensioned (numSrcCells, 4)
        @return line integrals on the destination grid, array dimensioned (numDstCells, 4)
        """

        # convert the edge field to a cell centred vector field with collocated components
        e2c = EdgeToCells()
        e2c.setGrid(self.srcGrid)
        e2c.setEdgeField('vector_field', srcData)

        dstNumCells = self.dstGrid.GetNumberOfCells()
        res = numpy.zeros((dstNumCells, 3), numpy.float64)

        vecData = e2c.getCellVectorField()

        #
        # interpolate the two components separately
        #

        # X
        self.esmfSrcField.data[:] = vecData[:, 0]
        field = self.regrid(self.esmfSrcField, self.esmfDstField)
        res[:, 0] = field.data[:]

        # Y
        self.esmfSrcField.data[:] = vecData[:, 1]
        field = self.regrid(self.esmfSrcField, self.esmfDstField)
        res[:, 1] = field.data[:]

        return res



    def createEsmfMesh(self, grid):
    	"""
    	Create ESMF mesh from VTK grid
        @param grid instance of vtkUnstructuredGrid
    	"""
    	mesh = ESMF.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=ESMF.CoordSys.SPH_RAD)
    	lons, lats = self.getGridLonLat(grid)

        numCells = grid.GetNumberOfCells()
        numNodes = numCells*4
    	self.xy = numpy.zeros((numNodes, 2), numpy.float64)
    	self.xy[:, 0] = lons.flat
    	self.xy[:, 1] = lats.flat

    	self.nodeIds = numpy.arange(0, numNodes) # zero indexing
    	self.nodeOwners = numpy.zeros(numNodes) # all running on PE 0
    	mesh.add_nodes(numNodes, self.nodeIds, self.xy, self.nodeOwners)

        self.cellConn = numpy.reshape(self.nodeIds, (numCells, 4))
        self.cellTypes = ESMF.MeshElemType.QUAD * numpy.ones((numCells,), numpy.int)
        self.cellIds = numpy.arange(0, numCells) # zero indexing?
        # cell centre coords
        self.cellCoords = 0.25 * self.xy.reshape((numCells, 4, 2)).sum(axis=1) # average cell coordinates
    	mesh.add_elements(numCells, self.cellIds, self.cellTypes, self.cellConn, element_coords=self.cellCoords)

        return mesh



###############################################################################
def main():
    from math import pi, sin, cos, log, exp
    import argparse

    parser = argparse.ArgumentParser(description='Regriod edge field as if it were a nodal field')
    parser.add_argument('-s', dest='src', default='src.vtk', help='Specify source file in VTK unstructured grid format')
    parser.add_argument('-v', dest='varname', default='edge_integrated_velocity', help='Specify edge staggered field variable name in source VTK file')
    parser.add_argument('-d', dest='dst', default='dst.vtk', help='Specify destination file in VTK unstructured grid format')
    parser.add_argument('-o', dest='output', default='', help='Specify output VTK file where regridded edge data is saved')
    args = parser.parse_args()

    rgrd = RegridEsmf()
    rgrd.setSrcFile(args.src)
    rgrd.setDstFile(args.dst)
    rgrd.computeWeights()

    # compute edge integrals
    srcEdgeVel = rgrd.getSrcEdgeData(args.varname)

    # regrid/apply the weights 
    dstCellVel = rgrd.applyWeights(srcEdgeVel)

    if args.output:
        varr = vtk.vtkDoubleArray()
        varr.SetNumberOfComponents(dstCellVel.shape[1])
        varr.SetNumberOfTuples(dstCellVel.shape[0])
        varr.SetVoidArray(dstCellVel, dstCellVel.shape[0]*dstCellVel.shape[1], 1)
        varr.SetName('cell_velocity')
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(args.output)
        writer.SetInputData(rgrd.getDstGrid())
        writer.Update()

if __name__ == '__main__':
    main()


