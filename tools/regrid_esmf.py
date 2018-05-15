from ugrid_reader import UgridReader
from latlon_reader import LatLonReader
from regrid_base import RegridBase
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
        numSrcCells = self.srcGrid.GetNumberOfCells()
        # average the line integrals to cell centres, taking into account the edge orientations
        srcAvgDataX = 0.5*(srcData[:, 0] - srcData[:, 2])
        srcAvgDataY = 0.5*(srcData[:, 1] - srcData[:, 3])

        numDstCells = self.dstGrid.GetNumberOfCells()
        dstDataX = numpy.zeros((numDstCells,), numpy.float64)
        dstDataY = numpy.zeros((numDstCells,), numpy.float64)

        # logical X component
        self.esmfSrcField.data[:] = srcAvgDataX
        self.regrid(self.esmfSrcField, self.esmfDstField)
        dstDataX[:] = self.esmfDstField.data

        # logical Y component
        self.esmfSrcField.data[:] = srcAvgDataY
        self.regrid(self.esmfSrcField, self.esmfDstField)
        dstDataY[:] = self.esmfDstField.data

        # extrapolate the cell centred line interpolation to 
        # the cell edges
        res = numpy.zeros((numDstCells, 4), numpy.float64)
        res[:, 0] = dstDataX
        res[:, 1] = dstDataY
        res[:, 2] = -dstDataX
        res[:, 3] = -dstDataY

        return res



    def createEsmfMesh(self, grid):
    	"""
    	Create ESMF mesh from VTK grid
        @param grid instance of vtkUnstructuredGrid
    	"""
    	mesh = ESMF.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=ESMF.CoordSys.SPH_RAD)
    	lons, lats = self.getGridLonLat(grid)
    	numNodes = grid.GetNumberOfPoints()
    	self.xy = numpy.zeros((numNodes, 2), numpy.float64)
    	self.xy[:, 0] = lons.flat
    	self.xy[:, 1] = lats.flat
    	self.nodeIds = numpy.arange(0, numNodes) # zero indexing?
    	self.nodeOwners = numpy.zeros(numNodes)
    	mesh.add_nodes(numNodes, self.nodeIds, self.xy, self.nodeOwners)

        numCells = grid.GetNumberOfCells()
        nIds = self.nodeIds.reshape((numCells, 4))
        self.cellConn = numpy.reshape(self.nodeIds, (numCells, 4))
        self.cellTypes = ESMF.MeshElemType.QUAD * numpy.ones((numCells,), numpy.int)
        self.cellIds = numpy.arange(0, numCells) # zero indexing?
    	mesh.add_elements(numCells, self.cellIds, self.cellTypes, self.cellConn)
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
    dstEdgeVel = rgrd.applyWeights(srcEdgeVel)

    # loop integrals for each cell
    cellLoops = dstEdgeVel.sum(axis=1)

    # statistics
    absCellIntegrals = numpy.abs(cellLoops)
    minAbsLoop = absCellIntegrals.min()
    maxAbsLoop = absCellIntegrals.max()
    avgAbsLoop = absCellIntegrals.sum() / float(absCellIntegrals.shape[0])

    print('Min/avg/max cell loop integrals: {}/{}/{}'.format(minAbsLoop, avgAbsLoop, maxAbsLoop))
    if args.output:
        rgrd.saveDstEdgeData(args.varname, dstEdgeVel, args.output)


if __name__ == '__main__':
    main()


