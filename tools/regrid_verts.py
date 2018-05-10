from regrid_base import RegridBase, edgeIntegralFromStreamFunction
import numpy 
import vtk


class RegridVerts(RegridBase):

    # tolerance for finding cell
    EPS = 1.e-12

    def __init__(self):
        """
        Constructor
        no args
        """
        super(RegridVerts, self).__init__()


    def computeWeights(self):
        """
        Compute the interpolation weights
        assuming the field values are stored on vertices
        """

        dstPtIds = vtk.vtkIdList()
        cell = vtk.vtkGenericCell()
        pcoords = numpy.zeros((3,), numpy.float64)
        ws = numpy.zeros((4,), numpy.float64)
        
        numSrcCells = self.srcGrid.GetNumberOfCells()
        numDstCells = self.dstGrid.GetNumberOfCells()

        # iterate over the dst grid cells
        for dstCellId in range(numDstCells):

            # iterate over the four vertices of the dst cell
            self.dstGrid.GetCellPoints(dstCellId, dstPtIds)
            for i0 in range(4):

                dstVert = self.dstGrid.GetPoint(dstPtIds.GetId(i0))

                # bilinear interpolation
                srcCellId = self.srcLoc.FindCell(dstVert, self.EPS, cell, pcoords, ws)
                if srcCellId >= 0:
                    k = (dstCellId, srcCellId)
                    if not self.weights.has_key(k):
                        # initialize the weights
                        self.weights[k] = self.ZERO4x4.copy()
                    # a point can only be in one cell, so no need to use +=
                    self.weights[k][i0, :] = ws


###############################################################################
def main():
    from math import pi, sin, cos, log, exp
    import argparse

    parser = argparse.ArgumentParser(description='Regriod edge field')
    parser.add_argument('-s', dest='src', default='um:um.nc', help='Specify source grid file as FORMAT:FILENAME.nc where FORMAT is "ugrid" or "um"')
    parser.add_argument('-p', dest='padding', type=int, default=0, 
                              help='Specify by how much the source grid should be padded on the high lon side (only for UM grids)')
    parser.add_argument('-d', dest='dst', default='ugrid:mesh_C4.nc', help='Specify destination grid file as FORMAT:FILENAME.nc where FORMAT is "ugrid" or "um"')
    parser.add_argument('-f', dest='streamFunc', default='x', 
                        help='Stream function as a function of x (longitude in rad) and y (latitude in rad)')
    args = parser.parse_args()

    srcFormat, srcFilename = args.src.split(':')
    dstFormat, dstFilename = args.dst.split(':')

    rgrd = RegridVerts()
    rgrd.setSrcGridFile(srcFilename, format=srcFormat, padding=args.padding)
    rgrd.setDstGridFile(dstFilename, format=dstFormat)
    rgrd.computeWeights()

    # compute stream function on cell vertices
    x, y = rgrd.getSrcLonLat()
    srcPsi = eval(args.streamFunc)

    # compute edge integrals
    srcEdgeVel = edgeIntegralFromStreamFunction(srcPsi)

    # apply the weights 
    dstEdgeVel = rgrd.applyWeights(srcEdgeVel)

    # compute the exact edge field on the destination grid
    x, y = rgrd.getDstLonLat()
    dstPsi = eval(args.streamFunc)
    dstEdgeVelExact = edgeIntegralFromStreamFunction(dstPsi)

    # compute the error
    diff = numpy.fabs(dstEdgeVelExact - dstEdgeVel)
    maxError = diff.max()
    minError = diff.min()
    print('Min/max error              : {}/{}'.format(minError, maxError))
    error = numpy.fabs(diff).sum()
    print('Sum of interpolation errors: {}'.format(error))

    rgrd.saveDstLoopData(dstEdgeVel, 'dstVertsLoops.vtk')



if __name__ == '__main__':
    main()

