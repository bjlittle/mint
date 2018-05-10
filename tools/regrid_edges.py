from regrid_base import RegridBase, edgeIntegralFromStreamFunction
from polyline_iter import PolylineIter
from polysegment_iter import PolysegmentIter
import numpy
import vtk


class RegridEdges(RegridBase):
    """
    Class for regridding edge field to edge field using a cell by cell approach
    """

    def __init__(self):
        """
        Constructor
        no args
        """
        super(RegridEdges, self).__init__()


    def computeWeights(self):
        """
        Compute the interpolation weights
        """

        dstPtIds = vtk.vtkIdList()
        
        numSrcCells = self.srcGrid.GetNumberOfCells()
        numDstCells = self.dstGrid.GetNumberOfCells()

        # iterate over the dst grid cells
        for dstCellId in range(numDstCells):

            # iterate over the four edges of each dst cell
            self.dstGrid.GetCellPoints(dstCellId, dstPtIds)
            for i0 in range(4):

                i1 = (i0 + 1) % 4

                # get the start/end points of the dst edge
                id0 = dstPtIds.GetId(i0)
                id1 = dstPtIds.GetId(i1)
                dstEdgePt0 = self.dstGrid.GetPoint(id0)
                dstEdgePt1 = self.dstGrid.GetPoint(id1)

                # represent the edge as a broken line
                bli = PolylineIter([dstEdgePt0, dstEdgePt1])

                # compute the intersections of the dst edge with the src grid
                bsi = PolysegmentIter(self.srcGrid, self.srcLoc, bli)

                # compute the contributions to this edge
                for seg in bsi:

                    srcCellId = seg.getCellId()
                    xia = seg.getBegCellParamCoord()
                    xib = seg.getEndCellParamCoord()
                    coeff = seg.getCoefficient()

                    dxi = xib - xia
                    xiMid = 0.5*(xia + xib)

                    k = (dstCellId, srcCellId)

                    # compute the weights from each src edge
                    ws = numpy.array([+ dxi[0] * (1.0 - xiMid[1]) * coeff,
                                      + dxi[1] * (0.0 + xiMid[0]) * coeff,
                                      - dxi[0] * (0.0 + xiMid[1]) * coeff,
                                      - dxi[1] * (1.0 - xiMid[0]) * coeff])

                    if not self.weights.has_key(k):
                        # initialize the weights
                        self.weights[k] = self.ZERO4x4.copy()

                    self.weights[k][i0, :] += ws

                totalT = bsi.getIntegratedParamCoord()
                if abs(totalT - 1.0) > 1.e-6:
                    print('Warning: total t of segment: {:.3f} != 1 (diff={:.1g}), dst cell {} points=[{}, {}]'.format(\
                        totalT, totalT - 1.0, dstCellId, dstEdgePt0, dstEdgePt1))

        # DEBUG
        #print '*** self.weights[71, 71L] = ', self.weights.get((71, 71L), [])
        #print '*** self.weights[71, 34L] = ', self.weights.get((71, 34L), [])
        #print '*** self.weights[71, 67L] = ', self.weights.get((71, 67L), [])
        #print '*** self.weights[71, 70L] = ', self.weights.get((71, 70L), [])
        #print '*** self.weights[71, 75L] = ', self.weights.get((71, 75L), [])



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

    rgrd = RegridEdges()
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

    rgrd.saveDstLoopData(dstEdgeVel, 'dstEdgesLoops.vtk')

if __name__ == '__main__':
    main()

