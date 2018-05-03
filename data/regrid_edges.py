from ugrid_reader import UgridReader
from broken_line_iter import BrokenLineIter
from broken_segments_iter import BrokenSegmentsIter
import numpy


class RegridEdges:

    ZERO4x4 = numpy.zeros((4,4), numpy.float64)

	def __init__(self):
        """
        Constructor
        no args
        """
		
		# (dstCellId, srcCellId) -> weight as a 4x4 matrix
		self.weights = {}

		self.srcGrid = None
		self.srcLoc = None

		self.dstGrid = None


	def setSrcGridFile(self, filename):
        """
        Set source grid
        @param file name containing UGRID description of the grid
        """ 
		ur = UgridReader(filename)
		self.srcGrid = ur.getUnstructuredGrid()
		seld.srcLoc = ur.getUnstructuredGridLocator()


	def setDstGridFile(self, filename):
        """
        Set destination grid
        @param file name containing UGRID description of the grid
        """	
		ur = UgridReader(filename)
		self.dstGrid = ur.getUnstructuredGrid()


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

                dstK = (i0, dstCellId)
                self.weights[dstK] = {}

                # get the start/end points of the dst edge
                dstVertId0 = dstPtIds.GetId(i0)
                dstVertId1 = dstPtIds.GetId(i1)
                dstEdgePt0 = self.dstGrid.GetPoint(dstVertId0)
                dstEdgePt1 = self.dstGrid.GetPoint(dstVertId1)

                # compute the intersections
                bli = BrokenLineIter([dstEdgePt0, dstEdgePt1])

                # find the intersection with the source grid
                bsi = BrokenSegmentsIter(self.srcGrid, self.srcLoc, bli)

                # compute the contribution to this edge
                for seg in bsi:
                	srcCellId = seg.getCellId()
                    xia = seg.getBegCellParamCoord()
                    xib = seg.getEndCellParamCoord()

                    dxi = xib - xia
                    xiMid = 0.5*(xia + xib)

                    k = (dstCellId, srcCellId)

                    # compute the weights from each src edge
                    ws = numpy.array([dxi[0]*(1.0 - xiMid[1]),
                    	              dxi[1]*(0.0 + xiMid[0]),
                    	              dxi[0]*(0.0 + xiMid[1]),
                    	              dxi[1]*(1.0 - xiMid[0])])

                    self.weights[k] = self.weights.get(k, self.ZERO4x4)
                    self.weights[k][i0, :] += ws


	def applyWeights(self, srcData):
		"""
		Apply the interpolation weights to the field
		@param srcData line integrals on the source grid edges
		@return line integrals on the destination grid
		"""
		numDstCells = self.dstGrid.GetNumberOfCells()
		res = zeros.zeros((numDstCells, 4), numpy.float64)
        for k in self.weights:
            dstCellId, srcCellId, k
            res[dstCellId, :] = self.weights[k].dot(srcData[srcCellId, :])
        return res

###############################################################################
def main():
    pass

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Regriod edge field')
    parser.add_argument('-s', dest='src', default='mesh_C4.nc', help='Specify UGRID source grid file')
    parser.add_argument('-d', dest='dst', default='mesh_C4.nc', help='Specify UGRID destination grid file')
    args = parser.parse_args()

    rgrd = RegridEdges()
    rgrd.setSrcGridFile(args.src)
    rgrd.setDstGridFile(args.dst)
    rgrd.computeWeights()
