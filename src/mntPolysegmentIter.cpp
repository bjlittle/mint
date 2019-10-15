#include <mntPolysegmentIter.h>
#include <vtkIdList.h>
#include <vtkGenericCell.h>
#include <limits>
#include <map>
#include <vector>

struct TCmpFunctor {
    TCmpFunctor(const std::vector<double>& ts) {
        this->tVals = ts;
    }
    bool operator()(size_t i, size_t j) {
        return (this->tVals[i] < this->tVals[j]);
    }
    std::vector<double> tVals;
};


PolysegmentIter::PolysegmentIter(vtkUnstructuredGrid* grid,
                                 vmtCellLocator* locator, 
                                 const double p0In[], const double p1In[],
                                 double periodicityLength) {

    // small tolerances 
    this->eps = 10 * std::numeric_limits<double>::epsilon();
    this->eps100 = 100. * this->eps;
    this->tol = 1.e-3; // to determine if a point is inside a cell

    this->xPeriodicity = periodicityLength;

    // set the grid and the grid locator
    this->grid = grid;
    this->locator = locator;

    Vec3 p0(p0In);
    this->__makePeriodic(p0);
    Vec3 p1(p1In);
    this->__makePeriodic(p1);

    Vec3 dp = p1 - p0;

    std::vector< std::pair<vtkIdType, Vec2> > cellIdLambdas = this->locator->findIntersectionsWithLine(p0, p1);

    // arrays of cell Ids, start/end t values, start/end xi param coords, and 
    // duplicity coefficients for each subsegment
    this->segCellIds.resize(0);
    this->segTas.resize(0);
    this->segTbs.resize(0);
    this->segXias.resize(0);
    this->segXibs.resize(0);
    this->segCoeffs.resize(0);

    double closestPoint[3];
    int subId;
    Vec3 xia, xib;
    double dist2;
    double weights[8];
    for (auto& cIdLam : cellIdLambdas) {

        vtkIdType cId = cIdLam.first;

        double ta = cIdLam.second[0];
        double tb = cIdLam.second[1];

        Vec3 pa = p0 + dp*ta;
        Vec3 pb = p0 + dp*tb;

        // compute the cell parametric coords
        vtkCell* cell = this->grid->GetCell(cId);
        int ina = cell->EvaluatePosition(&pa[0], closestPoint, subId, &xia[0], dist2, weights);
        int inb = cell->EvaluatePosition(&pb[0], closestPoint, subId, &xib[0], dist2, weights);

        // fill in
        this->segCellIds.push_back(cId);
        this->segTas.push_back(ta);
        this->segTbs.push_back(tb);
        this->segXias.push_back(xia);
        this->segXibs.push_back(xib);
        this->segCoeffs.push_back(1.0);
    }


    // assign coefficients that account for duplicity, ie segments 
    // that are shared between two cells. Output is this->segCoeffs
    this->__assignCoefficientsToSegments();

    this->numSegs = this->segCellIds.size();

    // compute the total, integrated linear param coord
    // should amoount to 1 is the target is entirely 
    // contained within the source grid
    this->totalT = 0.0;
    for (size_t i = 0; i < this->numSegs; ++i) {
        double ta = this->segTas[i];
        double tb = this->segTbs[i];
        double coeff = this->segCoeffs[i];
        this->totalT += (tb - ta) * coeff;
    }

    // reset the iterator
    this->reset();
}


double 
PolysegmentIter::getIntegratedParamCoord() const {
    return this->totalT;
}


void 
PolysegmentIter::reset() {
    this->index = 0;
}


bool
PolysegmentIter::next() {
    if (this->index < this->numSegs - 1) {
        this->index++;
        return true;
    }
    return false;
}


vtkIdType 
PolysegmentIter::getCellId() const {
    return this->segCellIds[this->index];
}

const Vec3& 
PolysegmentIter::getBegCellParamCoord() const {
    return this->segXias[this->index];
}        


const Vec3& 
PolysegmentIter::getEndCellParamCoord() const {
    return this->segXibs[this->index];
}
 

double
PolysegmentIter::getBegLineParamCoord() const {
    return this->segTas[this->index];
}
        

double 
PolysegmentIter::getEndLineParamCoord() const {
    return this->segTbs[this->index];
}
     

double 
PolysegmentIter::getCoefficient() const {
    return this->segCoeffs[this->index];
}
 
size_t
PolysegmentIter::getNumberOfSegments() const {
    return this->numSegs;
}


///////////////////////////////////////////////////////////////////////////////
// private methods

void
PolysegmentIter::__assignCoefficientsToSegments() {

    size_t n = this->segCellIds.size();

    // copy
    std::vector<vtkIdType> sCellIds = this->segCellIds;
    std::vector<double> sTas = this->segTas;
    std::vector<double> sTbs = this->segTbs;
    std::vector<Vec3> sXias = this->segXias;
    std::vector<Vec3> sXibs = this->segXibs;
    std::vector<double> sCoeffs = this->segCoeffs;
    this->segCellIds.resize(0);
    this->segTas.resize(0);
    this->segTbs.resize(0);
    this->segXias.resize(0);
    this->segXibs.resize(0);
    this->segCoeffs.resize(0);
    for (size_t i = 0; i < n; ++i) {
        double ta = sTas[i];
        double tb = sTbs[i];
        if (std::abs(tb - ta) > this->eps100) {
            this->segCellIds.push_back(sCellIds[i]);
            this->segTas.push_back(sTas[i]);
            this->segTbs.push_back(sTbs[i]);
            this->segXias.push_back(sXias[i]);
            this->segXibs.push_back(sXibs[i]);
            this->segCoeffs.push_back(sCoeffs[i]);
        }
    }

    // reduce contribution for overlapping segments. If two 
    // segments overlap then the coefficient of first segment
    // is set to 1.0 - overlap/(tb - ta). Assumes overlap 
    // can only happen for pairs of segment
    n = this->segCellIds.size(); // changed after removing zero length sub-segments
    // iterate over sub-segment pairs
    for (int i0 = 0; i0 < (int) n - 1; ++i0) {
        size_t i1 = i0 + 1;
        double ta0 = this->segTas[i0];
        double tb0 = this->segTbs[i0];
        double ta1 = this->segTas[i1];
        double tb1 = this->segTbs[i1];
        double overlap = std::max(0., std::min(tb0, tb1) - std::max(ta1, ta0));
        this->segCoeffs[i0] = 1.0 - overlap/(tb0 - ta0);
    }

}

void
PolysegmentIter::__collectIntersectionPoints(const Vec3& pBeg, 
                                             const Vec3& pEnd,
                                             std::vector<vtkIdType>& cIds,
                                             std::vector<double>& lambRays,
                                             std::vector<Vec3>& points) {
    LineLineIntersector intersector;
    vtkIdList* cellIds = vtkIdList::New();
    vtkIdList* ptIds = vtkIdList::New();

    Vec3 v0;
    Vec3 v1;

    // vector from start to finish
    Vec3 dp = pEnd - pBeg;

    // find all the cells intersected by the line
    this->locator->FindCellsAlongLine((double*) &pBeg[0], 
                                      (double*) &pEnd[0], 
                                      this->tol, cellIds);

    //
    // collect the intersection points
    //

    // iterate over the cells along the line
    for (vtkIdType i = 0; i < cellIds->GetNumberOfIds(); ++i) {

        // this cell Id
        vtkIdType cId = cellIds->GetId(i);

        // vertices, ptIds.GetNumberOfIds() should return 4
        // since we're dealing with quads only
        this->grid->GetCellPoints(cId, ptIds);


        // iterate over the quads' edges
        const int numQuadNodes = 4;
        for (int edgeId = 0; edgeId < numQuadNodes; ++edgeId) {

        	int j0 = edgeId;
            int j1 = (j0 + 1) % numQuadNodes;

            // swap direction for the last two edges
            // (edges always point in the positive direction)
            if (j0 >= 2) {
            	int tmp = j0;
            	j0 = j1;
            	j1 = tmp;
            }

            this->grid->GetPoint(ptIds->GetId(j0), &v0[0]);
            this->grid->GetPoint(ptIds->GetId(j1), &v1[0]);

            // look for an intersection
            intersector.setPoints(&pBeg[0], &pEnd[0], &v0[0], &v1[0]);

            if (! intersector.hasSolution(this->eps)) {
                // skip if no solution. FindCellsAlongLine may be too generous with
                // returning the list of intersected cells
                continue;
            }

            // we have a solution but it could be degenerate

            if (std::abs(intersector.getDet()) > this->eps) {
                // normal intersection, 1 solution
                Vec2 sol = intersector.getSolution();
                double lambRay = sol[0];
                double lambEdg = sol[1];

                // is it valid? Intersection must be within (p0, p1) and (q0, q1)
                if (lambRay >= (0. - this->eps100) && lambRay <= (1. + this->eps100)  && 
                    lambEdg >= (0. - this->eps100) && lambEdg <= (1. + this->eps100)) {

                    // compute the intersection point
                    Vec3 p = pBeg + lambRay*dp;

                    // add to list
                    cIds.push_back(cId);
                    lambRays.push_back(lambRay);
                    points.push_back(p); // copies
                }
            }
            else {
                // det is almost zero
                // looks like the two lines (p0, p1) and (q0, q1) are overlapping
                // add the start/end points
                const std::pair<double, double> sol = intersector.getBegEndParamCoords();

                // linear param coord along line
                double lama = sol.first;
                double lamb = sol.second;

                // compute the points
                Vec3 pa = pBeg + lama*dp;
                Vec3 pb = pBeg + lamb*dp;

                // add to lists both points
                cIds.push_back(cId);
                lambRays.push_back(lama);
                points.push_back(pa);

                cIds.push_back(cId);
                lambRays.push_back(lamb); // same Id as before
                points.push_back(pb);

            }

        } // end of edge loop

    } // end of cell loop

    // clean up
    cellIds->Delete();
    ptIds->Delete();
  
}


void  
PolysegmentIter::__makePeriodic(Vec3& v) {

    // fix start/end points if they fall outside the domain and the domain is periodic
    if (this->xPeriodicity > 0.) {
        double xmin = this->grid->GetBounds()[0];
        double xmax = this->grid->GetBounds()[1];
        if (v[0] < xmin) {
            std::cerr << "Warning: adding periodicity length " << this->xPeriodicity << 
                         " to point " << v << "\n";
            v[0] += this->xPeriodicity;
        }
        else if (v[0] > xmax) {
            std::cerr << "Warning: subtracting periodicity length " << this->xPeriodicity << 
                         " from point " << v << "\n";
            v[0] -= this->xPeriodicity;
        }
    }

}

