#include <vmtCellLocator.h>
#include <vtkPoints.h>
#include <vtkCell.h>
#include <mntLineLineIntersector.h>


struct LambdaBegFunctor {
    // compare two elements of the array
    bool operator()(const std::pair<vtkIdType, std::vector<double> >& x, 
                    const std::pair<vtkIdType, std::vector<double> >& y) {
        return (x.second[0] < y.second[0]);
    }
};

void 
vmtCellLocator::BuildLocator() {


    // attach an empty array of face Ids to each bucket
    for (int m = 0; m < numBucketsX; ++m) {
        for (int n = 0; n < numBucketsX; ++n) {
            int bucketId = m * numBucketsX + n;
            // create empty bucket
            this->bucket2Faces.insert( std::pair< int, std::vector<vtkIdType> >(bucketId, std::vector<vtkIdType>()) );
        }
    }

    // assign each face to one or more buckets depending on where the face's nodes fall
    vtkIdType numFaces = this->grid->GetNumberOfCells();
    for (vtkIdType faceId = 0; faceId < numFaces; ++faceId) {
        std::vector< Vector<double> > nodes = getFacePoints(faceId);
        for (const Vector<double>& p : nodes) {
            int bucketId = this->getBucketId(&p[0]);
            this->bucket2Faces[bucketId].push_back(faceId);
        }
    }

}

bool 
vmtCellLocator::containsPoint(vtkIdType faceId, const double point[3], double tol) const {

    tol = std::abs(tol);
    bool res = true;
    double circ = 0;


    std::vector< Vector<double> > nodes = this->getFacePoints(faceId);
    size_t npts = nodes.size();

    for (size_t i0 = 0; i0 < npts; ++i0) {

        size_t i1 = (i0 + 1) % npts;

        double* p0 = &nodes[i0][0];
        double* p1 = &nodes[i1][0];

        // vector from point to the vertices
        double d0[] = {point[0] - p0[0], point[1] - p0[1]};
        double d1[] = {point[0] - p1[0], point[1] - p1[1]};

        double cross = d0[0]*d1[1] - d0[1]*d1[0];

        if (cross < -tol) {
            // negative area
            res = false;
        }
    }

    return res;
}


vtkIdType
vmtCellLocator::FindCell(const double point[3], double tol, vtkGenericCell *cell, double pcoords[3], double *weights) {

    int bucketId = this->getBucketId(point);
    double closestPoint[3];
    int subId;
    double dist2;

    const std::vector<vtkIdType>& faces = this->bucket2Faces.find(bucketId)->second;

    for (const vtkIdType& cId : faces) {
        if (this->containsPoint(cId, point, tol)) {
        	vtkCell* cell = this->grid->GetCell(cId);
        	cell->EvaluatePosition((double*) point, closestPoint, subId, pcoords, dist2, weights);
            return cId;
        }
    }
    return -1;

}


void
vmtCellLocator::FindCellsAlongLine(const double p0[3], const double p1[3], double tol2, vtkIdList *cells) {

	cells->Reset();

    int begM, endM, begN, endN, bucketId, begBucketId, endBucketId;

    Vector<double> point0{p0[0], p0[1], p0[2]};
    Vector<double> point1{p0[0], p1[1], p1[2]};

    // choose the number of sections heuristically. Too few and we'll end up adding too many
    // cells. No point in having more sections than the number of buckets
    this->getBucketIndices(this->getBucketId(p0), &begM, &begN);
    this->getBucketIndices(this->getBucketId(p1), &endM, &endN);

    int mLo = std::min(begM, endM);
    int mHi = std::max(begM, endM);
    int nLo = std::min(begN, endN);
    int nHi = std::max(begN, endN);

    // dm and dn are positive
    int dm = mHi - mLo + 1;
    int dn = nHi - nLo + 1;

    // want more segments when the line is 45 deg. Want more segments when 
    // the points are far apart.
    size_t nSections = std::max(1, std::min(dn, dm));
    Vector<double> du = point1 - point0;
    du /= (double) nSections;

    for (size_t iSection = 0; iSection < nSections; ++iSection) {

        // start/end points of the segment
        Vector<double> pBeg = point0 + (double) iSection * du;
        Vector<double> pEnd = point0 + du;
    
        // get the start bucket
        begBucketId = this->getBucketId(&pBeg[0]);
        this->getBucketIndices(begBucketId, &begM, &begN);

        // get end bucket
        endBucketId = this->getBucketId(&pEnd[0]);
        this->getBucketIndices(endBucketId, &endM, &endN);

        mLo = std::min(begM, endM);
        mHi = std::max(begM, endM);
        nLo = std::min(begN, endN);
        nHi = std::max(begN, endN);

        // iterate over the buckets
        for (int m = mLo; m <= mHi; ++m) {
            for (int n = nLo; n <= nHi; ++n) {
                bucketId = m * numBucketsX + n;
                for (const vtkIdType& faceId : this->bucket2Faces.find(bucketId)->second) {
                	cells->InsertUniqueId(faceId);
                }
            }
        }
    }

}


std::vector< std::pair<vtkIdType, std::vector<double> > >
vmtCellLocator::findIntersectionsWithLine(const Vector<double>& pBeg, const Vector<double>& pEnd) {

    // store result
    std::vector< std::pair<vtkIdType, std::vector<double> > > res;

    // linear parameter on entry and exit of the cell
    std::pair<double, double> lamdas;

    // collect the cells intersected by the line
    vtkIdList* cells = vtkIdList::New();
    const double eps = 10 * std::numeric_limits<double>::epsilon();    
    this->FindCellsAlongLine(&pBeg[0], &pEnd[0], eps, cells);

    // iterate over the intersected cells
    for (vtkIdType i = 0; i < cells->GetNumberOfIds(); ++i) {

    	vtkIdType cellId = cells->GetId(i);

        std::vector<double> lambdas = this->collectIntersectionPoints(cellId, &pBeg[0], &pEnd[0]);

        if (lambdas.size() >= 2) {
            // found entry/exit points so add
            res.push_back( 
                std::pair<vtkIdType, std::vector<double> >(
                     cellId, std::vector<double>{lambdas[0], lambdas[lambdas.size() - 1]}
                                                       )
                         );
        }
    }
    cells->Delete();

    // sort by starting lambda
    std::sort(res.begin(), res.end(), LambdaBegFunctor());

    // to avoid double counting, shift lambda entry to be always >= to 
    // the preceding lambda exit and make sure lambda exit >= lambda entry
    for (size_t i = 1; i < res.size(); ++i) {

        double thisLambdaBeg = res[i].second[0];
        double thisLambdaEnd = res[i].second[1];
        double precedingLambdaEnd = res[i - 1].second[1];

        thisLambdaBeg = std::min(thisLambdaEnd, std::max(thisLambdaBeg, precedingLambdaEnd));

        // reset lambda entry
        res[i].second[0] = thisLambdaBeg;
    }

    return res;
}

std::vector<double>
vmtCellLocator::collectIntersectionPoints(vtkIdType cellId, 
                                          const double pBeg[3],
                                          const double pEnd[3]) {

    std::vector<double> lambdas;
    // expect two values
    lambdas.reserve(2);

    const double eps = 10 * std::numeric_limits<double>::epsilon();
    const double eps100 = 100*eps;


    std::vector< Vector<double> > nodes = this->getFacePoints(cellId);
    size_t npts = nodes.size();

    // computes the intersection point of two lines
    LineLineIntersector intersector;

    // is the starting point inside the cell?
    if (this->containsPoint(cellId, pBeg, eps)) {
        lambdas.push_back(0.);
    }

    // iterate over the cell's edges
    size_t i0, i1;
    for (i0 = 0; i0 < npts; ++i0) {

    	i1 = (i0 + 1) % npts;

        // compute the intersection point
        double* p0 = &nodes[i0][0];
        double* p1 = &nodes[i1][0];

        intersector.setPoints(&pBeg[0], &pEnd[0], &p0[0], &p1[0]);

        if (! intersector.hasSolution(eps)) {
            // no solution, skip
            continue;
        }

        // we have a solution but it could be degenerate
        if (std::abs(intersector.getDet()) > eps) {

            // normal intersection, 1 solution
            std::vector<double> sol = intersector.getSolution();
            double lambRay = sol[0];
            double lambEdg = sol[1];

            // is it valid? Intersection must be within (p0, p1) and (q0, q1)
            if (lambRay >= (0. - eps100) && lambRay <= (1. + eps100)  && 
                lambEdg >= (0. - eps100) && lambEdg <= (1. + eps100)) {
                // add to list
                lambdas.push_back(lambRay);
            }
        }
        else {
            // det is almost zero
            // looks like the two lines (p0, p1) and (q0, q1) are overlapping
            // add the starting/ending points
            const std::pair<double, double>& sol = intersector.getBegEndParamCoords();
            // add start/end linear param coord along line
            lambdas.push_back(sol.first);
            lambdas.push_back(sol.second);
        }
    }

    // is the end point inside the cell?
    if (this->containsPoint(cellId, pEnd, eps)) {
        lambdas.push_back(1.);
    }

    // order the lambdas
    std::sort(lambdas.begin(), lambdas.end());

    return lambdas;
}

