#include <vmtCellLocator.h>
#include <vtkQuad.h>
#include <vtkPoints.h>
#include <vtkCell.h>
#include <mntLineLineIntersector.h>

struct LambdaBegFunctor {
    // compare two elements of the array
    bool operator()(const std::pair<vtkIdType, Vec2>& x, 
                    const std::pair<vtkIdType, Vec2>& y) {
        return (x.second[0] < y.second[0]);
    }
};

void 
vmtCellLocator::BuildLocator() {

    std::set<vtkIdType> empty;
    // attach an empty set of face Ids to each bucket
    for (int m = 0; m < this->numBucketsX; ++m) {
        for (int n = 0; n < this->numBucketsX; ++n) {
            int bucketId = m * this->numBucketsX + n;
            // create empty bucket
            this->bucket2Faces.insert( std::pair< int, std::set<vtkIdType> >(bucketId, empty) );
        }
    }

    // assign each face to one or more buckets depending on where the face's nodes fall
    vtkIdType numFaces = this->grid->GetNumberOfCells();
    for (vtkIdType faceId = 0; faceId < numFaces; ++faceId) {
        std::vector<Vec3> nodes = getFacePoints(faceId);
        for (const Vec3& p : nodes) {
            // assumes that the buckets are always bigger than the faces, or alternatively
            // that no corners are fully inside a face
            int bucketId = this->getBucketId(&p[0]);
            this->bucket2Faces[bucketId].insert(faceId);
        }
    }

}

bool 
vmtCellLocator::containsPoint(vtkIdType faceId, const double point[3], double tol) const {

    tol = std::abs(tol);
    bool res = true;

    std::vector<Vec3> nodes = this->getFacePoints(faceId);
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
vmtCellLocator::FindCell(const double point[3], double tol, vtkGenericCell *notUsed, double pcoords[3], double *weights) {

    int bucketId = this->getBucketId(point);
    double closestPoint[3];
    int subId;
    double dist2;

    const std::set<vtkIdType>& faces = this->bucket2Faces.find(bucketId)->second;

    for (const vtkIdType& cId : faces) {
        if (this->containsPoint(cId, point, tol)) {
            vtkCell* quad = this->grid->GetCell(cId);
            int inside = quad->EvaluatePosition((double*) point, closestPoint, subId, pcoords, dist2, weights);
            return cId;
        }
    }
    return -1;

}


void
vmtCellLocator::FindCellsAlongLine(const double p0[3], const double p1[3], double tol2, vtkIdList *cellIds) {

    cellIds->Reset();

    int begM, endM, begN, endN, bucketId, begBucketId, endBucketId;

    Vec3 point0(p0);
    Vec3 point1(p1);

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

    // break the line into segments, the number of segments should not affect the result. Only used as a performance 
    // improvement. Want more segments when the line is 45 deg. Want more segments when the points are far apart. 
    size_t nSections = std::max(1, std::min(dn, dm));
    Vec3 du = point1 - point0;
    du /= (double) nSections;

    for (size_t iSection = 0; iSection < nSections; ++iSection) {

        // start/end points of the segment
        Vec3 pBeg = point0 + (double) iSection * du;
        Vec3 pEnd = pBeg + du;
    
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
                    cellIds->InsertUniqueId(faceId);
                }
            }
        }
    }

}


std::vector< std::pair<vtkIdType, Vec2> >
vmtCellLocator::findIntersectionsWithLine(const Vec3& pBeg, const Vec3& pEnd, double xPeriodicity) {

    // direction of the line, does not change after adding/subtracting periodicity
    Vec3 dp = pEnd - pBeg;

    // find all the intersections of the lime with the grid boundary
    std::vector<double> boundLambdas = this->findLineGridBoundaryIntersections(pBeg, pEnd);

    double lambdaInOut[2];

    // store result
    std::vector< std::pair<vtkIdType, Vec2> > res;

    const double eps = 10 * std::numeric_limits<double>::epsilon();
    const double eps100 = 100*eps;

    size_t numSegs = boundLambdas.size() - 1; // 2 or more values
    for (size_t iSeg = 0; iSeg < numSegs; ++iSeg) {

        // slightly adjust the points to fall inside the segment
        Vec3 p0 = pBeg + (boundLambdas[iSeg + 0] + eps100)*dp;
        Vec3 p1 = pBeg + (boundLambdas[iSeg + 1] - eps100)*dp;

        if (xPeriodicity > 0.) {
            // add/subtract a periodicity length if need be
            this->makePeriodic(p0, xPeriodicity);
            this->makePeriodic(p1, xPeriodicity);
        }

        // collect the cells intersected by the line
        vtkIdList* cells = vtkIdList::New();
        this->FindCellsAlongLine(&p0[0], &p1[0], eps, cells);

        // iterate over the intersected cells
        for (vtkIdType i = 0; i < cells->GetNumberOfIds(); ++i) {

            vtkIdType cellId = cells->GetId(i);

            std::vector<double> lambdas = this->collectIntersectionPoints(cellId, &p0[0], &p1[0]);

            if (lambdas.size() >= 2) {

                lambdaInOut[0] = lambdas[0];
                lambdaInOut[1] = lambdas[lambdas.size() - 1];

                // found entry/exit points so add
                res.push_back(  std::pair<vtkIdType, Vec2>( cellId, Vec2(lambdaInOut) )  );
            }
        }
        cells->Delete();
    }

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

void
vmtCellLocator::makePeriodic(Vec3& v, double xPeriodicity) {

    // fix start/end points if they fall outside the domain and the domain is periodic
    if (xPeriodicity > 0.) {
        double xmin = this->grid->GetBounds()[0];
        double xmax = this->grid->GetBounds()[1];
        if (v[0] < xmin) {
            std::cerr << "Warning: adding periodicity length " << xPeriodicity << 
                         " to point " << v << "\n";
            v[0] += xPeriodicity;
        }
        else if (v[0] > xmax) {
            std::cerr << "Warning: subtracting periodicity length " << xPeriodicity << 
                         " from point " << v << "\n";
            v[0] -= xPeriodicity;
        }
    }
}

std::vector<double>
vmtCellLocator::findLineGridBoundaryIntersections(const Vec3& pBeg, const Vec3& pEnd) {

    const double eps = 10 * std::numeric_limits<double>::epsilon();
    const double eps100 = 100*eps;

    std::vector<double> lambdas{0.};

    // get the grid boundary
    double* box = this->grid->GetBounds();

    // box vertices
    double xmin = box[0];
    double xmax = box[1];
    double ymin = box[2];
    double ymax = box[3];
    double v0[] = {xmin, ymin, 0.};
    double v1[] = {xmax, ymin, 0.};
    double v2[] = {xmax, ymax, 0.};
    double v3[] = {xmin, ymax, 0.};

    std::vector<double*> nodes{v0, v1, v2, v3};
    const size_t npts = nodes.size();

    // computes the intersection point of two lines
    LineLineIntersector intersector;

    // iterate over the cell's edges
    size_t i0, i1;
    for (i0 = 0; i0 < npts; ++i0) {

        i1 = (i0 + 1) % npts;

        // compute the intersection point
        double* p0 = &nodes[i0][0];
        double* p1 = &nodes[i1][0];

        intersector.setPoints(&pBeg[0], &pEnd[0], p0, p1);

        if (! intersector.hasSolution(eps)) {
            // no solution, skip
            continue;
        }

        // we have a solution but it could be degenerate
        if (std::abs(intersector.getDet()) > eps) {

            // normal intersection, 1 solution
            Vec2 sol = intersector.getSolution();
            double lambRay = sol[0];
            double lambEdg = sol[1];

            // is it valid? Intersection must be within (p0, p1) and (q0, q1)
            if (lambRay >= (0. - eps100) && lambRay <= (1. + eps100)  && 
                lambEdg >= (0. - eps100) && lambEdg <= (1. + eps100)) {
                // add to list
                if (lambRay > lambdas[lambdas.size() - 1] + eps100) {
                    lambdas.push_back(lambRay);
                }
            }
        }
        // no need to worry about the degenerate case
    }

    if (lambdas[lambdas.size() - 1] < 1.0 - eps100) {
        // only add if not already in the array
        lambdas.push_back(1.0);
    }

    // order the lambdas
    std::sort(lambdas.begin(), lambdas.end());

    return lambdas;
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


    std::vector<Vec3> nodes = this->getFacePoints(cellId);
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
            Vec2 sol = intersector.getSolution();
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


