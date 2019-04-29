#include <mntUgrid2D.h>
#include <mntLineLineIntersector.h>
#include <mntQuadEdgeIter.h>
#include <netcdf.h>
#include <iostream>
#include <set>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>

#define LON_INDEX 0
#define LAT_INDEX 1
#define ELV_INDEX 2

struct LambdaBegFunctor {
    // compare two elements of the array
    bool operator()(const std::pair<size_t, std::vector<double> >& x, 
                    const std::pair<size_t, std::vector<double> >& y) {
        return (x.second[0] < y.second[0]);
    }
};


std::vector< Vector<double> > 
Ugrid2D::getFacePoints(size_t faceId) const {

    const size_t* pointIds = this->getFacePointIds(faceId);
    std::vector< Vector<double> > res(4); // 2d 4 points per quad

    // iterate over the 4 points
    for (size_t i = 0; i < 4; ++i) {
        const double* p = this->getPoint(pointIds[i]);
        res[i] = Vector<double>(p, p + NUM_SPACE_DIMS);
    }

    return res;
}


std::vector< Vector<double> > 
Ugrid2D::getEdgePoints(size_t edgeId) const {

    std::vector< Vector<double> > res;

    // itereate over the 2 points spanning the edge
    for (size_t i = 0; i < 2; ++i) {

        // get the point id
        size_t pointId = this->edge2Points[i + edgeId*2];

        // get the coordinates of this point
        const double* p = this->getPoint(pointId);

        // add
        res.push_back( Vector<double>(p, p + NUM_SPACE_DIMS) );
    }
    return res;
}

bool 
Ugrid2D::containsPoint(size_t faceId, const Vector<double>& point, double tol) const {

    tol = std::abs(tol);
    bool res = true;
    double circ = 0;
    std::vector< Vector<double> > vertices = getFacePointsRegularized(faceId);
    for (size_t i0 = 0; i0 < 4; ++i0) {

        size_t i1 = (i0 + 1) % 4;

        // vector from point to the vertices
        double d0[] = {point[0] - vertices[i0][0], point[1] - vertices[i0][1]};
        double d1[] = {point[0] - vertices[i1][0], point[1] - vertices[i1][1]};

        double cross = d0[0]*d1[1] - d0[1]*d1[0];

        if (cross < -tol) {
            // negative area
            res = false;
        }
    }

    return res;
}

void
Ugrid2D::getRange(double xMin[], double xMax[]) const {
    for (size_t i = 0; i < NUM_SPACE_DIMS; ++i) {
        xMin[i] = this->xmin[i];
        xMax[i] = this->xmax[i];
    }
}

int 
Ugrid2D::load(const std::string& filename, const std::string& meshname) {

    int ier = 0;
    int ncid;

    // open the file
    ier = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    if (ier != NC_NOERR) {
        std::cerr << "ERROR: cannot open \"" << filename << "\"\n";
        return 1;
    }

    // mesh variable
    int meshid;
    ier = nc_inq_varid(ncid, meshname.c_str(), &meshid);
    if (ier != NC_NOERR) {
        std::cerr << "ERROR: cannot find variable named \"" << meshname << "\"\n";
        nc_close(ncid);
        return 2;
    }

    ier = this->readConnectivityData(ncid, meshid, 
                "face_node_connectivity", this->face2Points);
    if (ier != NC_NOERR) {
        std::cerr << "ERROR: variable \"" << meshname
        << "\" does not have attribute \"face_node_connectivity\"\n";
        nc_close(ncid);
        return 3;
    }

    ier = this->readConnectivityData(ncid, meshid, 
                "face_edge_connectivity", this->face2Edges);
    if (ier != NC_NOERR) {
        std::cerr << "ERROR: variable \"" << meshname
        << "\" does not have attribute \"face_edge_connectivity\"\n";
        nc_close(ncid);
        return 4;
    }

    ier = this->readConnectivityData(ncid, meshid, 
                "edge_node_connectivity", this->edge2Points);
    if (ier != NC_NOERR) {
        std::cerr << "ERROR: variable \"" << meshname
        << "\" does not have attribute \"edge_node_connectivity\"\n";
        nc_close(ncid);
        return 5;
    }

    // read the node coordinates
    ier = this->readPoints(ncid, meshid);
    if (ier != NC_NOERR) {
        std::cerr << "ERROR: cannot read node coordinates for mesh \"" 
        << meshname << "\"\n";
        nc_close(ncid);
        return 6;
    }

    size_t n = this->face2Points.size();
    this->numFaces = n / 4; // 4 points per face
    n = this->edge2Points.size();
    this->numEdges = n / 2; // 2 points per edge


    // compute min/max values after regularizing the coords across the faces
    // (ie adding/subtracting 360 deg for the longitude to make the face area positive)
    this->xmin.resize(NUM_SPACE_DIMS, +std::numeric_limits<double>::max());
    this->xmax.resize(NUM_SPACE_DIMS, -std::numeric_limits<double>::max());
    for (size_t faceId = 0; faceId < this->numFaces; ++faceId) {
        std::vector< Vector<double> > nodes = this->getFacePointsRegularized(faceId);
        for (const Vector<double>& p : nodes) {
            for (size_t j = 0; j < this->xmin.size(); ++j) {
                this->xmin[j] = (p[j] < this->xmin[j]? p[j]: this->xmin[j]);
                this->xmax[j] = (p[j] > this->xmax[j]? p[j]: this->xmax[j]);
            }                     
        }
    }

    return 0;
}

int
Ugrid2D::readConnectivityData(int ncid, int meshid, 
                              const std::string& role,
                              std::vector<size_t>& data) {

    int ier;

    // get the lengths of the attribute string
    size_t len;
    ier = nc_inq_attlen(ncid, meshid, role.c_str(), &len);
    if (ier != NC_NOERR) return 1;

    // read the attribute value, ie the name of the variables we will need to read
    std::string varname(len, ' ');
    ier = nc_get_att_text(ncid, meshid, role.c_str(), &varname[0]);
    if (ier != NC_NOERR) return 2;

    // fetch the variable Id for this variable name
    int varid;
    ier = nc_inq_varid(ncid, varname.c_str(), &varid);
    if (ier != NC_NOERR) return 3;

    // dimensions of the variable to read
    int dimids[2];
    size_t n0, n1;
    // either 0 or 1
    int startIndex;

    // read the data
    ier = nc_inq_vardimid(ncid, varid, dimids);
    if (ier != NC_NOERR) return 4;

    ier = nc_inq_dimlen(ncid, dimids[0], &n0);
    if (ier != NC_NOERR) return 5;

    ier = nc_inq_dimlen(ncid, dimids[1], &n1);
    if (ier != NC_NOERR) return 6;

    ier = nc_get_att_int(ncid, varid, "start_index", &startIndex);
    if (ier != NC_NOERR) return 8;

    std::vector<unsigned long long> buffer(n0 * n1);
    ier = nc_get_var_ulonglong(ncid, varid, &buffer[0]);
    if (ier != NC_NOERR) return 7;

    // subtract start_index
    data.resize(n0 * n1);
    for (size_t i = 0; i < n0 * n1; ++i) {
        data[i] = buffer[i] - startIndex;
    }

    return 0;
}

int
Ugrid2D::readPoints(int ncid, int meshid) {

    int ier;

    // get the lengths of the attribute string
    size_t len;
    ier = nc_inq_attlen(ncid, meshid, "node_coordinates", &len);
    if (ier != NC_NOERR) return 10;

    // read the attribute, lists the name of the lon and lat coordinates
    std::string val(len, ' ');
    ier = nc_get_att_text(ncid, meshid, "node_coordinates", &val[0]);
    if (ier != NC_NOERR) return 11;

    // val is "varx vary" where var{x,y} are the variable names
    size_t n = val.size();
    size_t spaceL = val.find(' ');
    size_t spaceR = val.rfind(' ');
    if (spaceL >= n) {
        // could not find space
        std::cerr << "ERROR: node_coordinates attribute \""
        << val << "\" should contain space separated list of coordinate names\n";
        return 19;
    }
    std::string varx = val.substr(0, spaceL);
    std::string vary = val.substr(spaceR + 1, n - spaceR - 1);

    int varxid, varyid;
    ier = nc_inq_varid(ncid, varx.c_str(), &varxid);
    if (ier != NC_NOERR) return 12;
    ier = nc_inq_varid(ncid, vary.c_str(), &varyid);
    if (ier != NC_NOERR) return 13;

    int dimids[1];

    this->numPoints = 0;
    this->points.resize(0);
    int varids[] = {varxid, varyid};
    for (int ivar = 0; ivar < 2; ++ivar) {

        int varid = varids[ivar];

        // get the attribute length
        ier = nc_inq_attlen(ncid, varid, "standard_name", &len);
        if (ier != NC_NOERR) {
            std::cerr << "ERROR: variable with varid = " << varid
            << " has no attribute \"standard_name\"\n";
            return 14;
        }
        std::string var_stdn(len, ' ');

        // read the attribute
        ier = nc_get_att_text(ncid, varid, "standard_name", &var_stdn[0]);
        if (ier != NC_NOERR) return 15;

        // get the dimension
        ier = nc_inq_vardimid(ncid, varid, dimids);
        if (ier != NC_NOERR) return 16;

        ier = nc_inq_dimlen(ncid, dimids[0], &this->numPoints);
        if (ier != NC_NOERR) return 17;

        // allocate/resize
        std::vector<double> data(this->numPoints);
        if (this->points.size() == 0) {
            this->points.resize(this->numPoints * NUM_SPACE_DIMS, 0.0);
        }

        // read the data
        ier = nc_get_var_double(ncid, varid, &data[0]);
        if (ier != NC_NOERR) {
            std::cerr << "ERROR: could not read \""
            << varx << "\"\n";
            return 18;
        }
        
        // associate data  our coordinate variable
        size_t j;
        if (var_stdn == "longitude") {
            j = LON_INDEX;
        }
        else if (var_stdn == "latitude") {
            j = LAT_INDEX;
        }
        else {
            std::cerr << "ERROR: unknown coordinate with standard_name \""
            << var_stdn << "\"\n";
            return 19; 
        }
        for (size_t i = 0; i < this->numPoints; ++i) {
            this->points[j + NUM_SPACE_DIMS*i] = data[i];
        }
    }

    return 0;
}

std::vector< Vector<double> > 
Ugrid2D::getFacePointsRegularized(size_t faceId) const {

    std::vector< Vector<double> > res = this->getFacePoints(faceId);

    bool allNodesInsideDomain = true;
    for (const Vector<double>& node : res) {
        allNodesInsideDomain &= node[LON_INDEX] >= 0.0;
        allNodesInsideDomain &= node[LON_INDEX] <= 360.0;
    }

    if (allNodesInsideDomain) {
        // no need to regularize if the nodes are all inside the domain
        return res;
    }

    // regularize
    for (size_t i = 1; i < res.size(); ++i) {

        // add/subtract 360 
        double dLon = res[i][LON_INDEX] - res[0][LON_INDEX];
        double dLonsPM360[] = {std::abs(dLon - 360.), 
                               std::abs(dLon       ), 
                               std::abs(dLon + 360.)};
        double* minDLon = std::min_element(&dLonsPM360[0], &dLonsPM360[3]);
        int indexMin = (int) std::distance(dLonsPM360, minDLon);
        res[i][LON_INDEX] += (indexMin - 1)*360.0;
    }

    // if at the poles, set the longitude to the average of the three other nodes
    int indexPole = -1;
    double avgLon = 0.;
    for (size_t i = 0; i < res.size(); ++i) {
        // detect if node is on/near pole
        if (std::abs(std::abs(res[i][LAT_INDEX]) -  90.) < 1.e-12) {
            indexPole = i;
        }
        else {
            avgLon += res[i][LON_INDEX];
        }
    }
    avgLon /= 3.;

    if (indexPole >= 0) {
        // one of the nodes is at the pole...

        // longitude at the pole is ill defined - we can set it to any value
        if (avgLon > 180.0) {
            res[indexPole][LON_INDEX] = 270.0;
        }
        else {
            res[indexPole][LON_INDEX] = 90.0;
        }
    }

    return res;
}

void 
Ugrid2D::buildLocator(int avgNumFacesPerBucket) {


    // number of buckets along one dimension (2D)
    this->numBucketsX = (int) std::max(1.0, 
                              std::sqrt((double) this->getNumberOfFaces() / (double) avgNumFacesPerBucket)
                                      );

    // attach an empty array of face Ids to each bucket
    for (int m = 0; m < numBucketsX; ++m) {
        for (int n = 0; n < numBucketsX; ++n) {
            int bucketId = m * numBucketsX + n;
            this->bucket2Faces.insert( std::pair< int, std::vector<size_t> >(bucketId, std::vector<size_t>()));
        }
    }

    // assign each face to one or more buckets depending on where the face's nodes fall
    for (size_t faceId = 0; faceId < this->getNumberOfFaces(); ++faceId) {
        std::vector< Vector<double> > nodes = getFacePointsRegularized(faceId);
        for (const Vector<double>& p : nodes) {
            int bucketId = this->getBucketId(p);
            this->bucket2Faces[bucketId].push_back(faceId);
        }
    }

}

bool
Ugrid2D::findCell(const Vector<double>& point, double tol, size_t* faceId) const {

    int bucketId = this->getBucketId(point);
    const std::vector<size_t>& faces = this->bucket2Faces.find(bucketId)->second;
    for (const size_t& cId : faces) {
        if (this->containsPoint(cId, point, tol)) {
            *faceId = cId;
            return true;
        }
    }
    return false;
}

std::set<size_t> 
Ugrid2D::findCellsAlongLine(const Vector<double>& point0,
                            const Vector<double>& point1) const {

    std::set<size_t> res;
    int begM, endM, begN, endN, bucketId, begBucketId, endBucketId;

    // choose the number of sections heuristically. Too few and we'll end up adding too many
    // cells. No point in having more sections than the number of buckets
    this->getBucketIndices(this->getBucketId(point0), &begM, &begN);
    this->getBucketIndices(this->getBucketId(point1), &endM, &endN);

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

    std::cerr << "<<< findCellsAlongLine points " << point0 << " -> " << point1 << " nSections = " << nSections << '\n';
    for (size_t iSection = 0; iSection < nSections; ++iSection) {

        // start/nd points of the segment
        Vector<double> p0 = point0 + (double) iSection * du;
        Vector<double> p1 = p0 + du;
    
        // get the start bucket
        begBucketId = this->getBucketId(p0);
        this->getBucketIndices(begBucketId, &begM, &begN);

        // get end bucket
        endBucketId = this->getBucketId(p1);
        this->getBucketIndices(endBucketId, &endM, &endN);

        mLo = std::min(begM, endM);
        mHi = std::max(begM, endM);
        nLo = std::min(begN, endN);
        nHi = std::max(begN, endN);

        // iterate over the buckets
        std::cerr << "<<< \nm = " << mLo << " ... " << mHi << " n = " << nLo << " ... " << nHi << '\n';
        for (int m = mLo; m <= mHi; ++m) {
            for (int n = nLo; n <= nHi; ++n) {
                bucketId = m * numBucketsX + n;
                for (const size_t& faceId : this->bucket2Faces.find(bucketId)->second) {
                    std::cerr << "<<< adding face " << faceId << "\n";
                    res.insert(faceId);
                }
            }
        }
    }

    return res;
}

void
Ugrid2D::setCellPoints(size_t cellId) {
    std::vector< Vector<double> > nodes = this->getFacePointsRegularized(cellId);
    for (size_t i = 0; i < nodes.size(); ++i) {
        this->cellPoints->SetPoint(i, &nodes[i][0]);
    }
    this->cell->Initialize(nodes.size(), this->cellPoints);
}

bool
Ugrid2D::getParamCoords(const Vector<double>& point, double pcoords[]) {
    double closestPoint[3];
    int subId;
    double dist2;
    double weights[8];
    int inside = this->cell->EvaluatePosition((double*) &point[0], closestPoint, subId, pcoords, dist2, weights);
    return (inside > 0); // 0 is outside, -1 numerical problem
}

void 
Ugrid2D::interpolate(const Vector<double>& pcoords, double point[]) {
    int subId = 0;
    double weights[8]; // not used
    this->cell->EvaluateLocation(subId, (double*) &pcoords[0], point, weights);
}

std::vector< std::pair<size_t, std::vector<double> > >
Ugrid2D::findIntersectionsWithLine(const Vector<double>& pBeg, const Vector<double>& pEnd) {

    // store result
    std::vector< std::pair<size_t, std::vector<double> > > res;

    // linear parameter on entry and exit of the cell
    std::pair<double, double> lamdas;

    // collect the cells intersected by the line
    std::set<size_t> cells = this->findCellsAlongLine(pBeg, pEnd);

    // iterate over the intersected cells
    for (const size_t& cellId : cells) {

        std::vector<double> lambdas = this->collectIntersectionPoints(cellId, pBeg, pEnd);

        if (lambdas.size() >= 2) {
            // found entry/exit points so add
            res.push_back( 
                std::pair<size_t, std::vector<double> >(
                     cellId, std::vector<double>{lambdas[0], lambdas[lambdas.size() - 1]}
                                                       )
                         );
        }
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

std::vector<double>
Ugrid2D::collectIntersectionPoints(size_t cellId, 
                                   const Vector<double>& pBeg,
                                   const Vector<double>& pEnd) {

    std::vector<double> lambdas;
    // expect two values
    lambdas.reserve(2);

    const double eps = 10 * std::numeric_limits<double>::epsilon();
    const double eps100 = 100*eps;

    // cell nodes with 360 deg added/subtracted
    std::vector< Vector<double> > nodes = this->getFacePointsRegularized(cellId);

    // computes the intersection point of two lines
    LineLineIntersector intersector;

    // is the starting point inside the cell?
    if (this->containsPoint(cellId, pBeg, eps)) {
        lambdas.push_back(0.);
    }

    // iterate over the cell's edges
    QuadEdgeIter edgeIt;
    for (int edgeIndex = 0; edgeIndex < edgeIt.getNumberOfEdges(); ++edgeIndex) {

        int j0, j1;
        edgeIt.getCellPointIds(edgeIndex, &j0, &j1);

        // compute the intersection point
        intersector.setPoints(&pBeg[0], &pEnd[0], &nodes[j0][0], &nodes[j1][0]);

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


void
Ugrid2D::dumpGridVtk(const std::string& filename) {
    std::ofstream f;
    f.open(filename);
    f << "# vtk DataFile Version 4.2\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    f << "POINTS " << 4 * this->numFaces << " double\n";
    for (size_t faceId = 0; faceId < this->numFaces; ++faceId) {
        const std::vector< Vector<double> > nodes = this->getFacePointsRegularized(faceId);
        for (const Vector<double>& node : nodes) {
            f << node << ' ';
        }
        f << '\n';
    }
    f << "CELLS " << this->numFaces << ' ' << 5 * this->numFaces << '\n'; // 2D
    for (size_t faceId = 0; faceId < this->numFaces; ++faceId) {
        // 4 points per face
        f << "4 " << faceId*4 + 0 << ' '
                  << faceId*4 + 1 << ' '
                  << faceId*4 + 2 << ' '
                  << faceId*4 + 3 << '\n';
    }
    f << "CELL_TYPES " << this->numFaces << '\n';
    for (size_t faceId = 0; faceId < this->numFaces; ++faceId) {
        f << "9 ";
        if (faceId % 10 == 0) f << '\n';
    }
    f << '\n';
    f.close();
}

std::vector<size_t> 
Ugrid2D::getNegativeFaces(double tol) const {

    std::vector<size_t> res;

    for (size_t faceId = 0; faceId < this->numFaces; ++faceId) {

        std::vector< Vector<double> > nodes = this->getFacePointsRegularized(faceId);

        bool negativeArea = false;
        for (size_t i0 = 0; i0 <= 2; i0 += 2) {
            size_t i1 = (i0 + 1) % 4; // 4 points per face
            size_t i2 = (i0 + 3) % 4; // 4 points per face
            Vector<double> d10 = nodes[i1] - nodes[i0];
            Vector<double> d20 = nodes[i2] - nodes[i0];
            double cross = d10[0]*d20[1] - d10[1]*d20[0];
            negativeArea |= (cross < tol);
        }
        if (negativeArea) {
            res.push_back(faceId);
        }
    }

    return res;
}

