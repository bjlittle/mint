#include <vector>
#include <set>
#include <algorithm>
#include "MvVector.h"
#include <map>
#include <string>
#include <limits>
#include <vtkPoints.h>
#include <vtkQuad.h>
#include <vtkUnstructuredGrid.h>
#ifndef MNT_UGRID_2D
#define MNT_UGRID_2D

#define NUM_PARAM_DIMS 2
#define NUM_SPACE_DIMS 3

class Ugrid2D {

public:

/**
 * Constructor
 */
Ugrid2D() {

    this->cellPoints = vtkPoints::New();
    this->cellPoints->SetDataTypeToDouble();
    this->cellPoints->SetNumberOfPoints(4); // 2d quad

    this->cell = vtkQuad::New();
}

/**
 * Destructor
 */
~Ugrid2D() {
    this->cell->Delete();
    this->cellPoints->Delete();
}


/**
 * Get the number of faces
 * @return number
 */
size_t getNumberOfFaces() const {
    return this->numFaces;
}

/**
 * Get the number of unique edges
 * @return number
 */
size_t getNumberOfEdges() const {
    return this->numEdges;
}

/**
 * Get the number of points/vertices
 * @return number
 */
size_t getNumberOfPoints() const {
    return this->numPoints;
}

/**
 * Get pointer to the point Ids
 * @param face Id
 * @return pointer
 */
const size_t* getFacePointIds(size_t faceId) const {
    return &this->face2Points[faceId*4];
}
const std::vector<size_t>& getFacePointIds() const {
    return this->face2Points;
}

/**
 * Get pointer to the edge Ids
 * @param face Id 
 * @return pointer
 */
const size_t* getFaceEdgeIds(size_t faceId) const {
    return &this->face2Edges[faceId*4];
}
const std::vector<size_t>& getFaceEdgeIds() const {
    return this->face2Edges;
}

/**
 * Get pointer to the point Ids
 * @param edge Id 
 * @return pointer
 */
const size_t* getEdgePointIds(size_t edgeId) const {
    return &this->edge2Points[edgeId*2];
}
const std::vector<size_t>& getEdgePointIds() const {
    return this->edge2Points;
}

/**
 * Get pointer to the coordinates
 * @param pointId point id
 * @return pointer
 */
const double* getPoint(size_t pointId) const {
    return &this->points[pointId*NUM_SPACE_DIMS];
}
const std::vector<double>& getPoints() const {
    return this->points;
}

/**
 * Get the regularized points of the edge
 * @param edgeId edge Id
 * @return the two points with 360 added/substracted to minimize the edge length
 */
std::vector< Vector<double> > getEdgePointsRegularized(size_t edgeId) const;

/**
 * Get the regularized points of the face
 * @param faceId face Id
 * @return the four points with 360 added/substracted to make the cell have a positive area
 */
std::vector< Vector<double> > getFacePointsRegularized(size_t faceId) const;

/**
 * Is point inside face?
 * @param faceId face Id
 * @param point point
 * @param tol tolerance
 * @return true/false
 */
bool containsPoint(size_t faceId, const Vector<double>& point, double tol) const;

/**
 * Get min/max range of the domain
 * @param xmin low point of the domain (output)
 * @param xmax high point of the domain (output)
 */
void getRange(double xmin[], double xmax[]) const;


/**
 * Load from Ugrid file 
 * @param filename file name
 * @param meshname mesh name
 * @return error (0=OK)
 */
int load(const std::string& filename, const std::string& meshname);

/**
 * Get the face vertex coordinates
 * @return array of points
 */
std::vector< Vector<double> > getFacePoints(size_t faceId) const;

/**
 * Get the edge vertex coordinates
 * @return array of points
 */
std::vector< Vector<double> > getEdgePoints(size_t edgeId) const;

/**
 * Build 2d locator
 * @param avgNumFacesPerBucket approximate number of faces per bucket
 */
void buildLocator(int avgNumFacesPerBucket);

/**
 * Find the cell that contains a point
 * @param point target point
 * @param tol tolerance
 * @param cellId cellId (output)
 * @return true if a cell was found, false otherwise
 */
bool findCell(const Vector<double>& point, double tol, size_t* cellId) const;

/**
 * Find all the cells that are intesected by a line
 * @param point0 start point of the line
 * @param point1 end point of the line
 * @return array of cell Ids
 */
std::set<size_t> findCellsAlongLine(const Vector<double>& point0,
                                    const Vector<double>& point1) const;
/**
 * Set the cell nodes
 * @param cellId cell Id
 * @note use this prior to computing the parametric coordinates or interpolating 
 */
void setCellPoints(size_t cellId);

/**
 * Get the parametric coordinates of a point in a face/cell
 * @param point point
 * @param pcoords parametric coordinates (output)
 * @return true if the point is in the cell, false otherwise
 */
bool getParamCoords(const Vector<double>& point, double pcoords[]);

/**
 * Interpolate a point in cell
 * @param pcoords parametric coordinates
 * @param point point (output)
 */
void interpolate(const Vector<double>& pcoords, double point[]);

/**
 * Find the intersections of the grid with a line
 * @param pBeg start point of the line
 * @param pEnd end point of the line
 * @return array of [faceId, (lambda0, lambda1)] elements where lambda{0,1} are the 
 *         start/end linear parameters of the line
 */
std::vector< std::pair<size_t, std::vector<double> > >
findIntersectionsWithLine(const Vector<double>& pBeg, const Vector<double>& pEnd);

/**
 * Dump the grid to a Vtk file
 * @param filename file name
 */
void dumpGridVtk(const std::string& filename);

/**
 * Get the two triangle areas of a face
 * @param faceId face Id
 * @return area013, area231
 */
std::pair<double, double> getFaceAreas(size_t faceId);

/**
 * Get the faces that have a negative area
 * @param tol sall area tolerance. Area is negative if < tol.
 * @return list of faceIds
 */
std::vector<size_t> getNegativeFaces(double tol) const;

private:

    // vertex coordinates
    std::vector<double> points;

    // domain min/max
    Vector<double> xmin;
    Vector<double> xmax;

    size_t numPoints;
    size_t numEdges;
    size_t numFaces;

    // face to node connectivity
    std::vector<size_t> face2Points;

    // face to edge connectivity
    std::vector<size_t> face2Edges;

    // edge to node connectivity
    std::vector<size_t> edge2Points;

    // number of buckets along each direction
    int numBucketsX;

    // maps a bucket to a list of faces
    std::map<int, std::vector<size_t> > bucket2Faces;

    // for interpolation
    vtkPoints* cellPoints;
    vtkQuad* cell;

    int readConnectivityData(int ncid, int meshid, 
		                     const std::string& role,
		                     std::vector<size_t>& data);

    int readPoints(int ncid, int meshid);

    void fixPeriodicity();

    /**
     * Get the flat array index of a bucket containing a given point
     * @param point point
     * @return index
     * @note assumes there is thickness in the domain
     *       will return index ven if the point is outside the domain
     */
    inline int getBucketId(const Vector<double>& point) const {

        // required to make sure std::floor does not return the 
        // next integer below if we're close to an integer
        const double eps = 10 * std::numeric_limits<double>::epsilon();

        Vector<double> x = (point - this->xmin) / (this->xmax - this->xmin); // must have some thickness!

        // bucket coordinates
        int m = (int) std::floor( this->numBucketsX * x[0] + eps);
        int n = (int) std::floor( this->numBucketsX * x[1] + eps);

        // make sure the bucket coordinates fit in the domain
        m = std::max(0, std::min(this->numBucketsX - 1, m));
        n = std::max(0, std::min(this->numBucketsX - 1, n));

        // return flat array index
        return m * this->numBucketsX + n;
    }

    /**
     * Get the bucket index coordinates
     * @param bucketId flat bucket Id
     * @param m index (output)
     * @param n index (output)
     */
     inline void getBucketIndices(int bucketId, int* m, int* n) const {
        *m = bucketId / this->numBucketsX;
        *n = bucketId % this->numBucketsX;
    }

    /**
     * Collect the intersection points between line and cell
     * @param cellId cell Id
     * @param pBeg start point of the line
     * @param pEnd end point of the line
     * @return array of line parameter coordinates in increasing order
     * @note expect either 0 (no intersection) or 2 values (intersection) to be returned.
     *       start/end points qualify as intersection if they fall into the cell
     */
	std::vector<double> collectIntersectionPoints(size_t cellId, 
                                                  const Vector<double>& pBeg,
                                                  const Vector<double>& pEnd);

};

#endif // MNT_UGRID_2D
