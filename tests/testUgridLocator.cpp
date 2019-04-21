#include <mntUgrid2D.h>
#include <iostream>
#undef NDEBUG // turn on asserts


void testPoint(const Vector<double>& p) {

    std::string file = "@CMAKE_SOURCE_DIR@/data/cs_4.nc";

    Ugrid2D ug;

    int ier = ug.load(file, "physics");
    assert(ier == 0);

    ug.dumpGridVtk("ug_cs_4.vtk");

    // build the locator
    ug.buildLocator(1);

    // find face
    const double tol = 1.e-10;
    size_t faceId;
    bool found = ug.findCell(p, tol, &faceId);
    if (found) {
        std::vector< Vector<double> > nodes = ug.getFacePointsRegularized(faceId);
        std::cout << "OK: point " << p << " is inside face " << faceId << " which has nodes:\n";
        for (const Vector<double>& node : nodes) {
            std::cout << node << '\n';
        }
    }
    else {
        std::cout << "point " << p << " was not found\n";
    }
    assert(found);

}

void testLine(const Vector<double>& p0, const Vector<double>& p1) {

    std::string file = "@CMAKE_SOURCE_DIR@/data/cs_4.nc";

    Ugrid2D ug;

    int ier = ug.load(file, "physics");
    assert(ier == 0);

    // build the locator
    ug.buildLocator(1);

    std::set<size_t> faceIds = ug.findCellsAlongLine(p0, p1);
    std::cout << "point " << p0 << " -> " << p1 << "overlaps with " << faceIds.size() << " cells:\n";
    for (const size_t& faceId : faceIds) {
        std::cout << faceId << ' ';
    }
    std::cout << '\n';

    // check that we found all the cells by dividing the line in 1000 segments, checking that each point 
    // is inside of the cells we found
    size_t nSegments = 1000;
    Vector<double> du = p1 - p0;
    du /= (double) nSegments;
    size_t cId;
    const double tol = 1.e-14;
    for (size_t iSegment = 0; iSegment < nSegments; ++iSegment) {
        Vector<double> p = p0 + (double) iSegment * du;
        bool found = ug.findCell(p, tol, &cId);
        if (found) {
            if (faceIds.find(cId) == faceIds.end()) {
                std::cerr << "ERROR: unable to find point " << p << " belonging to face " << cId 
                          << " and along line " << p0 << " -> " << p1 << " among the above faces/cells!\n";
                assert(false);
            }
        }
    }



}


int main() {

    Vector<double> p0(3, 0.0);

    p0[0] = 180.; p0[1] = 0.;
    testPoint(p0);

    p0[0] = 90.; p0[1] = 90.;
    testPoint(p0);

    p0[0] = 90.; p0[1] = -90.;
    testPoint(p0);

    p0[0] = 0.; p0[1] = 67.;
    testPoint(p0);

    Vector<double> p1(3, 0.0);
    p0[0] =   0.; p0[1] = -67.;
    p1[0] = 360.; p1[1] =  67.;
    testLine(p0, p1);


    return 0;
}   
