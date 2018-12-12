#include <mntLineTriangleIntersector.h>
#undef NDEBUG // turn on asserts
#include <cassert>
#include <cmath>
#include <iostream>

#define TOL 1.e-10

void testGetParamLocation1() {

    Vector<double> q0(3); q0[0] = 0.; q0[1] = 0.; q0[2] = 0.;
    Vector<double> q1(3); q1[0] = 1.; q1[1] = 0.; q1[2] = 0.;
    Vector<double> q2(3); q2[0] = 0.; q2[1] = 1.; q2[2] = 0.;

    Vector<double> p(3); p[0] = 0.; p[1] = 0.1; p[2] = 0.;

	Vector<double> xi = getTriangleParamLocation(q0, q1, q2, p);
    std::cout << "testGetParamLocation1: xi = " << xi[0] << ", " << xi[1] << '\n';
    assert(std::abs(xi[0] - 0.) < TOL);
    assert(std::abs(xi[1] - 0.1) < TOL);
}

void testNormal() {

    // line
    double p0[] = {0., 0., 1.};
    double p1[] = {0., 0., 0.};

    // triangle
    double q0[] = {0., 0., 0.};
    double q1[] = {1., 0., 0.};
    double q2[] = {0., 1., 0.};
    LineTriangleIntersector lti;
    lti.setPoints(p0, p1, q0, q1, q2);

    double det = lti.getDet();
    assert(std::abs(det - (-1.)) < TOL);
    assert(lti.hasSolution(TOL));
    Vector<double> sol = lti.getSolution();
    double lam = sol[0];
    assert(std::abs(lam - 1.0) < TOL);
    std::vector<double> xi = std::vector<double>(&sol[1], &sol[3]);
    assert(std::abs(xi[0] - 0.) < TOL);
    assert(std::abs(xi[1] - 0.) < TOL);
}

void testNoSolution() {

    // line
    double p0[] = {0., 0., 0.1};
    double p1[] = {0., 1., 0.1};

    // triangle
    double q0[] = {0., 0., 0.};
    double q1[] = {1., 0., 0.};
    double q2[] = {0., 1., 0.};
    LineTriangleIntersector lti;
    lti.setPoints(p0, p1, q0, q1, q2);

    double det = lti.getDet();
    assert(std::abs(det - (0.)) < TOL);
    assert(!lti.hasSolution(TOL));
}

void testDegenerate() {

    // line
    double p0[] = {0., 0.1, 0.};
    double p1[] = {1., 0.1, 0.};

    // triangle
    double q0[] = {0., 0., 0.};
    double q1[] = {1., 0., 0.};
    double q2[] = {0., 1., 0.};
    LineTriangleIntersector lti;
    lti.setPoints(p0, p1, q0, q1, q2);

    double det = lti.getDet();
    assert(std::abs(det - (0.)) < TOL);
    assert(lti.hasSolution(TOL));
    const std::pair< double, double > lamBegEnd = lti.getBegEndParamCoords();
    assert(std::abs(lamBegEnd.first - 0.0) < TOL);
    assert(std::abs(lamBegEnd.second - 0.9) < TOL);    
}


int main(int argc, char** argv) {

    testGetParamLocation1();
    testNormal();
    testNoSolution();
    testDegenerate();

    return 0;
}
