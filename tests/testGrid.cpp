#include <mntGrid.h>

void test() {
    mntGrid_t* grd;
    mnt_grid_new(&grd);
    mnt_grid_load(&grd, "${CMAKE_SOURCE_DIR}/data/cs.vtk");
    mnt_grid_del(&grd);
}


int main(int argc, char** argv) {

    test();

    return 0;
}