#include <mntRegridEdges3d.h>
#include <mntPolysegmentIter.h>
#include <iostream>
#include <cstdio>
#include <vtkIdList.h>
#include <netcdf.h>

extern "C"
int mnt_regridedges3d_new(RegridEdges3d_t** self) {
    *self = new RegridEdges3d_t();
    (*self)->srcGrid = NULL;
    (*self)->dstGrid = NULL;
    (*self)->srcLoc = vtkCellLocator::New();
    (*self)->weights.clear();
    (*self)->numSrcCells = 0;
    (*self)->numDstCells = 0;
    (*self)->numEdgesPerCell = 4; // 2d
    (*self)->srcGridObj = NULL;
    (*self)->dstGridObj = NULL;
    return 0;
}

/**
 * Destructor
 * @return error code (0 is OK)
 */
extern "C"
int mnt_regridedges3d_del(RegridEdges3d_t** self) {
    // destroy the source and destination grids if this instance owns them
    if ((*self)->srcGridObj) {
        mnt_grid_del(&(*self)->srcGridObj);
    }
    if ((*self)->dstGridObj) {
        mnt_grid_del(&(*self)->dstGridObj);
    }
    (*self)->srcLoc->Delete();
    delete *self;
    return 0;
}

extern "C"
int mnt_regridedges3d_setSrcGrid(RegridEdges3d_t** self, vtkUnstructuredGrid* grid) {
    // check
    if (grid->GetNumberOfPoints() != 8 * grid->GetNumberOfCells()) {
        std::cerr << "mnt_regridedges_setSrcGrid: ERROR num points != 8 * num cells!\n";
        return 1;
    }

    // borrow pointer
    (*self)->srcGrid = grid;

    return 0;
}

extern "C"
int mnt_regridedges3d_setDstGrid(RegridEdges3d_t** self, vtkUnstructuredGrid* grid) {
    // check
    if (grid->GetNumberOfPoints() != 8 * grid->GetNumberOfCells()) {
        std::cerr << "mnt_regridedges_setDstGrid: ERROR num points != 8 * num cells!\n";
        return 1;
    }

    // borrow pointer
    (*self)->dstGrid = grid;
    return 0;
}

extern "C"
int mnt_regridedges3d_setSrcPointsPtr(RegridEdges3d_t** self, size_t nVertsPerCell, size_t ncells, const double verts[]) {
    mnt_grid_new(&(*self)->srcGridObj);
    mnt_grid_setPointsPtr(&(*self)->srcGridObj, (int) nVertsPerCell, (vtkIdType) ncells, verts);
    mnt_grid_get(&(*self)->srcGridObj, &(*self)->srcGrid);
    return 0;
}

extern "C"
int mnt_regridedges3d_setDstPointsPtr(RegridEdges3d_t** self, size_t nVertsPerCell, size_t ncells, const double verts[]) {
    mnt_grid_new(&(*self)->dstGridObj);
    mnt_grid_setPointsPtr(&(*self)->dstGridObj, (int) nVertsPerCell, (vtkIdType) ncells, verts);
    mnt_grid_get(&(*self)->dstGridObj, &(*self)->dstGrid);
    return 0;
}

extern "C"
int mnt_regridedges3d_build(RegridEdges3d_t** self, int numCellsPerBucket) {

    // checks
    if (!(*self)->srcGrid) {
        std::cerr << "mnt_regridedges_build: ERROR must set source grid!\n";
        return 1;
    }
    if (!(*self)->dstGrid) {
        std::cerr << "mnt_regridedges_build: ERROR must set destination grid!\n";
        return 2;
    }

    // build the locator
    (*self)->srcLoc->SetDataSet((*self)->srcGrid);
    (*self)->srcLoc->SetNumberOfCellsPerBucket(numCellsPerBucket);
    (*self)->srcLoc->BuildLocator(); 

    // compute the weights
    vtkIdList* dstPtIds = vtkIdList::New();
    vtkIdList* srcCellIds = vtkIdList::New();
    double dstEdgePt0[] = {0., 0., 0.};
    double dstEdgePt1[] = {0., 0., 0.};

    (*self)->numSrcCells = (*self)->srcGrid->GetNumberOfCells();
    (*self)->numDstCells = (*self)->dstGrid->GetNumberOfCells();

    // iterate over the dst grid cells
    for (vtkIdType dstCellId = 0; dstCellId < (*self)->numDstCells; ++dstCellId) {

        // get this cell vertex Ids
        (*self)->dstGrid->GetCellPoints(dstCellId, dstPtIds);

        // iterate over the four edges of each dst cell
        for (size_t i0 = 0; i0 < 4; ++i0) {

            size_t i1 = (i0 + 1) % 4;

            // get the start/end points of the dst edge
            vtkIdType id0 = dstPtIds->GetId(i0);
            vtkIdType id1 = dstPtIds->GetId(i1);
                
            (*self)->dstGrid->GetPoints()->GetPoint(id0, dstEdgePt0);
            (*self)->dstGrid->GetPoints()->GetPoint(id1, dstEdgePt1);

            // break the edge into sub-edges
            PolysegmentIter polySegIter = PolysegmentIter((*self)->srcGrid, 
                                                          (*self)->srcLoc,
                                                          dstEdgePt0, dstEdgePt1);

            //std::cout << "dst cell " << dstCellId
            //          << " dstEdgePt0=" << dstEdgePt0[0] << ',' << dstEdgePt0[1]
            //          << " dstEdgePt1=" << dstEdgePt1[0] << ',' << dstEdgePt1[1] << '\n';

            // number of sub-segments
            size_t numSegs = polySegIter.getNumberOfSegments();

            // iterate over the sub-segments. Each sub-segment gets a src cell Id,
            //start/end cell param coords, the coefficient...
            polySegIter.reset();
            for (size_t iseg = 0; iseg < numSegs; ++iseg) {

                const vtkIdType srcCellId = polySegIter.getCellId();
                const std::vector<double>& xia = polySegIter.getBegCellParamCoord();
                const std::vector<double>& xib = polySegIter.getEndCellParamCoord();
                const double coeff = polySegIter.getCoefficient();

                std::vector<double> dxi({xib[0] - xia[0], xib[1] - xia[1]});
                std::vector<double> xiMid({0.5*(xia[0] + xib[0]), 0.5*(xia[1] + xib[1])});

                std::pair<vtkIdType, vtkIdType> k = std::pair<vtkIdType, vtkIdType>(dstCellId, 
                                                                                    srcCellId);

                // compute the weight contributions from each src edge
                double ws[] = {+ dxi[0] * (1.0 - xiMid[1]) * coeff,
                               + dxi[1] * (0.0 + xiMid[0]) * coeff,
                               - dxi[0] * (0.0 + xiMid[1]) * coeff,
                               - dxi[1] * (1.0 - xiMid[0]) * coeff};

                //std::cout << "\t seg=" << iseg << " srcCellId=" << srcCellId << " xia=" << xia[0] << ',' << xia[1] << " xib=" << xib[0] << ',' << xib[1]
                //          << " coeff=" << coeff << " dxi=" << dxi[0] << ',' << dxi[1] << '\n';

                if ((*self)->weights.find(k) == (*self)->weights.end()) {
                    // initialize the weights to a zero 4x4 matrix
                    std::vector<double> zeros(4, 0.0);
                    std::pair< std::pair<vtkIdType, vtkIdType>, std::vector<double> > kv 
                      = std::pair< std::pair<vtkIdType, vtkIdType>, std::vector<double> >(k, zeros);
                    (*self)->weights.insert(kv);
                }

                std::map< std::pair<vtkIdType, vtkIdType>, std::vector<double> >::iterator 
                     it = (*self)->weights.find(k);

                // add the weights
                for (size_t j = 0; j < it->second.size(); ++j) {
                    it->second[j] += ws[j];
                }

                // next segment
                polySegIter.next();

            }

            double totalT = polySegIter.getIntegratedParamCoord();
            if (std::abs(totalT - 1.0) > 1.e-10) {
            	printf("Warning: total t of segment: %lf != 1 (diff=%lg) dst cell %lld points (%18.16lf, %18.16lf), (%18.16lf, %18.16lf)\n",
            		   totalT, totalT - 1.0, dstCellId, dstEdgePt0[0], dstEdgePt0[1], dstEdgePt1[0], dstEdgePt1[1]);
            }

        }
    }

    // clean up
    srcCellIds->Delete();
    dstPtIds->Delete();

    return 0;
}

extern "C"
int mnt_regridedges3d_getNumSrcCells(RegridEdges3d_t** self, int* n) {
    *n = (*self)->numSrcCells;
    return 0;
}

extern "C"
int mnt_regridedges3d_getNumDstCells(RegridEdges3d_t** self, int* n) {
    *n = (*self)->numDstCells;
    return 0;
}

extern "C"
int mnt_regridedges3d_getNumEdgesPerCell(RegridEdges3d_t** self, int* n) {
    *n = (*self)->numEdgesPerCell;
    return 0;
}


extern "C"
int mnt_regridedges3d_applyWeights(RegridEdges3d_t** self, const double src_data[], double dst_data[]) {

    // initialize the data to zero
    size_t numEdges = (*self)->numDstCells * (*self)->numEdgesPerCell;
    for (size_t i = 0; i < numEdges; ++i) {
        dst_data[i] = 0.0;
    }

    for (std::map< std::pair<vtkIdType, vtkIdType>, std::vector<double> >::const_iterator 
         it = (*self)->weights.begin(); it != (*self)->weights.end(); ++it) {

        vtkIdType dstCellId = it->first.first;
        vtkIdType srcCellId = it->first.second;
        const std::vector<double>& weights = it->second;

        size_t kd = dstCellId * (*self)->numEdgesPerCell;
        size_t ks = srcCellId * (*self)->numEdgesPerCell;

        for (size_t ie = 0; ie < (*self)->numEdgesPerCell; ++ie) {
            dst_data[kd + ie] += weights[ie] * src_data[ks + ie];
        }
    }

    return 0;
}

extern "C"
int mnt_regridedges3d_load(RegridEdges3d_t** self, const char* filename) {

    int ncid, ier;
    ier = nc_open(filename, NC_NOWRITE, &ncid);

    // get the sizes
    size_t numWeights, numEdgesPerCell;
    int numWeightsId;
    int numEdgesId;
    ier = nc_inq_dimid(ncid, "num_weights", &numWeightsId);
    ier = nc_inq_dimlen(ncid, numWeightsId, &numWeights);
    ier = nc_inq_dimid(ncid, "num_edges_per_cell", &numEdgesId);
    ier = nc_inq_dimlen(ncid, numEdgesId, &numEdgesPerCell);

    // should check that numEdgesPerCell and (*self)->numEdgesPerCell match

    int dstCellIdsId, srcCellIdsId, weightsId;

    ier = nc_inq_varid(ncid, "dst_cell_ids", &dstCellIdsId);
    ier = nc_inq_varid(ncid, "src_cell_ids", &srcCellIdsId);
    ier = nc_inq_varid(ncid, "weights", &weightsId);

    // read
    std::vector<long> dstCellIds(numWeights);
    std::vector<long> srcCellIds(numWeights);
    std::vector<double> weights(numWeights * numEdgesPerCell);
    ier = nc_get_var_long(ncid, dstCellIdsId, &dstCellIds[0]);
    ier = nc_get_var_long(ncid, srcCellIdsId, &srcCellIds[0]);
    ier = nc_get_var_double(ncid, weightsId, &weights[0]);

    ier = nc_close(ncid);    

    // store in map
    for (int i = 0; i < numWeights; ++i) {
        std::pair< vtkIdType, vtkIdType > k(dstCellIds[i], srcCellIds[i]);

        // create vector of weights for this cell
        std::vector<double> v(&weights[numEdgesPerCell*i], &weights[numEdgesPerCell*i + numEdgesPerCell]);

        // create pair of entries
        std::pair< std::pair<vtkIdType, vtkIdType>, std::vector<double> > kv(k, v);

        // insert
        (*self)->weights.insert(kv);
    }

    return 0;
}

extern "C"
int mnt_regridedges3d_dump(RegridEdges3d_t** self, const char* filename) {

    size_t numWeights = (*self)->weights.size();

    int ncid, ier;
    ier = nc_create(filename, NC_CLOBBER, &ncid);

    // create dimensions
    int numWeightsId;
    int numEdgesId;
    ier = nc_def_dim(ncid, "num_weights", (int) numWeights, &numWeightsId);
    ier = nc_def_dim(ncid, "num_edges_per_cell", (*self)->numEdgesPerCell, &numEdgesId);


    // create variables
    int numWeightsAxis[] = {numWeightsId};
    int numWeightsNumEdgesAxes[] = {numWeightsId, numEdgesId};

    int dstCellIdsId;
    ier = nc_def_var(ncid, "dst_cell_ids", NC_LONG, 1, numWeightsAxis, &dstCellIdsId);

    int srcCellIdsId;
    ier = nc_def_var(ncid, "src_cell_ids", NC_LONG, 1, numWeightsAxis, &srcCellIdsId);

    int weightsId;
    ier = nc_def_var(ncid, "weights", NC_DOUBLE, 1, numWeightsNumEdgesAxes, &weightsId);

    // close define mode
    ier = nc_enddef(ncid);

    // load into arrays
    std::vector<long> dstCellIds(numWeights);
    std::vector<long> srcCellIds(numWeights);
    std::vector<double> weights(numWeights * (*self)->numEdgesPerCell);
    size_t i = 0;
    for (std::map< std::pair<vtkIdType, vtkIdType>, std::vector<double> >::const_iterator 
        it = (*self)->weights.begin(); it != (*self)->weights.end(); ++it) {
        dstCellIds[i] = it->first.first;
        srcCellIds[i] = it->first.second;
        for (int j = 0; j < (*self)->numEdgesPerCell; ++j) {
        weights[(*self)->numEdgesPerCell*i + j] = it->second[j];
        }
        i++;
    }

    // write
    ier = nc_put_var_long(ncid, dstCellIdsId, &dstCellIds[0]);
    ier = nc_put_var_long(ncid, srcCellIdsId, &srcCellIds[0]);
    ier = nc_put_var_double(ncid, weightsId, &weights[0]);

    ier = nc_close(ncid);

    return 0;
}

