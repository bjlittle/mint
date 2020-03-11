#include <mntRegridEdges.h>
#include <mntNcAttributes.h>
#include <mntNcFieldWrite.h>
#include <mntGrid.h>
#include <CmdLineArgParser.h>
#include <vtkUnstructuredGrid.h>
#include <vtkAbstractArray.h>
#include <vtkCellData.h>
#include <iostream>
#include <limits>
#include <cmath>
#include <netcdf.h>

int main(int argc, char** argv) {

    int ier;
    CmdLineArgParser args;
    args.setPurpose("Regrid an edge centred field.");
    args.set("-s", std::string(""), "UGRID source grid file and mesh name, specified as \"filename:meshname\"");
    args.set("-v", std::string(""), "Specify edge staggered field variable name in source UGRID file, varname[@filename:meshname]");
    args.set("-d", std::string(""), "UGRID destination grid file name");
    args.set("-w", std::string(""), "Write interpolation weights to file");
    args.set("-W", std::string(""), "Load interpolation weights from file");
    args.set("-o", std::string(""), "Specify output VTK file where regridded edge data are saved");
    args.set("-O", std::string(""), "Specify output 2D UGRID file where regridded edge data are saved");
    args.set("-S", 1, "Set to zero if you want to disable source grid regularization. This might be required for uniform lon-lat grids");
    args.set("-D", 1, "Set to zero if you want to disable destination grid regularization. This might be required for uniform lon-lat grids");
    args.set("-N", 10, "Average number of cells per bucket");

    bool success = args.parse(argc, argv);
    bool help = args.get<bool>("-h");

    if (success && !help) {
        std::string srcFile = args.get<std::string>("-s");
        std::string dstFile = args.get<std::string>("-d");
        std::string weightsFile = args.get<std::string>("-w");
        std::string loadWeightsFile = args.get<std::string>("-W");
        std::string vtkOutputFile = args.get<std::string>("-o");
        std::string dstEdgeDataFile = args.get<std::string>("-O");

        if (srcFile.size() == 0) {
            std::cerr << "ERROR: must specify a source grid file (-s)\n";
            return 1;
        }
        if (dstFile.size() == 0) {
            std::cerr << "ERROR: must specify a destination grid file (-d)\n";
            return 2;
        }

        RegridEdges_t* rg;
        mnt_regridedges_new(&rg);

        // defaults are suitable for cubed-sphere 
        int fixLonAcrossDateline = 1;
        int averageLonAtPole = 1;
        if (args.get<int>("-S") == 0) {
            fixLonAcrossDateline = 0;
            averageLonAtPole = 0;
            std::cout << "info: no regularization applied to source grid\n";
        }
        ier = mnt_regridedges_setSrcGridFlags(&rg, fixLonAcrossDateline, averageLonAtPole);

        // ...destination griod
        fixLonAcrossDateline = 1;
        averageLonAtPole = 1;
        if (args.get<int>("-D") == 0) {
            fixLonAcrossDateline = 0;
            averageLonAtPole = 0;
            std::cout << "info: no regularization applied to destination grid\n";
        }
        ier = mnt_regridedges_setDstGridFlags(&rg, fixLonAcrossDateline, averageLonAtPole);

        // read the source grid
        ier = mnt_regridedges_loadSrcGrid(&rg, srcFile.c_str(), srcFile.size());
        if (ier != 0) {
            std::cerr << "ERROR: could not read file \"" << srcFile << "\"\n";
            return 3;
        }

        // read the destination grid
        ier = mnt_regridedges_loadDstGrid(&rg, dstFile.c_str(), dstFile.size());
        if (ier != 0) {
            std::cerr << "ERROR: could not read file \"" << dstFile << "\"\n";
            return 4;
        }

        if (loadWeightsFile.size() == 0) {
            ier = mnt_regridedges_build(&rg, args.get<int>("-N"));
            if (ier != 0) return 5;
        
            if (weightsFile.size() != 0) {
                std::cout << "info: saving weights in file " << weightsFile << '\n';
                ier = mnt_regridedges_dumpWeights(&rg, weightsFile.c_str(), (int) weightsFile.size());
            }

        }
        else {
            std::cout << "info: loading weights from file " << loadWeightsFile << '\n';
            ier = mnt_regridedges_loadWeights(&rg, loadWeightsFile.c_str(), (int) loadWeightsFile.size());
            if (ier != 0) return 6;
        }

        // regrid
        size_t numSrcEdges, numDstEdges;
        mnt_regridedges_getNumSrcEdges(&rg, &numSrcEdges);
        mnt_regridedges_getNumDstEdges(&rg, &numDstEdges);
        std::vector<double> srcEdgeData(numSrcEdges);
        std::vector<double> dstEdgeData(numDstEdges);

        std::string varAtFileMesh = args.get<std::string>("-v");
        if (varAtFileMesh.size() > 0) {

            // by default the variable is stored in srcFile
            std::string fileAndMeshName = srcFile;
            std::string vname = varAtFileMesh;

            size_t posAt = varAtFileMesh.find('@');
            if (posAt < std::string::npos) {
                // user specified the file and mesh names
                fileAndMeshName = varAtFileMesh.substr(posAt + 1);
                vname = varAtFileMesh.substr(0, posAt);
            }

            std::string srcFileName = fileAndMeshName;
            size_t posColumn = fileAndMeshName.find(':');
            if (posColumn < std::string::npos) {
                srcFileName = fileAndMeshName.substr(0, posColumn);
            }

            NcAttributes_t* attrs = NULL;
            ier = mnt_ncattributes_new(&attrs);
            int ncid;
            ier = nc_open(srcFileName.c_str(), NC_NOWRITE, &ncid);
            int varid;
            ier = nc_inq_varid(ncid, vname.c_str(), &varid);
            // read the attributes
            ier = mnt_ncattributes_read(&attrs, ncid, varid);
            ier = nc_close(ncid);

            std::cout << "info: loading field " << vname << " from file \"" << fileAndMeshName << "\"\n";
            ier = mnt_regridedges_loadEdgeField(&rg, 
                                                fileAndMeshName.c_str(), fileAndMeshName.size(),
                                                vname.c_str(), vname.size(),
                                                numSrcEdges, &srcEdgeData[0]);
            if (ier != 0) {
                std::cerr << "ERROR: could not load edge centred data \"" << vname << "\" from file \"" << fileAndMeshName << "\"\n";
                return 6;
            }

            // regrid
            ier = mnt_regridedges_apply(&rg, &srcEdgeData[0], &dstEdgeData[0]);
            if (ier != 0) {
                std::cerr << "ERROR: failed to apply weights to dst field \"" << vname << "\"\n";
                return 6;
            }

            // compute loop integrals for each cell
            size_t numDstCells, dstEdgeId;
            int dstEdgeSign;
            mnt_regridedges_getNumDstCells(&rg, &numDstCells);
            int numEdgesPerCell;
            mnt_regridedges_getNumEdgesPerCell(&rg, &numEdgesPerCell);
            std::vector<double> loop_integrals(numDstCells);
            double minAbsLoop = std::numeric_limits<double>::max();
            double maxAbsLoop = - std::numeric_limits<double>::max();
            double avgAbsLoop = 0.0;
            for (size_t dstCellId = 0; dstCellId < numDstCells; ++dstCellId) {
                double loop = 0.0;
                for (int ie = 0; ie < numEdgesPerCell; ++ie) {

                    ier = mnt_grid_getEdgeId(&rg->dstGridObj, dstCellId, ie, &dstEdgeId, &dstEdgeSign);
                    assert(ier == 0);

                    // +1 for ie = 0, 1; -1 for ie = 2, 3
                    int sgn = 1 - 2*(ie/2);
                    loop += sgn * dstEdgeSign * dstEdgeData[dstEdgeId];
                }

                loop_integrals[dstCellId] = loop;
                loop = std::abs(loop);
                minAbsLoop = std::min(loop, minAbsLoop);
                maxAbsLoop = std::max(loop, maxAbsLoop);
                avgAbsLoop += loop;
            }
            avgAbsLoop /= double(numDstCells);
            std::cout << "Min/avg/max cell loop integrals: " << minAbsLoop << "/" << avgAbsLoop << "/" << maxAbsLoop << '\n';

            if (vtkOutputFile.size() > 0) {

                // cell by cell data
                std::vector<double> dstCellByCellData(numDstCells * numEdgesPerCell);
                for (size_t dstCellId = 0; dstCellId < numDstCells; ++dstCellId) {
                    for (int ie = 0; ie < 4; ++ie) {
                        ier = mnt_grid_getEdgeId(&rg->dstGridObj, dstCellId, ie, &dstEdgeId, &dstEdgeSign);
                        size_t k = dstCellId*numEdgesPerCell + ie;
                        dstCellByCellData[k] = dstEdgeData[dstEdgeId] * dstEdgeSign;
                    }
                }

                // attach field to grid so we can save the data in file
                mnt_grid_attach(&rg->dstGridObj, vname.c_str(), numEdgesPerCell, &dstCellByCellData[0]);

                std::string loop_integral_varname = std::string("loop_integrals_of_") + vname;
                mnt_grid_attach(&rg->dstGridObj, loop_integral_varname.c_str(), 1, &loop_integrals[0]);

                std::cout << "info: writing \"" << vname << "\" to " << vtkOutputFile << '\n';
                mnt_grid_dump(&rg->dstGridObj, vtkOutputFile.c_str());
            }

            if (dstEdgeDataFile.size() > 0) {

                /*
                std::cout << "info: writing \"" << vname << "\" to " << dstEdgeDataFile << '\n';
                mnt_regridedges_dumpEdgeField(&rg, dstEdgeDataFile.c_str(), dstEdgeDataFile.size(), 
                                               vname.c_str(), vname.size(), 
                                               numDstEdges, &dstEdgeData[0]);
                                               */

                size_t columnL = dstEdgeDataFile.find(':');

                // get the file name
                std::string filename = dstEdgeDataFile.substr(0, columnL);
                // get the mesh name
                std::string meshname = dstEdgeDataFile.substr(columnL + 1);

                int ier;
                NcFieldWrite_t* wr = NULL;

                int n1 = filename.size();
                int n2 = vname.size();
                const int append = 0; // new file
                ier = mnt_ncfieldwrite_new(&wr, filename.c_str(), n1, vname.c_str(), n2, append);
                if (ier != 0) {
                    std::cerr << "ERROR: create file " << filename << " with field " 
                              << vname << " in append mode " << append << '\n';
                    return 1;
                }

                ier = mnt_ncfieldwrite_setNumDims(&wr, 1); // 1D array only in this implementation
                if (ier != 0) {
                    std::cerr << "ERROR: cannot set the number of dimensions for field " << vname << " in file " << filename << '\n';
                    ier = mnt_ncfieldwrite_del(&wr);
                    return 2;
                }

                // add num_edges axis
                std::string axname = "num_edges";
                int n3 = axname.size();
                ier = mnt_ncfieldwrite_setDim(&wr, 0, axname.c_str(), n3, numDstEdges);
                if (ier != 0) {
                    std::cerr << "ERROR: setting dimension 0 (" << axname << ") to " << numDstEdges
                              << " for field " << vname << " in file " << filename << '\n';
                    ier = mnt_ncfieldwrite_del(&wr);
                    return 3;
                }

                // add the attributes
                ier = mnt_ncattributes_write(&attrs, wr->ncid, wr->varid);
                if (ier != 0) {
                    std::cerr << "ERROR: writing attributes for field " << vname << " in file " << filename << '\n';
                    ier = mnt_ncfieldwrite_del(&wr);
                    return 3;
                }


                // write the data to disk
                ier = mnt_ncfieldwrite_data(&wr, &dstEdgeData[0]);
                if (ier != 0) {
                    std::cerr << "ERROR: writing data for field " << vname << " in file " << filename << '\n';
                    ier = mnt_ncfieldwrite_del(&wr);
                    return 5;
                }

                // clean up
                ier = mnt_ncfieldwrite_del(&wr);
                ier = mnt_ncattributes_del(&attrs);


            }
        }

        // cleanup
        mnt_regridedges_del(&rg);

    }
    else if (help) {
        args.help();
    }
    else {
        std::cerr << "ERROR when parsing command line arguments\n";
    }

    return 0;
}
