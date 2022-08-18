static const char help[] = "Testing particle access to face geometry in parallel";

#include <petscdmplex.h>
#include <petscdmswarm.h>

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 2;
    PetscInt faces[2] = {6, 6};
    PetscReal lower[2] = {0, 0};
    PetscReal upper[2] = {1, 1};
    DMBoundaryType bc[2] = {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    Vec faceGeomVec = NULL;  //!< Vector used to describe the entire face geom of the dm.  This is constant and does not depend upon region.
    Vec cellGeomVec = NULL;
    DM faceDM;
    const PetscScalar *faceGeomArray;
    PetscFVFaceGeom *faceGeom;

    PetscReal minCellRadius;
    DMPlexGetMinRadius(dm,
                       &minCellRadius);                    //!< The minimum cell radius is used to scale the face stepping procedures
    DMPlexComputeGeometryFVM(dm, &cellGeomVec, &faceGeomVec);  //!< Get the geometry vectors
    VecGetDM(faceGeomVec, &faceDM);
    VecGetArrayRead(faceGeomVec, &faceGeomArray);

    /** Get MPI information */
    PetscMPIInt rank;
    PetscMPIInt numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &rank);      //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);  //!< Get the number of ranks in the simulation.

    /** Make a struct to store the range of cells in the domain */
    struct Range {
        IS is;
        PetscInt start;
        PetscInt end;
        const PetscInt *points;
    } cellRange;

    /** check to see if there is a ghost label */
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel);

    /** Get the cell range associated with this domain so that it can be iterated through */
    PetscInt depth;
    DMPlexGetDepth(dm, &depth);
    // Start out getting all the points
    IS allPointIS;
    DMGetStratumIS(dm, "dim", depth, &allPointIS);
    if (!allPointIS) {
        DMGetStratumIS(dm, "depth", depth, &allPointIS);
    }

    // Get the point range
    if (cellRange.is == NULL) {
        // There are no points in this region, so skip
        cellRange.start = 0;
        cellRange.end = 0;
        cellRange.points = NULL;
    } else {
        // Get the range
        ISGetPointRange(cellRange.is, &cellRange.start, &cellRange.end, &cellRange.points);
        // Clean up the allCellIS
        ISDestroy(&allPointIS);
    }

    /** Iterate through the cells in the domain and show that there are faces with an invalid geometry */
    for (PetscInt c = cellRange.start;
         c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscReal centroid[3];
        DMPlexComputeCellGeometryFVM(dm, iCell, NULL, centroid, NULL);

        PetscInt numberFaces;
        const PetscInt *cellFaces;
        DMPlexGetConeSize(dm, iCell, &numberFaces);
        DMPlexGetCone(dm, iCell, &cellFaces);  //!< Get the face geometry associated with the current cell

        /** The cells with empty faces are the cells with ghost labels */
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, iCell, &ghost);
        }
        if (ghost >= 0) {
            printf("Ghost label!\n");
        }

        /** All but one of the faces for these ghost cells will have an invalid geometry */
        for (PetscInt f = 0; f < numberFaces; f++) {
            PetscInt face = cellFaces[f];
            DMPlexPointLocalRead(faceDM, face, faceGeomArray,
                                 &faceGeom);  //!< Reads the cell location from the current cell
            if (faceGeom->normal[0] == 0 && faceGeom->normal[1] == 0 && faceGeom->normal[2] == 0) {
                printf("Empty face normal!\n");
            }
        }
    }

    //    PetscInt npoints = 1;  //!< Number of points to insert into the particle field. One particle for each ray.
    //    PetscInt dim; //! Get the number of dimensions in the domain
    //    DMGetDimension(dm, &dim);
    //
    //    /** Create the DMSwarm */
    //    DM radsearch;  //!< DM which the search particles occupy
    //    DMCreate(MPI_COMM_WORLD, &radsearch);
    //    DMSetType(radsearch, DMSWARM);
    //    DMSetDimension(radsearch, dim);
    //    DMSwarmSetType(radsearch, DMSWARM_PIC); //!< Configure radsearch to be of type PIC
    //    DMSwarmSetCellDM(radsearch, dm);
    //
    //    /** Put a particle in the domain */
    //
    //    /** Step the particle throught the domain until it crosses a boundary */
    //
    //    /** Read out the particle properties as it is being stepped through the domain */

    /** Do some cleanup */
    if (cellRange.is) {
        ISRestorePointRange(cellRange.is, &cellRange.start, &cellRange.end, &cellRange.points);
        ISDestroy(&cellRange.is);
    }
    PetscCall(DMDestroy(&dm));

    PetscFinalize();
    return 0;
}

/*TEST

    args: -dm_refine 1
TEST*/
