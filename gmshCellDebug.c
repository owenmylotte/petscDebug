static const char help[] = "Reproducer of bug in gMsh and hash location interaction. Bug disappears with hash location off.";

#include <petscdmplex.h>
#include "petscdm.h"
#include <petsc.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    DM dm = NULL;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));

    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    /** Get the dimensionality of the mesh */
    PetscInt dim = 0;
    PetscCall(DMGetDimension(dm, &dim));

    /** Get the indexes of all of the cells in the mesh
     * These indexes can be read to produce valid centroid locations
     * If DMLocatePoints fails to see any of these points, this demonstrates a bug.
     * */
    PetscInt start; //! Variables indicating the cell indexes of the mesh
    PetscInt end;
    const PetscInt *points;
    PetscInt depth;
    IS allPointIS;

    /** Get the range of valid points that exist within the dm */
    PetscCall(DMPlexGetDepth(dm, &depth));
    PetscCall(DMGetStratumIS(dm, "dim", depth, &allPointIS));
    if (!allPointIS) PetscCall(DMGetStratumIS(dm, "depth", depth, &allPointIS));
    PetscCall(ISGetPointRange(allPointIS, &start, &end, &points));

    PetscInt nPoints = (end - start); //! How many cells are there?

    /** Set up the vector to fill with points for the DMLocatePoints call */
    Vec intersect;
    PetscCall(VecCreate(PETSC_COMM_SELF, &intersect));  //! Instantiates the vector object
    PetscCall(VecSetBlockSize(intersect, dim)); //! Set the block size for each point location
    PetscCall(VecSetSizes(intersect, PETSC_DECIDE, nPoints * dim));  //! Set total size of the vector
    PetscCall(VecSetFromOptions(intersect)); //! Create the vector based on these options
    PetscInt i[3] = {0, 1, 2}; //! Establish the vector index here so that it can be iterated.

    PetscCall(DMGetCoordinatesLocalSetUp(dm));

    /** Iterate through all of the cells in the mesh
     * Get the centroid of each cell in the mesh
     * Use DMLocatePoints to get the cell index back out of that location
     * If DMLocatePoints fails to find a cell in that location, something is broken.
     * */
    for (PetscInt c = start; c < end; ++c) {
        const PetscInt iCell = points ? points[c] : c; //! Represents the cell index
        PetscReal centroid[3];
        /** Get the centroid of the cell at this index */
        PetscCall(DMPlexComputeCellGeometryFVM(dm, iCell, PETSC_NULLPTR, centroid, PETSC_NULLPTR));

        /** At this point the centroid pointer should contain a valid cell location
         * Get the centroid coordinates and put them into the vector */
        PetscReal position[3] = {
                (centroid[0]),   //!< x component
                (centroid[1]),   //!< y component
                (centroid[2])};  //!< z component

        /** This block creates the vector pointing to the cell whose index will be stored during the current loop */
        VecSetValues(intersect, dim, i, position, INSERT_VALUES);  //!< Input the values of the vector
        i[0] += dim; //!< Iterate the index by the number of dimensions so that the DMLocatePoints function can be called collectively.
        i[1] += dim;
        i[2] += dim;
    }

    /**Get the cells sitting on those points */
    PetscSF cellSF = PETSC_NULLPTR;
    PetscCall(DMLocatePoints(dm, intersect, DM_POINTLOCATION_NONE, &cellSF));

    /** Get the cells that were found from DMLocatePoints */
    PetscInt nFound;
    const PetscSFNode *cell = PETSC_NULLPTR;
    const PetscInt *found = PETSC_NULLPTR;
    PetscSFGetGraph(cellSF, NULL, &nFound, &found, &cell);

    /** Check whether all of the cell centroids were found as valid cell locations by DMLocatePoints */
    if (nFound != nPoints) printf("One or more cell locations were invalid!\n");

    /** Iterate through all of the cells one more time.
     * Check each cell index in the set against the point that is returned by DMLocatePoints.
     * If the cell indexes are not the same, then the input cell index is not recognized by DMLocatePoints.
     * */
    PetscInt iter = 0;
    for (PetscInt c = start; c < end; ++c) {
        const PetscInt iCell = points ? points[c] : c; //! Represents the cell index

        PetscReal centroid[3];
        /** Get the centroid of the cell at this index */
        PetscCall(DMPlexComputeCellGeometryFVM(dm, iCell, PETSC_NULLPTR, centroid, PETSC_NULLPTR));

        if (cell[iter].index != iCell)
            printf("Output index is not equal to input! %i %i\n", iCell, cell[iter].index);
        iter = iter + 1;
    }

    /** Clean up the objects */
    PetscCall(VecDestroy(&intersect));
    PetscCall(PetscSFDestroy(&cellSF));
    PetscCall(ISDestroy(&allPointIS));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

-dm_plex_filename /path/to/petscDebug/radEqTest.msh -dm_view "hdf5:dm.h5" -dm_adapt_view "hdf5:adaptDm.h5" -dm_plex_hash_location "true"
-dm_plex_filename /path/to/petscDebug/CombustionChamberV5_unrefinedv1.1.msh -dm_view "hdf5:dm.h5" -dm_adapt_view "hdf5:adaptDm.h5" -dm_plex_hash_location "true"

TEST*/
