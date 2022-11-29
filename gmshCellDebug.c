static const char help[] = "Tests DMPlex gmsh compatability";

#include <petscdmplex.h>
#include "petscdm.h"
#include <petsc.h>

int main(int argc, char **argv) {
    DM dm, dma = NULL, dmDist = NULL;

    PetscInt dim = 2;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));

    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    /** Get the indexes of all of the cells in the mesh
     * These indexes can be read to produce valid centroid locations
     * If DMLocatePoints fails to see any of these points, this demonstrates a bug.
     * */
    PetscInt start; //! Variables indicating the cell indexes of the mesh
    PetscInt end;
    const PetscInt *points;

    PetscInt depth;
    PetscCall(DMPlexGetDepth(dm, &depth));
    IS allPointIS;
    PetscCall(DMGetStratumIS(dm, "dim", depth, &allPointIS));
    if (!allPointIS) PetscCall(DMGetStratumIS(dm, "depth", depth, &allPointIS));
    PetscCall(ISGetPointRange(allPointIS, &start, &end, &points));

    /** Iterate through all of the cells in the mesh
     * Get the centroid of each cell in the mesh
     * Use DMLocatePoints to get the cell index back out of that location
     * If DMLocatePoints fails to find a cell in that location, throw an error.
     * */
    for (PetscInt c = start; c < end; ++c) {
        const PetscInt iCell = points ? points[c] : c; //! Represents the cell index
        PetscReal centroid[3];
        /** Get the centroid of the cell at this index */
        PetscCall(DMPlexComputeCellGeometryFVM(dm, iCell, PETSC_NULLPTR, centroid, PETSC_NULLPTR));

        /** At this point the centroid pointer should contain a valid cell location */

        Vec intersect;
        PetscCall(VecCreate(PETSC_COMM_SELF, &intersect));  //! Instantiates the vector
        PetscCall(VecSetBlockSize(intersect, dim)); //! Set the block size for each point location
        PetscCall(VecSetSizes(intersect, PETSC_DECIDE, dim));  //! Set size
        PetscCall(VecSetFromOptions(intersect));
        PetscInt i[3] = {0, 1, 2};                   //!< Establish the vector here so that it can be iterated.

        /** Get the centroid coordinates and put them into the vector */
        PetscReal position[3] = {(centroid[0]),   //!< x component
                                 (centroid[1]),   //!< y component
                                 (centroid[2])};  //!< z component

        /** This block creates the vector pointing to the cell whose index will be stored during the current loop */
        VecSetValues(intersect, dim, i, position,
                     INSERT_VALUES);  //!< Actually input the values of the vector (There are 'dim' values to input)

        /** Loop through points to try to get the cell that is sitting on that point */
        PetscSF cellSF = PETSC_NULLPTR;
        PetscCall(DMLocatePoints(dm, intersect, DM_POINTLOCATION_NONE, &cellSF));

        /** Get the cells that were found from DMLocatePoints */
        PetscInt nFound;
        const PetscSFNode *cell = PETSC_NULLPTR;
        const PetscInt *found = PETSC_NULLPTR;
        PetscSFGetGraph(cellSF, NULL, &nFound, &found, &cell);

        if (nFound == 0)
            printf("No cell was found at centroid! %i %f %f %f\n", iCell, centroid[0], centroid[1],
                   centroid[2]);
        if (nFound == 1 && cell[0].index != iCell)
            printf("Output index is not equal to input! %i %i\n", iCell, cell[0].index);

        /** Cleanup from DMLocatePoints */
        PetscCall(VecDestroy(&intersect));
        PetscCall(PetscSFDestroy(&cellSF));
    }

    /** Clean up the objects */
    PetscCall(ISDestroy(&allPointIS));
    PetscCall(DMDestroy(&dm));
    PetscCall(DMDestroy(&dma));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

-dm_plex_filename /home/owen/CLionProjects/petscDebug/R1_5m.msh -dm_view "hdf5:dm.h5" -dm_adapt_view "hdf5:adaptDm.h5"
-dm_plex_filename /home/owen/CLionProjects/petscDebug/CombustionChamberV5_unrefinedv1.1.msh -dm_view "hdf5:dm.h5" -dm_adapt_view "hdf5:adaptDm.h5"

TEST*/
