static const char help[] = "DM Locate";

#include <petsc.h>

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 3;
    PetscInt faces[3] = {5, 5, 5};
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    // create any ghost cells that are needed
    DM dmDist = NULL;
    PetscCall(DMPlexDistribute(dm, 3, NULL, &dmDist));
    if(dmDist){
        DMDestroy(&dm);
        dm = dmDist;
    }

    // get the rank
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    Vec coords;
    PetscCall(VecCreate(PETSC_COMM_SELF, &coords));
    PetscCall(VecSetBlockSize(coords, dimensions));
    PetscCall(VecSetSizes(coords, 1 * dimensions, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(coords));
    PetscInt i[3] = {0, 1, 2};
    PetscReal position[3] = {.2, .3, .4};

    VecSetValues(coords, dimensions, i, position, INSERT_VALUES);
    VecAssemblyBegin(coords);
    VecAssemblyEnd(coords);

    // locate
    PetscSF sfcell = NULL;
    PetscCall(DMLocatePoints(dm, coords, DM_POINTLOCATION_NONE, &sfcell));

    const PetscSFNode *cells;
    PetscInt nFound;
    const PetscInt *found;
    PetscCall(PetscSFGetGraph(sfcell, NULL, &nFound, &found, &cells));

    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Rank %d found %" PetscInt_FMT "\n", rank, nFound);
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    PetscCall(PetscSFDestroy(&sfcell));
    PetscCall(DMDestroy(&dm));

    PetscFinalize();
    return 0;
}
/**
* Reports found on 56 ranks
 * mpirun -n 56 ./dmLocate -dm_plex_hash_location true
 *
 * or
 *
 * mpirun -n 56 ./dmLocate -dm_plex_hash_location false

 */