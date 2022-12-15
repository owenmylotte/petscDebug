static const char help[] = "Simple Tests";

#include <petsc.h>

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 3;
    PetscInt faces[3] = {16, 8, 8};
    PetscReal lower[3] = {0.0, 0.0, 0};
    PetscReal upper[3] = {1, 1, 1};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Create a dm swarm
    DM swarmDm;
    PetscCall(DMCreate(PETSC_COMM_WORLD, &swarmDm));
    PetscCall(DMSetType(swarmDm, DMSWARM));
    PetscCall(DMSetDimension(swarmDm, dimensions));

    PetscCall(DMSwarmSetType(swarmDm, DMSWARM_PIC));
    PetscCall(DMSwarmSetCellDM(swarmDm, dm));

    PetscCall(DMSwarmFinalizeFieldRegister(swarmDm));

    PetscCall(DMSwarmSetLocalSizes(swarmDm,  rank == 0? 1 : 0, 10));
    PetscCall(DMSetFromOptions(swarmDm));

    PetscInt np;
    PetscCall(DMSwarmGetLocalSize(swarmDm, &np));

    PetscReal *coords;                   //!< Pointer to the coordinate field information
    PetscCall(DMSwarmGetField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
    for (PetscInt p = 0; p < np; ++p) {
        for (PetscInt d = 0; d < dimensions; ++d) {
            coords[p * dimensions + d] = 0.1;
        }
    }
    PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
    DMView(swarmDm, PETSC_VIEWER_STDOUT_WORLD);
    PetscCall(DMSwarmMigrate(swarmDm, PETSC_TRUE));
    DMView(swarmDm, PETSC_VIEWER_STDOUT_WORLD);


    PetscInt size;
    PetscCall(DMSwarmGetSize(swarmDm, &size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NumberParticles %" PetscInt_FMT "\n", size));

    PetscCall(DMDestroy(&swarmDm));
    PetscCall(DMDestroy(&dm));

    PetscFinalize();
    return 0;
}