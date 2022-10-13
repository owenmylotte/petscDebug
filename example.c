static const char help[] = "Extrudes the nozzle example";

#include <petsc.h>

struct {
    PetscInt origin;
    PetscInt iCell;
    PetscInt ntheta;
    PetscInt nphi;
    PetscInt nsegment;
} Identifier;

struct Virtualcoord {
    PetscReal x;
    PetscReal y;
    PetscReal z;
    PetscReal xdir;
    PetscReal ydir;
    PetscReal zdir;
    PetscReal hhere;
} Virtualcoord;

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions  = 3;
    PetscInt faces[3] = {560,80,80};
    PetscReal lower[3] = {0.0, 0.0, -0.0127};
    PetscReal upper[3] = { 0.165354, 0.027686, 0.0127};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions,  PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));


    // Create a dm swarm
    DM swarmDm;
    PetscCall(DMCreate(PETSC_COMM_WORLD, &swarmDm));
    PetscCall(DMSetType(swarmDm, DMSWARM));
    PetscCall(DMSetDimension(swarmDm, dimensions));

    PetscCall(DMSwarmSetType(swarmDm, DMSWARM_PIC));
    PetscCall(DMSwarmSetCellDM(swarmDm, dm));

    PetscCall(DMSwarmFinalizeFieldRegister(swarmDm));

    PetscCall(DMSwarmSetLocalSizes(swarmDm, 0, 0));
    PetscCall(DMSetFromOptions(swarmDm));

    PetscReal* coord;                   //!< Pointer to the coordinate field information
    PetscCall(DMSwarmGetField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void**)&coord));
    PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void**)&coord));

    PetscCall(DMSwarmMigrate(swarmDm, PETSC_TRUE));

    PetscCall(DMDestroy(&swarmDm));
    PetscCall(DMDestroy(&dm));

    PetscFinalize();
    return 0;
}

/*TEST

    args: -dm_refine 1
TEST*/
