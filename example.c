static const char help[] = "Extrudes the nozzle example";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions  = 2;
    PetscInt faces[2] = {6, 6};
    PetscReal lower[2] = {0, 0};
    PetscReal upper[2] = {1, 1};
    DMBoundaryType bc[2] = {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions,  PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    PetscCall(DMDestroy(&dm));

    PetscFinalize();
    return 0;
}

/*TEST

    args: -dm_refine 1
TEST*/
