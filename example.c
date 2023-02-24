static const char help[] = "Simple Tests";

#include <petsc.h>

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 3;
    PetscInt faces[3] = {2, 2, 2};
    PetscReal lower[3] = {0.0, 0.0, 0};
    PetscReal upper[3] = {1, 1, 1};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    PetscCall(DMDestroy(&dm));

    PetscFinalize();
    return 0;
}