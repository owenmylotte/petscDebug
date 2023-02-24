static const char help[] = "Simple Tests";

#include <petsc.h>

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 3;
    PetscInt faces[3] = {188, 42, 42};
    PetscReal lower[3] = {-0.00053763440860215054, -0.00063499999999999993, -0.013335};
    PetscReal upper[3] = {0.10053763440860215, 0.026034999999999999, 0.013335};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    PetscCall(DMDestroy(&dm));

    PetscFinalize();
    return 0;
}