static const char help[] = "Debug Code";

#include <petsc.h>

int main(int argc, char **argv) {
    DM dm;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_init"));
    PetscCall(DMGetCoordinatesLocalSetUp(dm));

    // March over each cell
    PetscInt cStart, cEnd;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));


    for(PetscInt c = cStart; c < cEnd; ++c){
        // compute the volume for this cell
        PetscReal volume;
        PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &volume, NULL, NULL));

        if(volume < 0){
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Negative Volume in Cell %" PetscInt_FMT " is %f\n", c, volume));
        }
    }

    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
}
