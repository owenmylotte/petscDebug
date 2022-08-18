static const char help[] = "SubDMIoTest";

#include <petscdmplex.h>

static PetscErrorCode CreateHalfLabel(DM dm, DMLabel* label){
    PetscFunctionBeginUser;

    PetscCall(DMCreateLabel(dm, "half"));
    PetscCall(DMGetLabel(dm, "half", label));

    PetscInt cStart, cEnd;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

    // Get the number of dimensions from the dm
    PetscInt nDims;
    PetscCall(DMGetCoordinateDim(dm, &nDims));

    // get the region value
    const PetscInt labelValue = 1;

    // March over each cell
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscReal centroid[3];

        // get the center of the cell/face/vertex
        PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));

        // determine if the point is positive or negative
        if (centroid[0] > 0.5) {
            PetscCall(DMLabelSetValue(*label, c, labelValue));
        }
    }
    PetscCall(DMPlexLabelComplete(dm, *label));

    PetscFunctionReturn(0);

}

static PetscErrorCode AddField(DM dm, const char name[], DMLabel label, PetscInt size) {
    PetscFunctionBeginUser;
    PetscFV fvm;
    PetscCall(PetscFVCreate(PetscObjectComm((PetscObject) dm), &fvm));
    PetscCall(PetscObjectSetName((PetscObject) fvm, name));
    PetscCall(PetscFVSetFromOptions(fvm));
    PetscCall(PetscFVSetNumComponents(fvm, size));

    // Only define the new field over the region used by this solver
    PetscCall(DMAddField(dm, label, (PetscObject) fvm));
    PetscCall(PetscFVDestroy(&fvm));

    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    DM dm;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions  = 2;
    PetscInt faces[2] = {20, 20};
    PetscReal lower[2] = {0, 0};
    PetscReal upper[2] = {1, 1};
    DMBoundaryType bc[2] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions,  PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    // add new label
    DMLabel label;
    PetscCall(CreateHalfLabel(dm, &label));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    // Create a subdm
    DM subDM;
    PetscCall(DMPlexFilter(dm, label, 1, &subDM));

    // Set the field in the subDM
    PetscCall(AddField(subDM, "testField", NULL, 1));

    // output
    PetscCall(DMViewFromOptions(subDM, NULL, "-subdm_view"));

    PetscCall(DMDestroy(&subDM));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

 2D Example
 mpirun -n 3 ./subDmTest  -dm_view "hdf5:dm.h5"  -subdm_view "hdf5:subdm.h5" -dm_distribute_overlap 1

TEST*/
