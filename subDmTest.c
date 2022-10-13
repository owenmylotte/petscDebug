static const char help[] = "SubDMIoTest";

#include <petscdmplex.h>

static PetscErrorCode CreateHalfLabel(DM dm, DMLabel *label, PetscScalar height) {
    PetscFunctionBeginUser;

    PetscCall(DMCreateLabel(dm, "half"));
    PetscCall(DMGetLabel(dm, "half", label));

    PetscInt fStart, fEnd, depth = 0;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-filterDepth", &depth, NULL));
    PetscCall(DMPlexGetHeightStratum(dm, depth, &fStart, &fEnd));

    // Get the number of dimensions from the dm
    PetscInt nDims;
    PetscCall(DMGetCoordinateDim(dm, &nDims));

    // get the region value
    const PetscInt labelValue = 1;

    // March over each face
    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscReal centroid[3];

        // get the center of the cell/face/vertex
        PetscCall(DMPlexComputeCellGeometryFVM(dm, f, NULL, centroid, NULL));

        // determine if the point is positive or negative
        if (centroid[1] < (height + 1E-8) && centroid[1] > (height - 1E-8) && centroid[0] > .8 ) {
            PetscCall(DMLabelSetValue(*label, f, labelValue));
        }
    }
    PetscCall(DMPlexLabelComplete(dm, *label));

    PetscFunctionReturn(0);

}

int main(int argc, char **argv) {
    DM dm;
    PetscInt cdim;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 2;
    PetscInt faces[3] = {5, 5, 5};
    PetscReal lower[3] = {0, 0, 0};
    PetscReal upper[3] = {1, 1, 0};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    PetscInt depth = 0;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-filterDepth", &depth, NULL));
    if(depth == 1){
        faces[0] = 6;
        faces[1] = 6;
        faces[2] = 6;
    }

    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    // add new label
    DMLabel label;
    PetscCall(CreateHalfLabel(dm, &label, 0.5));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    // Create a subdm
    DM subDM;
    PetscCall(DMPlexFilter(dm, label, 1, &subDM));
    //PetscCall(DMPlexCreateSubmesh(dm, label, 1, PETSC_TRUE, &subDM));
    PetscCall(PetscObjectSetName((PetscObject) subDM, "subDM"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) subDM, "sub_"));

    // Add a single field to the entire subDM
    PetscFV fvm;
    PetscCall(PetscFVCreate(PetscObjectComm((PetscObject) subDM), &fvm));
    PetscCall(PetscObjectSetName((PetscObject) fvm, "testField"));
    PetscCall(PetscFVSetNumComponents(fvm, 1));
    PetscCall(DMGetCoordinateDim(subDM, &cdim));
    PetscCall(PetscFVSetSpatialDimension(fvm, cdim));
    PetscCall(PetscFVSetFromOptions(fvm));

    PetscCall(DMAddField(subDM, NULL, (PetscObject) fvm));
    PetscCall(PetscFVDestroy(&fvm));
    PetscCall(DMCreateDS(subDM));

    // output
    PetscCall(DMView(subDM, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMViewFromOptions(subDM, NULL, "-dm_view"));

    // Create a globalVec
    Vec globVec;
    PetscCall(DMCreateGlobalVector(subDM, &globVec));
    PetscCall(PetscObjectSetName((PetscObject) globVec, "potential"));
    PetscCall(VecSet(globVec, 1.));
    PetscInt vecSize;
    PetscCall(VecGetSize(globVec, &vecSize));
    PetscPrintf(PETSC_COMM_WORLD, "VecSize: %d\n", vecSize);

    // output
    PetscCall(VecViewFromOptions(globVec, NULL, "-vec_view"));

    PetscCall(DMRestoreGlobalVector(subDM, &globVec));
    PetscCall(DMDestroy(&subDM));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

 # for same dim filter
    mpirun -n 3 ./subDmTest  -dm_view "hdf5:dm.h5"  -sub_dm_view "hdf5:subdm.h5" -vec_view "hdf5:subdm.h5::append" -filterDepth 0

 # for reduction in dim filter
     mpirun -n 3 ./subDmTest  -dm_view "hdf5:dm.h5"  -sub_dm_view "hdf5:subdm.h5" -vec_view "hdf5:subdm.h5::append" -filterDepth 1

TEST*/
