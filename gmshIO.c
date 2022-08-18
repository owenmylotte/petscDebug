static const char help[] = "Tests for mesh extrusion";

#include <petscdmplex.h>

static PetscErrorCode AddField(DM dm, const char name[], PetscInt size) {
    PetscFV fvm;
    PetscCall(PetscFVCreate(PetscObjectComm((PetscObject) dm), &fvm));
    PetscCall(PetscObjectSetName((PetscObject) fvm, name));
    PetscCall(PetscFVSetFromOptions(fvm));
    PetscCall(PetscFVSetNumComponents(fvm, size));

    // Only define the new field over the region used by this solver
    PetscCall(DMAddField(dm, NULL, (PetscObject) fvm));
    PetscCall(PetscFVDestroy(&fvm));
}

int main(int argc, char **argv) {
    DM dm, dma = NULL;
    DMLabel adaptLabel;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));

    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    // tag the boundary cells
    PetscCall(DMCreateLabel(dm, "marker"));
    DMLabel boundaryFaceLabel;
    PetscCall(DMGetLabel(dm, "marker", &boundaryFaceLabel));

    // mark the boundary faces
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, boundaryFaceLabel));
    PetscCall(DMPlexLabelComplete(dm, boundaryFaceLabel));


    // get the surface marker label
    PetscCall(DMGetLabel(dm, "marker", &adaptLabel));
    PetscCall(DMAdaptLabel(dm, adaptLabel, &dma));
    PetscCall(DMViewFromOptions(dma, NULL, "-dm_adapt_view"));

    // Create a fv field for viz
    PetscCall(AddField(dma, "rank", 1));
    PetscCall(AddField(dma, "position", 2));
    PetscCall(AddField(dma, "cellId", 1));
    PetscCall(DMCreateDS(dma));

    // Create the global vector
    Vec globalVector;
    VecSet(globalVector, 10);



    PetscCall(VecDestroy(&globalVector));
    PetscCall(DMDestroy(&dm));
    PetscCall(DMDestroy(&dma));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

 2D Example
 ./gMshExtrude -dm_plex_filename ${PETSC_DIR}/share/petsc/datafiles/meshes/square.msh -dm_view ascii::ascii_info_detail -dm_view :orgMesh.tex:ascii_latex -dm_adapt_view :adaptMesh.tex:ascii_latex  -dm_adaptor cellrefiner -dm_plex_transform_type extrude -dm_plex_transform_extrude_thickness 0.5  -dm_plex_check_all -dm_plex_gmsh_use_marker  -dm_plex_view_tikzscale 10.0 -dm_plex_transform_extrude_use_tensor 0

TEST*/
