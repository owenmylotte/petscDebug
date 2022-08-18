static const char help[] = "Tests for mesh extrusion";

#include <petscdmplex.h>


PetscErrorCode TagBoundary(DM dm, DMLabel adaptLabel, const char regionName[], PetscInt regionValue) {
    PetscFunctionBeginUser;
    // add points from each region
    DMLabel regionLabel;
    PetscCall(DMGetLabel(dm, regionName, &regionLabel));

    IS bdIS;
    const PetscInt *points;
    PetscInt n, i;

    PetscCall(DMLabelGetStratumIS(regionLabel, regionValue, &bdIS));
    if (!bdIS) {
        return 0;
    }
    PetscCall(ISGetLocalSize(bdIS, &n));
    PetscCall(ISGetIndices(bdIS, &points));
    for (i = 0; i < n; ++i) {
        PetscCall(DMLabelSetValue(adaptLabel, points[i], DM_ADAPT_REFINE));
    }
    PetscCall(ISRestoreIndices(bdIS, &points));
    PetscCall(ISDestroy(&bdIS));

    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    DM dm, dma = NULL, dmDist = NULL;
    DMLabel adaptLabel;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));


    // create a temporary label to hold adapt information
//    PetscCall(DMCreateLabel(dm, "AdaptationLabel"));
//    PetscCall(DMGetLabel(dm, "AdaptationLabel", &adaptLabel));
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "AdaptationLabel", &adaptLabel));

    // mark the boundary to extrude
    PetscCall(TagBoundary(dm, adaptLabel, "inlet", 18));
    PetscCall(TagBoundary(dm, adaptLabel, "outlet", 21));
    PetscCall(TagBoundary(dm, adaptLabel, "fuelGrain", 19));
    PetscCall(TagBoundary(dm, adaptLabel, "chamberWalls", 20));

    // mark the boundary faces
    PetscCall(DMPlexLabelComplete(dm, adaptLabel));

    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    // extrude based upon that label
    PetscCall(DMAdaptLabel(dm, adaptLabel, &dma));
    PetscCall(DMViewFromOptions(dma, NULL, "-dm_adapt_view"));

    // now distribute with the extruded cells
    PetscCall(DMPlexDistribute(dma, 1, NULL, &dmDist));

    PetscCall(DMLabelDestroy(&adaptLabel));
    PetscCall(DMDestroy(&dm));
    PetscCall(DMDestroy(&dma));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

 Case 2: 2D example with invalid corner cell
   mpirun -n 4  ./gMshExtrude -dm_plex_filename /Users/mcgurn/scratch/ablateInputs/test/combustionChamber2d_boundary_label_unrefinedv3.msh  -dm_view "hdf5:dm.h5" -dm_adapt_view "hdf5:adaptDm.h5"   -dm_adaptor cellrefiner  -dm_plex_transform_type extrude -dm_plex_transform_extrude_thickness 0.01  -dm_plex_check_all  -dm_plex_transform_extrude_use_tensor 0 -dm_plex_gmsh_use_regions -dm_plex_gmsh_mark_vertices -dm_partition_view


 Case 3: 2D Example with distribute error
 mpirun -n 34  ./gMshExtrude -dm_plex_filename /Users/mcgurn/scratch/ablateInputs/test/combustionChamber2d_boundary_label_unrefinedv3.msh  -dm_view "hdf5:dm.h5" -dm_adapt_view "hdf5:adaptDm.h5"  -dm_distribute_overlap 1   -dm_adaptor cellrefiner  -dm_plex_transform_type extrude -dm_plex_transform_extrude_thickness 0.05  -dm_plex_check_all  -dm_plex_transform_extrude_use_tensor 0 -dm_plex_gmsh_use_regions -dm_plex_gmsh_mark_vertices -dm_partition_view




 // unused inputs
 -dm_label_view AdaptationLabel

 -dm_
TEST*/
