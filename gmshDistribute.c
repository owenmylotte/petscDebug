static const char help[] = "Debug Code";

#include <petsc.h>

int main(int argc, char **argv)
{
    DM      dm,dma, dmDist = NULL;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMPlexDistributeSetDefault(dm, PETSC_TRUE));
    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view_before_extrude"));

    // get the surface marker label
    DMLabel adaptLabel;
    PetscCall(DMGetLabel(dm, "marker", &adaptLabel));
    PetscCall(DMAdaptLabel(dm, adaptLabel, &dma));
    if (dma) {
        PetscCall(DMDestroy(&dm));
        dm = dma;
    }

    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view_after_extrude"));

    // distribute the mesh with ghost cells
    PetscCall(DMPlexDistribute(dm, 2, NULL, &dmDist));
    if (dmDist) {
        PetscCall(DMDestroy(&dm));
        dm = dmDist;
    }

    // View the mesh
    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());

    return 0;
}

/*TEST
 *
  with -dm_plex_check_all
  mpirun -n 2 ./gMshDistribute -dm_plex_filename /Users/mcgurn/scratch/results/2Dproblem/combustionChamber2d_refinedv2.msh -dm_view :finalMesh.tex:ascii_latex -dm_view_after_extrude :afterExtrude.tex:ascii_latex -dm_view_before_extrude :beforeExtrude.tex:ascii_latex  -dm_adaptor cellrefiner -dm_plex_transform_type extrude -dm_plex_transform_extrude_thickness 0.0016  -dm_plex_check_all -dm_plex_gmsh_use_marker  -dm_plex_view_tikzscale 120.0 -dm_plex_transform_extrude_use_tensor 0

  // without -dm_plex_check_all
   mpirun -n 2 ./gMshDistribute -dm_plex_filename /Users/mcgurn/scratch/results/2Dproblem/combustionChamber2d_refinedv2.msh -dm_view :finalMesh.tex:ascii_latex -dm_view_after_extrude :afterExtrude.tex:ascii_latex -dm_view_before_extrude :beforeExtrude.tex:ascii_latex  -dm_adaptor cellrefiner -dm_plex_transform_type extrude -dm_plex_transform_extrude_thickness 0.0016 -dm_plex_gmsh_use_marker  -dm_plex_view_tikzscale 120.0 -dm_plex_transform_extrude_use_tensor 0

TEST*/
