static const char help[] = "Debug Code";

#include <petsc.h>

int main(int argc, char **argv)
{
    DM      dm, dmDist = NULL;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));

    // distribute the mesh with ghost cells
    PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
    PetscCall(DMPlexDistribute(dm, 1, NULL, &dmDist));
    if(dmDist){
        PetscCall(DMDestroy(&dm));
        dm = dmDist;
    }

    // View the mesh
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));


    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());

    return 0;
}

/*TEST
  args: -dm_plex_filename ${PETSC_DIR}/share/petsc/datafiles/meshes/square.msh -dm_view :mesh.tex:ascii_latex -dm_coord_space 0

TEST*/
