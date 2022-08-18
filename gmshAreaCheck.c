static const char help[] = "Tests for mesh extrusion";

#include <petscdmplex.h>


int main(int argc, char **argv) {
    DM dm;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));

    // compute the dm geometry
    Vec cellGeomVec, faceGeomVec;
    PetscCall(DMPlexComputeGeometryFVM(dm, &cellGeomVec, &faceGeomVec));

    // Get the dim
    PetscInt dim;
    PetscCall(DMGetDimension(dm, &dim));

    // Get the dm for each value
    DM cellDM, faceDM;
    PetscCall(VecGetDM(cellGeomVec, &cellDM));
    PetscCall(VecGetDM(faceGeomVec, &faceDM));

    // Get the array data from the geom vec
    const PetscScalar *cellGeomArray;
    const PetscScalar *faceGeomArray;
    PetscCall(VecGetArrayRead(cellGeomVec, &cellGeomArray));
    PetscCall(VecGetArrayRead(faceGeomVec, &faceGeomArray));

    // Get cell 1206
    PetscInt cell = 1206;

    // March over each face connected to this cell
    const PetscInt *faces;
    PetscInt numberFaces;
    PetscCall(DMPlexGetConeSize(dm, cell, &numberFaces));
    PetscCall(DMPlexGetCone(dm, cell, &faces));

    // Store the area sum
    PetscScalar areaSum[3] = {0.0, 0.0, 0.0};

    for (PetscInt f = 0; f < numberFaces; f++) {
        PetscInt face = faces[f];

        // get the cells that touch this face
        const PetscInt *cells;
        PetscInt numberCells;
        PetscCall(DMPlexGetSupport(dm, face, &cells));
        PetscCall(DMPlexGetSupportSize(dm, face, &numberCells));

        // Check if left or right
        PetscInt leftOrRight = cell == cells[0] ? -1 : 1;

        // get the area from the faceGeomArray, this already has the corrected normal direction based on cell order https://gitlab.com/petsc/petsc/-/blob/main/src/dm/impls/plex/plexgeometry.c#L2534
        PetscFVFaceGeom *fg;
        PetscCall(DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg));

        // Add to the running total
        for (PetscInt d = 0; d < dim; d++) {
            areaSum[d] += fg->normal[d] * leftOrRight;
        }

    }

    PetscPrintf(PETSC_COMM_SELF, "AreaSum:\n  %g, %g, %g", areaSum[0], areaSum[1], areaSum[2]);

    // cleanup
    PetscCall(VecRestoreArrayRead(cellGeomVec, &cellGeomArray));
    PetscCall(VecRestoreArrayRead(faceGeomVec, &faceGeomArray));
    PetscCall(VecDestroy(&faceGeomVec));
    PetscCall(VecDestroy(&cellGeomVec));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

   ./gmeshAreaCheck -dm_plex_filename /Users/mcgurn/scratch/petscTest/CombustionChamberV5_unrefinedv1.1.msh


TEST*/
