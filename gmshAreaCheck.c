static const char help[] = "Tests for mesh extrusion";

#include <petscdmplex.h>
#include <petsc/private/dmpleximpl.h>


int main(int argc, char **argv) {
    DM dm;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));


    // Determine the number of faces
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_init"));



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
        PetscPrintf(PETSC_COMM_SELF, "LeftOrRightFactor %d: %d\n", face, leftOrRight);



        // get the area from the faceGeomArray, this already has the corrected normal direction based on cell order https://gitlab.com/petsc/petsc/-/blob/main/src/dm/impls/plex/plexgeometry.c#L2534
        PetscFVFaceGeom *fg;
        PetscCall(DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg));
        PetscPrintf(PETSC_COMM_SELF, "faceArea: %g %g %g\n", fg->normal[0], fg->normal[1], fg->normal[2]);

        // Add to the running total
        for (PetscInt d = 0; d < dim; d++) {
            areaSum[d] += fg->normal[d] * leftOrRight;
        }

        // compute the default normal
        PetscReal faceCentroid[3] = {0.0, 0.0, 0.0};
        PetscReal faceNormal[3] = {0.0, 0.0, 0.0};
        PetscReal area;
        PetscCall(DMPlexComputeCellGeometryFVM(dm, face, &area, faceCentroid, faceNormal));

        PetscReal outputScale = .1;
        FILE *fp;
        char fname[PETSC_MAX_PATH_LEN];
        PetscSNPrintf(fname, sizeof(fname), "face.%d.dat", face);
        PetscFOpen(PETSC_COMM_SELF, fname, "w", &fp);
        PetscFPrintf(PETSC_COMM_SELF, fp, "p x y z\n");
        PetscFPrintf(PETSC_COMM_SELF, fp, "1 %f %f %f\n", faceCentroid[0], faceCentroid[1], faceCentroid[2]);
        PetscFPrintf(PETSC_COMM_SELF, fp, "2 %f %f %f\n", (faceCentroid[0] + outputScale * faceNormal[0]),
                     (faceCentroid[1] + outputScale * faceNormal[1]), (faceCentroid[2] + outputScale * faceNormal[2]));
        PetscFClose(PETSC_COMM_SELF, fp);

        // output the check vector
        PetscInt cEndInterior;
        PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
        if (cEndInterior < 0) cEndInterior = cEnd;

        // scale the normal by the area
        for (PetscInt d = 0; d < dim; ++d) {
            faceNormal[d] *= area;
        }

        PetscReal *lcentroid, *rcentroid;
        PetscFVCellGeom *cL, *cR;
        PetscReal l[3], r[3], v[3];
        PetscCall(DMPlexPointLocalRead(cellDM, cells[0], cellGeomArray, &cL));
        lcentroid = cells[0] >= cEndInterior ? faceCentroid : cL->centroid;
        if (numberCells > 1) {
            PetscCall(DMPlexPointLocalRead(cellDM, cells[1], cellGeomArray, &cR));
            rcentroid = cells[1] >= cEndInterior ? faceCentroid : cR->centroid;
        } else {
            rcentroid = faceCentroid;
        }
        for(PetscInt d =0; d < dim; d++) {
            l[d] = lcentroid[d];
            r[d] = rcentroid[d];
        }
//        PetscCall(DMLocalizeCoordinateReal_Internal(dm, dim, faceCentroid, lcentroid, l));
//        PetscCall(DMLocalizeCoordinateReal_Internal(dm, dim, faceCentroid, rcentroid, r));
        DMPlex_WaxpyD_Internal(dim, -1, l, r, v);

        PetscPrintf(PETSC_COMM_SELF, "faceNormalDot %d: %g\n", face, DMPlex_DotRealD_Internal(dim, faceNormal, v));

        PetscSNPrintf(fname, sizeof(fname), "faceCheck.%d.dat", face);
        PetscFOpen(PETSC_COMM_SELF, fname, "w", &fp);
        PetscFPrintf(PETSC_COMM_SELF, fp, "p x y z\n");
        PetscFPrintf(PETSC_COMM_SELF, fp, "1 %f %f %f\n", l[0], l[1], l[2]);
        PetscFPrintf(PETSC_COMM_SELF, fp, "2 %f %f %f\n", r[0], r[1], r[2]);
        PetscFClose(PETSC_COMM_SELF, fp);

    }

    PetscPrintf(PETSC_COMM_SELF, "AreaSum:\n  %g, %g, %g", areaSum[0], areaSum[1], areaSum[2]);


    {
        // output the mesh
        // Create a label
        PetscCall(DMCreateLabel(dm, "filterLabel"));
        DMLabel filterLabel;
        PetscCall(DMGetLabel(dm, "filterLabel", &filterLabel));
        PetscCall(DMLabelSetValue(filterLabel, cell, 1));

        DM filterDM;
        PetscCall(DMPlexFilter(dm, filterLabel, 1, &filterDM));
        PetscCall(DMViewFromOptions(filterDM, NULL, "-dm_view"));

    }





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


./gmeshAreaCheck -dm_plex_filename /Users/mcgurn/scratch/petscTest/CombustionChamberV5_unrefinedv1.1.msh -dm_view "hdf5:filterDm.h5" -dm_init "hdf5:dm.h5"
TEST*/
