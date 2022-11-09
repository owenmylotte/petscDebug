static const char help[] = "SubDMIoTest";

#include <petscdmplex.h>
#include "petscds.h"

void f0_shear(PetscInt dim, PetscInt Nf, PetscInt NfAux,
              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
              const PetscScalar u_t[], const PetscScalar u_x[],
              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
              const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants,
              const PetscScalar constants[], PetscScalar coords[]) {
    PetscInt dir = (PetscInt) constants[dim];
    for (PetscInt c = 0; c < dim; ++c)
        coords[c] = u[c] + (c == dir ? 0.0 : constants[c]*u[dir]);
}

PetscErrorCode DMPlexShearGeometryDebug(DM dm, DMDirection direction, PetscReal multipliers[])
{
  DM             cdm;
  PetscDS        cds;
  PetscObject    obj;
  PetscClassId   id;
  PetscScalar   *moduli;
  const PetscInt dir = (PetscInt)direction;
  PetscInt       dE, d, e;

  DMGetCoordinateDM(dm, &cdm);
  DMGetCoordinateDim(dm, &dE);
  PetscMalloc1(dE + 1, &moduli);
  moduli[0] = dir;
  for (d = 0, e = 0; d < dE; ++d) moduli[d + 1] = d == dir ? 0.0 : (multipliers ? multipliers[e++] : 1.0);
  DMGetDS(cdm, &cds);
  PetscDSGetDiscretization(cds, 0, &obj);
  PetscObjectGetClassId(obj, &id);
  if (id != PETSCFE_CLASSID) {
    Vec          lCoords;
    PetscSection cSection;
    PetscScalar *coords;
    PetscInt     vStart, vEnd, v;

    DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
    DMGetCoordinateSection(dm, &cSection);
    DMGetCoordinatesLocal(dm, &lCoords);
    VecGetArray(lCoords, &coords);
    for (v = vStart; v < vEnd; ++v) {
      PetscReal ds;
      PetscInt  off, c;

      PetscSectionGetOffset(cSection, v, &off);
      ds = PetscRealPart(coords[off + dir]);
      for (c = 0; c < dE; ++c) coords[off + c] += moduli[c] * ds;
    }
    VecRestoreArray(lCoords, &coords);
  } else {
    PetscDSSetConstants(cds, dE + 1, moduli);
    DMPlexRemapGeometry(dm, 0.0, f0_shear);
  }
  PetscFree(moduli);
  return 0;
}

int main(int argc, char **argv) {
    DM dm;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 2;
    PetscInt faces[3] = {5, 5, 5};
    PetscReal lower[3] = {0, 0, 0};
    PetscReal upper[3] = {1, 1, 0};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

//    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
//    PetscCall(DMSetType(dm, DMPLEX));
//    PetscCall(DMSetFromOptions(dm));

    // using shear
//    DMPlexShearGeometryDebug(dm, DM_X, NULL);

    // Projection
//    {
//        DM cdm;
//        PetscDS cds;
//        PetscInt cdim;
//        PetscScalar *moduli;
//        PetscCall(DMGetCoordinateDM(dm, &cdm));
//        PetscCall(DMGetCoordinateDim(dm, &cdim));
//        PetscCall(PetscMalloc1(cdim+1, &moduli));
//        for (PetscInt d = 0; d < cdim; ++d) moduli[d] = 2.0;
//        moduli[cdim] = cdim-1;
//        PetscCall(DMGetDS(cdm, &cds));
//        PetscCall(PetscDSSetConstants(cds, cdim + 1, moduli));
//        PetscCall(DMPlexRemapGeometry(dm, 0.0, f0_shear));
//        PetscCall(PetscFree(moduli));
//    }
    // Direct Modification
    {
        PetscReal v[3] = {1.0, 1.0, 1.0};
        Vec xl;
        PetscScalar *coords;
        PetscInt cdim, N;
        DMGetCoordinateDim(dm, &cdim);
        DMGetCoordinatesLocal(dm, &xl);
        VecGetLocalSize(xl, &N);
        VecGetArrayWrite(xl, &coords);
        for (PetscInt p = 0; p < N / cdim; ++p) {
            for (PetscInt d = 0; d < cdim; ++d) {
                coords[p * cdim + d] += v[d];
            }
        }
        VecRestoreArrayWrite(xl, &coords);
        DMSetCoordinatesLocal(dm, xl);
    }

    // get the min rad
    PetscReal minRad;
    PetscCall(DMPlexGetGeometryFVM(dm, NULL, NULL, &minRad));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "minRad %f\n", minRad));


    // output
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

 # for same dim filter
    mpirun -dm_view "hdf5:filterDm.h5"


TEST*/
