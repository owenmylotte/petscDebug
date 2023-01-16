static const char help[] = "Particle Disappearance Reproducer";

#include <petsc.h>
#include <stdio.h>

struct Direction {
    PetscReal xdir, ydir, zdir; //! Represent the travel direction of each particle
};

double r2() {
    return (double) rand() / (double) RAND_MAX;
}

void rdir(struct Direction *dir) {
    dir->xdir = r2();
    dir->ydir = r2();
    dir->zdir = r2();
}

int main(int argc, char **argv) {


    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 3;
    PetscInt faces[3] = {50, 50, 50};
    PetscReal lower[3] = {-0.1, -0.1, -0.1};
    PetscReal upper[3] = {1.1, 1.1, 1.1};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    // get the rank
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // Set the max number of particles
    PetscInt particleInitSize = 10;

    struct Direction dir;

    // Create a dm swarm
    DM swarmDm;
    PetscCall(DMCreate(PETSC_COMM_WORLD, &swarmDm));
    PetscCall(DMSetType(swarmDm, DMSWARM));
    PetscCall(DMSetDimension(swarmDm, dimensions));

    PetscCall(DMSwarmSetType(swarmDm, DMSWARM_PIC));
    PetscCall(DMSwarmSetCellDM(swarmDm, dm));

    PetscCall(DMSwarmRegisterUserStructField(swarmDm, "DirectionField",
                                             sizeof(dir)));                                         //!< A field representing the three dimensional coordinates of the particle. Three "virtual" dims are required.

    PetscCall(DMSwarmFinalizeFieldRegister(swarmDm));

    PetscCall(DMSwarmSetLocalSizes(swarmDm, rank == 0 ? particleInitSize : 0, 1));
    PetscCall(DMSetFromOptions(swarmDm));

    PetscReal *coords;
    PetscInt *cellid;
    struct Direction *direction;
    PetscCall(DMSwarmGetField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
    PetscCall(DMSwarmGetField(swarmDm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
    PetscCall(DMSwarmGetField(swarmDm, "DirectionField", NULL, NULL, (void **) &direction));

    PetscInt np;
    PetscCall(DMSwarmGetLocalSize(swarmDm, &np));
    for (PetscInt p = 0; p < np; ++p) {
        for (PetscInt d = 0; d < dimensions; ++d) {
            coords[p * dimensions +
                   d] = r2(); //! Set the initial coordinates of the particles to a random number between zero and one.
        }

        rdir(&direction[p]); //! Set a random direction for the particles as they are initialized.

        cellid[p] = p;
    }
    PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
    PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
    PetscCall(DMSwarmRestoreField(swarmDm, "DirectionField", NULL, NULL, (void **) &direction));

    DMSwarmMigrate(swarmDm, PETSC_TRUE);

    // while there are particles
    PetscInt npGlobal, npLocal;
    PetscInt migration = 0;
    PetscCall(DMSwarmGetSize(swarmDm, &npGlobal));

    // determine minum cell size
    PetscReal minCellRadius;
    PetscCall(DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius));
    minCellRadius *= 0.5;

    // keep track of the total maximum particles
    PetscInt maxParticles = 0;

    while (npGlobal) {

        PetscCall(DMSwarmGetLocalSize(swarmDm, &npLocal));
        PetscPrintf(PETSC_COMM_WORLD, "########################################################################\n");
        PetscPrintf(PETSC_COMM_WORLD, "Global Particles %" PetscInt_FMT " at Migration %"  PetscInt_FMT "\n", npGlobal,
                    migration++);

        PetscCall(DMSwarmGetField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
        PetscCall(DMSwarmGetField(swarmDm, DMSwarmField_pid, NULL, NULL, (void **) &cellid));
        PetscCall(DMSwarmGetField(swarmDm, "DirectionField", NULL, NULL, (void **) &direction));

        for (int r = 0; r < size; r++) {
            if (r == rank) {
                if (npLocal) {
                    printf("Rank %" PetscInt_FMT " Particles: %"  PetscInt_FMT "\n", (PetscInt) rank, npLocal);
                }
                for (PetscInt p = 0; p < npLocal; p++) {
                    printf("\t%"PetscInt_FMT ": %f, %f, %f, %f, %f, %f\n", cellid[p], coords[p * dimensions],
                           dimensions > 1 ? coords[p * dimensions + 1] : 0.0,
                           dimensions > 2 ? coords[p * dimensions + 2] : 0.0, direction[p].xdir, direction[p].ydir,
                           direction[p].zdir);
                }
            }
            MPI_Barrier(PETSC_COMM_WORLD);
        }

        /**
         * Modify vector for the particles to move around the domain
         * If the particles are in the boundary region of the domain then they should be reflected off of the wall.
         */
        for (PetscInt p = 0; p < npLocal; ++p) {
            for (PetscInt d = 0; d < dimensions; ++d) {
                if (coords[p * dimensions + d] > 1.0) {
                    switch (d) {
                        case 0:
                            direction[p].xdir = -1 * PetscAbsReal(direction[p].xdir);
                        case 1:
                            direction[p].ydir = -1 * PetscAbsReal(direction[p].ydir);
                        case 2:
                            direction[p].zdir = -1 * PetscAbsReal(direction[p].zdir);
                    }
                }
                if (coords[p * dimensions + d] < 0.0) {
                    switch (d) {
                        case 0:
                            direction[p].xdir = PetscAbsReal(direction[p].xdir);
                        case 1:
                            direction[p].ydir = PetscAbsReal(direction[p].ydir);
                        case 2:
                            direction[p].zdir = PetscAbsReal(direction[p].zdir);
                    }
                }
            }
        }

        // Move
        for (PetscInt p = 0; p < npLocal; ++p) {
            coords[p * dimensions + 0] += minCellRadius * direction[p].xdir;
            coords[p * dimensions + 1] += minCellRadius * direction[p].ydir;
            coords[p * dimensions + 2] += minCellRadius * direction[p].zdir;
        }

        PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
        PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmField_pid, NULL, NULL, (void **) &cellid));
        PetscCall(DMSwarmRestoreField(swarmDm, "DirectionField", NULL, NULL, (void **) &direction));

        // move and update
        DMSwarmMigrate(swarmDm, PETSC_TRUE);
        PetscCall(DMSwarmGetSize(swarmDm, &npGlobal));
        maxParticles = PetscMax(maxParticles, npGlobal);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Max Global Particles %" PetscInt_FMT "\n", maxParticles);

    PetscCall(DMDestroy(&swarmDm));
    PetscCall(DMDestroy(&dm));

    PetscFinalize();
    return 0;
}
/**
* Conserves particles:
*       -n 1 ./vanishingParticles -dm_distribute_overlap 1 -dm_plex_hash_location true
* Does not conserve particles:
*       -n 10 ./vanishingParticles -dm_distribute_overlap 1 -dm_plex_hash_location true
* Does not conserve particles:
*       -n 10 ./vanishingParticles -dm_distribute_overlap 1 -dm_plex_hash_location false
* Does not conserve particles:
*       -n 10 ./vanishingParticles -dm_distribute_overlap 0 -dm_plex_hash_location true
*/