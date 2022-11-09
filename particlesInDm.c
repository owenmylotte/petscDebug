static const char help[] = "Simple Tests";

#include <petsc.h>

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions  = 2;
    PetscInt faces[3] = {20,20};
    PetscReal lower[3] = {0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0};
    DMBoundaryType bc[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions,  PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    // get the rank
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // Set the max number of particles
    PetscInt particleInitSize = 10;


    // Create a dm swarm
    DM swarmDm;
    PetscCall(DMCreate(PETSC_COMM_WORLD, &swarmDm));
    PetscCall(DMSetType(swarmDm, DMSWARM));
    PetscCall(DMSetDimension(swarmDm, dimensions));

    PetscCall(DMSwarmSetType(swarmDm, DMSWARM_PIC));
    PetscCall(DMSwarmSetCellDM(swarmDm, dm));

    PetscCall(DMSwarmFinalizeFieldRegister(swarmDm));

    PetscCall(DMSwarmSetLocalSizes(swarmDm, rank == 0? particleInitSize : 0, 1));
    PetscCall(DMSetFromOptions(swarmDm));

    PetscReal* coords;
    PetscInt *cellid;
    PetscCall(DMSwarmGetField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void**)&coords));
    PetscCall(DMSwarmGetField(swarmDm, DMSwarmPICField_cellid, NULL, NULL, (void**)&cellid));

    PetscInt np;
    PetscCall(DMSwarmGetLocalSize(swarmDm, &np));
    for (PetscInt p = 0; p < np; ++p) {
        for (PetscInt d = 0; d < dimensions; ++d) {
            coords[p * dimensions] = 0.5;
            coords[p * dimensions + 1] = (upper[1] - lower[1])/particleInitSize* p + lower[1];
        }
        cellid[p] = 0;
    }
    PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
    DMSwarmMigrate(swarmDm, PETSC_TRUE);

    // while there are particles
    PetscInt npGlobal, npLocal;
    PetscInt migration = 0;
    PetscCall(DMSwarmGetSize(swarmDm, &npGlobal));

    // determine minum cell size
    PetscReal minCellRadius;
    PetscCall(DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius));
    minCellRadius*=0.5;

    // keep track of the total maximum particles
    PetscInt maxParticles = 0;

    while (npGlobal) {

        PetscCall(DMSwarmGetLocalSize(swarmDm, &npLocal));
        PetscPrintf(PETSC_COMM_WORLD, "########################################################################\n");
        PetscPrintf(PETSC_COMM_WORLD, "Global Particles %" PetscInt_FMT " at Migration %"  PetscInt_FMT "\n", npGlobal, migration++);

//        DMViewFromOptions(swarmDm, NULL, "-dm_view" );

        PetscCall(DMSwarmGetField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void**)&coords));
        PetscCall(DMSwarmGetField(swarmDm, DMSwarmField_pid, NULL, NULL, (void**)&cellid));


        for(int r =0; r < size; r++){
            if(r == rank){
                if(npLocal){
                    printf("Rank %" PetscInt_FMT " Particles: %"  PetscInt_FMT "\n", (PetscInt)rank, npLocal);
                }
                for(PetscInt p =0; p < npLocal; p++){
                    printf("\t%"PetscInt_FMT ": %f, %f, %f\n", cellid[p], coords[p*dimensions], dimensions > 1? coords[p*dimensions + 1]: 0.0, dimensions > 2? coords[p*dimensions+2]: 0.0 );
                }
            }
            MPI_Barrier(PETSC_COMM_WORLD);
        }


        // Move
        for (PetscInt p = 0; p < npLocal; ++p) {
//            coords[p * dimensions] += minCellRadius;

            for (PetscInt d = 0; d < dimensions; ++d) {
                coords[p * dimensions + d] += minCellRadius;
            }
        }

        PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
        PetscCall(DMSwarmRestoreField(swarmDm, DMSwarmField_pid, NULL, NULL, (void **)&cellid));

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
* Grows particles:
 * -n 4 ./particlesInDm -dm_distribute_overlap 1 -dm_plex_hash_location true
* Does not grow particles:
 * -n 4 ./particlesInDm -dm_distribute_overlap 1 -dm_plex_hash_location false

*/