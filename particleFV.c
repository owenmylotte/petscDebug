static const char help[] = "Testing particle access to face geometry in parallel";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscsf.h>
#include <unistd.h>
#include <petsc/private/dmpleximpl.h>
#include "stdbool.h"

int main(int argc, char **argv) {
    sleep(10);

    /** When run with -n 4, the face geometry of several ghost labelled cells is blank */

    bool ghostboundary = true; //!< Whether or not the mesh will be created with ghost cells
    bool ghostpartition = true;
    PetscReal steplength = 0.05; //!< How far the particle will be stepped on each iteration

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 2; //!< Number of dimensions to instantiate the mesh with
    PetscInt faces[2] = {10, 10};
    PetscReal lower[2] = {0, 0};
    PetscReal upper[2] = {1, 1};
    DMBoundaryType bc[2] = {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMPlexDistributeSetDefault(dm, PETSC_TRUE));
    PetscCall(DMSetFromOptions(dm));

    /** Add ghost cells to the mesh */
    if (ghostpartition) {        //!< Adds the inter-partition ghost cells to the dm
        DM dmDist = NULL;
        PetscInt ghostCellDepth = 1;
        PetscCall(DMPlexDistribute(dm, ghostCellDepth, NULL, &dmDist)); //!< create any ghost cells that are needed
        if (dmDist) {
            PetscCall(DMDestroy(&dm));
            dm = dmDist;
        }
//        PetscCall(DMDestroy(&dmDist));
        DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);
    }

    if (ghostboundary) {/** Adds boundary ghost cells to the dm */
        DM gdm;
        char labelName[5] = "ghost";
        DMPlexConstructGhostCells(dm, labelName, NULL, &gdm);
        if (gdm) {
            PetscCall(DMDestroy(&dm));
            dm = gdm;
        }
//        PetscCall(DMDestroy(&gdm));
    }

    /** Now that the mesh has been created, get access to the geometry */
    Vec faceGeomVec = NULL;  //!< Vector used to describe the entire face geom of the dm.  This is constant and does not depend upon region.
    Vec cellGeomVec = NULL;
    PetscCall(DMPlexComputeGeometryFVM(dm, &cellGeomVec, &faceGeomVec));
    DM faceDM, cellDM;
    PetscCall(VecGetDM(cellGeomVec, &cellDM));
    PetscCall(VecGetDM(faceGeomVec, &faceDM));
    const PetscScalar *cellGeomArray;
    const PetscScalar *faceGeomArray;
    PetscCall(VecGetArrayRead(cellGeomVec, &cellGeomArray));
    PetscCall(VecGetArrayRead(faceGeomVec, &faceGeomArray));
    PetscFVFaceGeom *faceGeom;

    /** Get MPI information */
    PetscMPIInt rank;
    PetscMPIInt numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &rank);      //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);  //!< Get the number of ranks in the simulation.

    /** check to see if there is a ghost label */
    DMLabel ghostLabel;
    PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));

    PetscInt npoints = (rank == 0) ? 1
                                   : 0;  //!< Number of points to insert into the particle field. One particle for each ray.
    PetscInt dim; //! Get the number of dimensions in the domain
    PetscCall(DMGetDimension(dm, &dim));

    /** Create the DMSwarm */
    DM radsearch;  //!< DM which the search particles occupy
    PetscCall(DMCreate(MPI_COMM_WORLD, &radsearch));
    PetscCall(DMSetType(radsearch, DMSWARM));
    PetscCall(DMSetDimension(radsearch, dim));
    PetscCall(DMSwarmSetType(radsearch, DMSWARM_PIC)); //!< Configure radsearch to be of type PIC
    PetscCall(DMSwarmSetCellDM(radsearch, dm));

    /** Put a particle in the domain */
    PetscCall(DMSwarmFinalizeFieldRegister(radsearch));
    PetscCall(DMSwarmSetLocalSizes(radsearch, npoints, 0));

    /** Initialize the particle at the center of the domain */
    PetscReal *coord;                    //!< Pointer to the coordinate field information
    PetscCall(DMSwarmGetField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));
    coord[dim * 0 + 0] = 0.05; //!< Set the coordinates of the particle to the center of the domain
    coord[dim * 0 + 1] = 0.05;
    //    coord[dim * 0 + 3] = 0;
    PetscCall(DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));

    /** Get information about the dm swarm to constrain the loop condition */
    PetscInt nglobalpoints = 0;
    PetscCall(
            DMSwarmGetSize(radsearch, &nglobalpoints));  //!< Calculate the number of particles that are in the domain.
    PetscCall(
            DMSwarmGetLocalSize(radsearch, &npoints));   //!< Calculate the number of particles that are in the domain.

    /** Loop through the particle stepping routine while there are particles in the mesh */
    while (nglobalpoints != 0) {
        /** Now that the swarm has been created, get the fields back and read the properties of the particle out */
        PetscCall(DMSwarmGetField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));

        /** Read out the particle properties as it is being stepped through the domain */
        PetscInt index;
        Vec intersect;
        PetscCall(VecCreate(PETSC_COMM_SELF, &intersect));  //!< Instantiates the vector
        PetscCall(VecSetBlockSize(intersect, dim));
        PetscCall(VecSetSizes(intersect, PETSC_DECIDE, npoints * dim));  //!< Set size
        PetscCall(VecSetFromOptions(intersect));
        PetscInt i[3] = {0, 1, 2};                   //!< Establish the vector here so that it can be iterated.

        /** Loop through each particle in this rank to get the cell index associated with its location */
        for (PetscInt ip = 0; ip <
                              npoints; ip++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no
            //!< particles then pass out of initialization.

            /** Get the particle coordinates here and put them into the intersect */
            PetscReal position[3] = {(coord[dim * ip +
                                            0]),   //!< x component conversion from spherical coordinates, adding the position of the current cell
                                     (coord[dim * ip +
                                            1]),   //!< y component conversion from spherical coordinates, adding the position of the current cell
                                     (coord[dim * ip +
                                            2])};  //!< z component conversion from spherical coordinates, adding the position of the current cell

            /** This block creates the vector pointing to the cell whose index will be stored during the current loop */
            PetscCall(VecSetValues(intersect, dim, i, position,
                                   INSERT_VALUES));  //!< Actually input the values of the vector (There are 'dim' values to input)
            i[0] += dim;                                               //!< Iterate the index by the number of dimensions so that the DMLocatePoints function can be called collectively.
            i[1] += dim;
            i[2] += dim;
        }

        /** Get the cell index of the location that the particle is sitting on */
        PetscSF cellSF = NULL;  //!< PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
        PetscCall(DMLocatePoints(dm, intersect, DM_POINTLOCATION_NONE,
                                 &cellSF));  //!< Call DMLocatePoints here, all of the processes have to call it at once.

        /** An array that maps each point to its containing cell can be obtained with the below
         * We want to get a PetscInt index out of the DMLocatePoints function (cell[n].index)
         * */
        PetscInt nFound;
        const PetscInt *point = NULL;
        const PetscSFNode *cell = NULL;
        PetscCall(PetscSFGetGraph(cellSF, NULL, &nFound, &point,
                                  &cell));  //!< Using this to get the petsc int cell number from the struct (SF)

        /** Read out the information to console that is associated with this particle
         * The particle will inhabit ghost boundary cells where there is no access to face geometry
         * */
        for (PetscInt ip = 0; ip < npoints; ip++) {
            /** Print some basic information to console for  */
            printf("Global Points: %i\n", nglobalpoints);
            printf("Rank of Particle: %i\n", rank);
            printf("    %f %f\n", coord[0], coord[1]);
            if (nFound > -1 && cell[ip].index >= 0) { //!< Only for valid points
                printf("        Cell Index: %i\n",
                       cell[ip].index); //!< Print the properties of the particle

                /** March over each face on this cell in order to check them for the one which intersects this ray next */
                PetscInt numberFaces;
                const PetscInt *cellFaces;
                PetscCall(DMPlexGetConeSize(dm, cell[ip].index, &numberFaces));
                PetscCall(DMPlexGetCone(dm, cell[ip].index,
                                        &cellFaces));  //!< Get the face geometry associated with the current cell

                // make sure we are not working on a ghost cell
                PetscInt ghost = -1;
                if (ghostLabel) {
                    PetscCall(DMLabelGetValue(ghostLabel, cell[ip].index, &ghost));
                }
                if (ghost >= 0) {
                    printf("            Ghost label!\n"); //!< All of the cells with empty face geometry are labelled as ghost cells
                }

                /** Check every face for intersection with the segment.
                 * The segment with the shortest path length for intersection will be the one that physically intercepts with the cell face and not with the nonphysical plane beyond the face.
                 * */
                for (PetscInt f = 0; f < numberFaces; f++) {
                    PetscInt face = cellFaces[f];
                    PetscCall(DMPlexPointLocalRead(faceDM, face, faceGeomArray,
                                                   &faceGeom));  //!< Reads the cell location from the current cell
                    printf("                %f %f %f\n", faceGeom->normal[0], faceGeom->normal[1],
                           faceGeom->normal[2]); //!< If the face normal is empty then this indicates that the face geometry is invalid
                    if (faceGeom->normal[0] == 0 && faceGeom->normal[1] == 0 && faceGeom->normal[2] == 0) {
                        printf("                No face geometry!\n");
                    }
                }
            }
            /** Now that all of the particle properties have been read, move the coordinates of the particle over some */
            coord[dim * 0 + 0] += steplength;
            coord[dim * 0 + 1] += steplength;
        }
        /** Restore the field now that the position vector has been written to */
        PetscCall(DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));
        PetscCall(
                DMSwarmMigrate(radsearch, PETSC_TRUE)); //!< Migrate the particles out of the domain if they need to be

        PetscCall(VecDestroy(&intersect));   //!< Return the vector to PETSc
        PetscCall(PetscSFDestroy(&cellSF));  //!< Return the SF to PETSc
        PetscCall(DMSwarmGetSize(radsearch,
                                 &nglobalpoints));  //!< Update the loop condition. Recalculate the number of particles that are in the domain.
        PetscCall(DMSwarmGetLocalSize(radsearch,
                                      &npoints));   //!< Update the loop condition. Recalculate the number of particles that are in the domain.
    }

    /** Do some cleanup */
    PetscCall(DMDestroy(&radsearch));
    PetscCall(DMDestroy(&dm));
    PetscFinalize();
    return 0;
}



/*TEST

    args: -dm_refine 1
TEST*/
