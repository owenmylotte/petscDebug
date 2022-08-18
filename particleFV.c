static const char help[] = "Testing particle access to face geometry in parallel";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscsf.h>

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    PetscInt dimensions = 2;
    PetscInt faces[2] = {10, 10};
    PetscReal lower[2] = {0, 0};
    PetscReal upper[2] = {1, 1};
    DMBoundaryType bc[2] = {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC};

    DM dm;
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dimensions, PETSC_FALSE, faces, lower, upper, bc, PETSC_TRUE, &dm));
    PetscCall(DMSetFromOptions(dm));

    Vec faceGeomVec = NULL;  //!< Vector used to describe the entire face geom of the dm.  This is constant and does not depend upon region.
    Vec cellGeomVec = NULL;
    DM faceDM;
    const PetscScalar *faceGeomArray;
    PetscFVFaceGeom *faceGeom;

    PetscReal minCellRadius;
    PetscCall(DMPlexGetMinRadius(dm,
                                 &minCellRadius));                    //!< The minimum cell radius is used to scale the face stepping procedures
    PetscCall(DMPlexComputeGeometryFVM(dm, &cellGeomVec, &faceGeomVec));  //!< Get the geometry vectors
    PetscCall(VecGetDM(faceGeomVec, &faceDM));
    PetscCall(VecGetArrayRead(faceGeomVec, &faceGeomArray));

    /** Get MPI information */
    PetscMPIInt rank;
    PetscMPIInt numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &rank);      //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);  //!< Get the number of ranks in the simulation.

    /** check to see if there is a ghost label */
    DMLabel ghostLabel;
    PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));

    PetscInt npoints = 1;  //!< Number of points to insert into the particle field. One particle for each ray.
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
    coord[dim * 0 + 0] = 0.5; //!< Set the coordinates of the particle to the center of the domain
    coord[dim * 0 + 1] = 0.5;
    //    coord[dim * 0 + 3] = 0;
    PetscCall(DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));

    PetscInt nglobalpoints = 0;
    PetscCall(
            DMSwarmGetSize(radsearch, &nglobalpoints));  //!< Calculate the number of particles that are in the domain.
    PetscCall(
            DMSwarmGetLocalSize(radsearch, &npoints));   //!< Calculate the number of particles that are in the domain.


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
        for (PetscInt ip = 0; ip <
                              npoints; ip++) {  //!< Iterate over the particles present in the domain. How to isolate the particles in this domain and iterate over them? If there are no
            //!< particles then pass out of initialization.

            /** FIRST TAKE THIS LOCATION INTO THE RAYS VECTOR
             * "I found a particle in my domain. Maybe it was just moved here and I've never seen it before.
             * Therefore, my first step should be to add this location to the local rays vector. Then I can adjust the coordinates and migrate the particle." */

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

        /** Restore the field now that the position vector has been written to */
        //        PetscCall(DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));

        /** Loop through points to try to get the cell that is sitting on that point*/
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

        for (PetscInt ip = 0; ip < npoints; ip++) {
            printf("Global Points: %i\n", nglobalpoints);
            if (nFound > -1 && cell[ip].index >= 0) { //!< Only for valid points
                printf("    %f %f    Cell Index: %i\n", coord[dim * ip + 0], coord[dim * ip + 1],
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
                    printf("        Ghost label!\n"); //!< All of the cells with empty face geometry are labelled as ghost cells
                }

                /** Check every face for intersection with the segment.
                 * The segment with the shortest path length for intersection will be the one that physically intercepts with the cell face and not with the nonphysical plane beyond the face.
                 * */
                for (PetscInt f = 0; f < numberFaces; f++) {
                    PetscInt face = cellFaces[f];
                    PetscCall(DMPlexPointLocalRead(faceDM, face, faceGeomArray,
                                                   &faceGeom));  //!< Reads the cell location from the current cell
                    if (faceGeom->normal[0] == 0 && faceGeom->normal[1] == 0 && faceGeom->normal[2] == 0) {
                        printf("            Empty face normal!\n");
                    }
                }

                /** Now that all of the particle properties have been read, move the coordinates of the particle over some */
                coord[dim * 0 + 1] += 0.05;
                //            coord[dim * 0 + 2] += 0.05;
                //    coord[dim * 0 + 3] += 0;
            } else {
                /** Delete the particle if it has entered an invalid region */
                PetscCall(DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));
                PetscCall(DMSwarmRemovePointAtIndex(radsearch, ip));  //!< Delete the particle!
                PetscCall(DMSwarmGetLocalSize(radsearch, &npoints));
                PetscCall(DMSwarmGetField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));
                ip--;
            }
        }
        /** Restore the field now that the position vector has been written to */
        PetscCall(DMSwarmRestoreField(radsearch, DMSwarmPICField_coor, NULL, NULL, (void **) &coord));

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
