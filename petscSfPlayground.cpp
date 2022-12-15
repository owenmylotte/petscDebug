static const char help[] = "Test overlapped communication on a single star forest (PetscSF)\n\n";

#include <petscvec.h>
#include <petscsf.h>
#include <petscviewer.h>
#include <vector>
#include <iostream>

struct SegmentData {
    PetscReal Ij = 0;    //!< Black body source for the segment. Make sure that this is reset every solve after the value has been transported.
    PetscReal Krad = 1;  //!< Absorption for the segment. Make sure that this is reset every solve after the value has been transported.
};

PetscInt SegmentDataRealSize = 2;

int main(int argc, char **argv)
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscMPIInt  rank, size;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    // setup the data type
    MPI_Datatype segmentDataType;
    MPI_Type_contiguous(2, MPIU_REAL, &segmentDataType);
    MPI_Type_commit(&segmentDataType);

    // create the sf
    PetscSF      sf;
    PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
    PetscCall(PetscSFSetFromOptions(sf));

//    // Create a vec to hold ray segment calculations
//    Vec localRaySegmentCalculations;
//
//    // Create an array that holds all ray information from neighbors
//    Vec raySummary;

    // Size these up
    PetscInt nleaves;
    PetscInt nroot;

    // assume the remote
    PetscSFNode *iremote;

    if(rank == 0){
        // assume that the zero rank did the work
        nroot =3;
        nleaves = 0;

        PetscCall(PetscMalloc1(nleaves, &iremote));
    }else{
        // assume that the zero rank did the work
        nroot =1;
        nleaves = 5;

        PetscCall(PetscMalloc1(nleaves, &iremote));
        iremote[0].rank  = 1;
        iremote[0].index = 0;
        iremote[1].rank  = 0;
        iremote[1].index = 0;
        iremote[2].rank  = 0;
        iremote[2].index = 1;
        iremote[3].rank  = 0;
        iremote[3].index = 2;
        iremote[4].rank  = 0;
        iremote[4].index = 2;
    }

    // create the sf
    PetscCall(PetscSFSetGraph(sf, nroot, nleaves, nullptr, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    PetscCall(PetscSFSetUp(sf));
    PetscCall(PetscSFView(sf, PETSC_VIEWER_STDOUT_WORLD));

    // create the data
//    PetscCall(VecCreate(PETSC_COMM_WORLD, &localRaySegmentCalculations));
//    PetscCall(VecSetSizes(localRaySegmentCalculations, SegmentDataRealSize*nroot, PETSC_DETERMINE));
//    PetscCall(VecSetFromOptions(localRaySegmentCalculations));
//    PetscCall(VecSetUp(localRaySegmentCalculations));
//
//    PetscCall(VecCreate(PETSC_COMM_WORLD, &raySummary));
//    PetscCall(VecSetSizes(raySummary, SegmentDataRealSize*nleaves, PETSC_DETERMINE));
//    PetscCall(VecSetFromOptions(raySummary));
//    PetscCall(VecSetUp(raySummary));
    std::vector<SegmentData> localRaySegmentCalculations(nroot);
    std::vector<SegmentData> raySummary(nleaves);


    for (PetscInt i = 0; i < nroot; i++) {
        localRaySegmentCalculations[i].Ij = (PetscScalar)(rank *100) + i;
        localRaySegmentCalculations[i].Krad = -(PetscScalar)(rank *100) - i;
    }

    // update the data
    // send from the root (localRaySegmentCalculations) to the leafs(raySummary)
    PetscCall(PetscSFBcastBegin(sf, segmentDataType, (const void *)localRaySegmentCalculations.data(), (void *)raySummary.data(), MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, segmentDataType, (const void *)localRaySegmentCalculations.data(), (void *)raySummary.data(), MPI_REPLACE));


    for(PetscMPIInt r =0; r < size; r++){
        if(r == rank){
            std::cout << "rank: " << r << std::endl;
            for (PetscInt i = 0; i < nroot; i++) {
                std::cout << i << " : " << localRaySegmentCalculations[i].Ij << ", " << localRaySegmentCalculations[i].Krad << std::endl;
            }
        }
        MPI_Barrier(PETSC_COMM_WORLD);
    }

    for(PetscMPIInt r =0; r < size; r++){
        if(r == rank){
            std::cout << "rank: " << r << std::endl;
            for (PetscInt i = 0; i < nleaves; i++) {
                std::cout << i << " : " << raySummary[i].Ij << ", " << raySummary[i].Krad << std::endl;
            }
        }
        MPI_Barrier(PETSC_COMM_WORLD);
    }


    PetscCall(PetscSFDestroy(&sf));

    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

   test:
      suffix: basic
      nsize: 2
      filter: grep -v "type" | grep -v "sort"
      args: -sf_type basic

   test:
      suffix: window
      nsize: 2
      filter: grep -v "type" | grep -v "sort"
      output_file: output/ex2_basic.out
      args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   # The nightly test suite with MPICH uses ch3:sock, which is broken when winsize == 0 in some of the processes
   test:
      suffix: window_shared
      nsize: 2
      filter: grep -v "type" | grep -v "sort"
      output_file: output/ex2_basic.out
      args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor shared
      requires: defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) !defined(PETSC_HAVE_MPICH_NUMVERSION) defined(PETSC_HAVE_MPI_ONE_SIDED)

TEST*/
