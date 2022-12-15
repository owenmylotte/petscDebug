cmake_minimum_required(VERSION 3.18.4)
find_package(PkgConfig REQUIRED)

# Check if a C compiler is explicitly stated
if (NOT DEFINED CMAKE_C_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(PETSC_C_COMPILER PETSc ccompiler)
    set(CMAKE_C_COMPILER ${PETSC_C_COMPILER})
endif ()

# Check if a CXX compiler is explicitly stated
if (NOT DEFINED CMAKE_CXX_COMPILER)
    # Set the compilers based upon the PETSc package
    pkg_get_variable(PETSC_CXX_COMPILER PETSc cxxcompiler)
    set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})
endif ()

# Set the project details
project(petscTest VERSION 0.8.30)

# Load the Required 3rd Party Libaries
pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET GLOBAL PETSc)

add_executable(example example.c)
target_link_libraries(example PUBLIC         PkgConfig::PETSc)

add_executable(particlesInDm particlesInDm.c)
target_link_libraries(particlesInDm PUBLIC         PkgConfig::PETSc)

add_executable(dmLocate dmLocate.c)
target_link_libraries(dmLocate PUBLIC         PkgConfig::PETSc)

add_executable(petscSf petscSf.c)
target_link_libraries(petscSf PUBLIC         PkgConfig::PETSc)

add_executable(petscSfPlayground petscSfPlayground.cpp)
target_link_libraries(petscSfPlayground PUBLIC         PkgConfig::PETSc)


#
#add_executable(gMshDistribute gmshDistribute.c)
#target_link_libraries(gMshDistribute PUBLIC         PkgConfig::PETSc)
#
#add_executable(gMshExtrude gmshExtrude.c)
#target_link_libraries(gMshExtrude PUBLIC         PkgConfig::PETSc)
#
#add_executable(gmshIO gmshIO.c)
#target_link_libraries(gmshIO PUBLIC         PkgConfig::PETSc)
#
#add_executable(subDmTest subDmTest.c)
#target_link_libraries(subDmTest PUBLIC         PkgConfig::PETSc)
#
#add_executable(gmshAreaCheck gmshAreaCheck.c)
#target_link_libraries(gmshAreaCheck PUBLIC         PkgConfig::PETSc)
#
#add_executable(periodicMeshOutput periodicMeshOutput.c)
#target_link_libraries(periodicMeshOutput PUBLIC         PkgConfig::PETSc)
#
#
#
## XDMF
#add_executable(xdmfGen xdmfGen.c)
#target_link_libraries(xdmfGen PUBLIC         PkgConfig::PETSc)

# Include system specific hacks
#if ("${APPLE}" AND (${CMAKE_C_COMPILER_ID} STREQUAL "AppleClang"))
    # check for homebrew gfortran and get path for libstdc++.dylib
    execute_process(COMMAND gfortran --print-file-name=libstdc++.dylib OUTPUT_VARIABLE LIBSTDCPP_PATH)

    # convert to an absolute path and get the directory
    get_filename_component(LIBSTDCPP_PATH ${LIBSTDCPP_PATH} ABSOLUTE)
    get_filename_component(LIBSTDCPP_PATH ${LIBSTDCPP_PATH} DIRECTORY)

    target_link_directories(particlesInDm PUBLIC ${LIBSTDCPP_PATH})
    target_link_directories(example PUBLIC ${LIBSTDCPP_PATH})
    target_link_directories(dmLocate PUBLIC ${LIBSTDCPP_PATH})
    target_link_directories(petscSf PUBLIC ${LIBSTDCPP_PATH})
    target_link_directories(petscSfPlayground PUBLIC ${LIBSTDCPP_PATH})

#endif ()
