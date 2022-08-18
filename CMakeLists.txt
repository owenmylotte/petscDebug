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

add_executable(gMshDistribute gmshDistribute.c)
target_link_libraries(gMshDistribute PUBLIC         PkgConfig::PETSc)

add_executable(gMshExtrude gmshExtrude.c)
target_link_libraries(gMshExtrude PUBLIC         PkgConfig::PETSc)

add_executable(gmshIO gmshIO.c)
target_link_libraries(gmshIO PUBLIC         PkgConfig::PETSc)

add_executable(subDmTest subDmTest.c)
target_link_libraries(subDmTest PUBLIC         PkgConfig::PETSc)

add_executable(gmshAreaCheck gmshAreaCheck.c)
target_link_libraries(gmshAreaCheck PUBLIC         PkgConfig::PETSc)