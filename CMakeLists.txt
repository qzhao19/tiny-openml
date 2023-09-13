cmake_minimum_required(VERSION 3.2)
project(tiny-openml)

set(CMAKE_CXX_STANDARD 17)

# Find OpenMP
IF (APPLE)
    message(STATUS "clang does not support OpenMP!")
ELSEIF (WIN32 OR UNIX)
    find_package(OpenMP REQUIRED)
    IF(OPENMP_FOUND)
        message(STATUS "OpenMP Found!")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ENDIF()
ENDIF ()

# Libkdtree source files