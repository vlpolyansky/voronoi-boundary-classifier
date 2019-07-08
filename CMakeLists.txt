cmake_minimum_required(VERSION 3.5)
project(VoronoiClassifier)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(libs)

INCLUDE(CheckCXXSourceCompiles)

include_directories(${CMAKE_SOURCE_DIR}/include)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_subdirectory(cpp)
