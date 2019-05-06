cmake_minimum_required(VERSION 3.1)

project(libutils)

set(HEADERS
        libutils/fast_random.h
        libutils/misc.h
        libutils/string_utils.h
        libutils/thread_mutex.h
        libutils/timer.h
        )

set(SOURCES
        libutils/misc.cpp
        libutils/string_utils.cpp
        libutils/thread_mutex.cpp
        )

set(CMAKE_CXX_STANDARD 11)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} Threads::Threads libgpu)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
