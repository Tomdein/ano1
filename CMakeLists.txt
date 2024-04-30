cmake_minimum_required(VERSION 3.22)

project(ano1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

unset(CUDA_ARCH_PTX)
set(CUDA_ARCH_BIN "7.5")

# These two are equivalent:
# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build)
# find_package(OpenCV REQUIRED)
# or:
find_package(OpenCV REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build)

# Links all folowing executables and libs with these specified libraries
link_libraries(${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/include)

add_subdirectory(lib)

add_subdirectory(exercise1)
add_subdirectory(exercise2)
add_subdirectory(exercise3)
add_subdirectory(exercise4)

add_subdirectory(bpnn)
include_directories(./bpnn/include)
