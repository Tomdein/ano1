include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(ano-lib floodfill.cpp color-generator.cpp moments.cpp detected-object.cpp ethalons.cpp)
