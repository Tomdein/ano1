include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(ano-bpnn backprop.cpp)

add_executable(bpnn-test ns_test.cpp)
target_link_libraries(bpnn-test ano-bpnn)
target_include_directories(exercise1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
