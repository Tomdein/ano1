add_executable(exercise3 main.cpp)

target_link_libraries(exercise3 ano-lib)

target_include_directories(exercise3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
