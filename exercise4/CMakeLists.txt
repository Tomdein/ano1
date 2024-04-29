add_executable(exercise4 main.cpp)

target_link_libraries(exercise4 ano-lib)

target_include_directories(exercise4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
