add_executable(exercise1 main.cpp)

target_link_libraries(exercise1 ano-lib)

target_include_directories(exercise1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)