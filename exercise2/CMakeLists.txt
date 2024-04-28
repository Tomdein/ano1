add_executable(exercise2 main.cpp)

target_link_libraries(exercise2 ano-lib)

target_include_directories(exercise2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)