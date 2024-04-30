add_executable(exercise5 main.cpp)

target_link_libraries(exercise5 ano-lib)
target_link_libraries(exercise5 ano-bpnn)

target_include_directories(exercise5 PRIVATE ./include)
