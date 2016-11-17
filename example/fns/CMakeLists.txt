cmake_minimum_required(VERSION 2.8)


if(NOT CPPTORCH_FOUND)
  find_package(cpptorch REQUIRED)
endif()


set(fns_SRC "fns.cpp")
add_executable(fns ${fns_SRC})
target_include_directories(fns PRIVATE ${CPPTORCH_INCLUDE_DIR})
target_link_libraries(fns ${CPPTORCH_LIBRARIES})

