cmake_minimum_required(VERSION 2.8)


macro(generate_test_project name src)
  add_executable(${name} ${src})

  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
  target_link_libraries(${name} cpptorch)

  set_target_properties(${name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set_target_properties(${name} PROPERTIES MACOSX_RPATH TRUE)
endmacro()

generate_test_project(cpptorch_tester "tester.cpp")

if(BUILD_CUDA)
  generate_test_project(cpptorch_tester_cuda "tester_cuda.cpp")
  target_link_libraries(cpptorch_tester_cuda cpptorch cpptorch_cuda)
endif()

