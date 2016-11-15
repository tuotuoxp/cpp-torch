cmake_minimum_required(VERSION 2.8)


if(UNIX)
  # add support for c++11
  add_definitions("-std=c++11")
endif()


if(NOT TH_FOUND)
  find_package(TH REQUIRED)
endif()


macro(generate_project name src)
  # create target
  add_library(${name} SHARED ${src})

  if(MSVC)
    target_compile_definitions(${name} PRIVATE API_CPPTORCH_DEF=1)
  endif()

  # include dirs
  target_include_directories(${name} PRIVATE "${TH_INCLUDE_DIR}" "${TH_INCLUDE_DIR}/../THNN" "${TH_INCLUDE_DIR}/..")

  # link
  find_library(TH_LIBRARY NAMES TH PATHS ${TH_LIB_DIR})
  find_library(THNN_LIBRARY NAMES THNN PATHS ${TH_LIB_DIR})
  target_link_libraries(${name} ${TH_LIBRARY} ${THNN_LIBRARY})

  # install target
  install(TARGETS ${name}
    RUNTIME DESTINATION "bin"
    ARCHIVE DESTINATION "lib"
    LIBRARY DESTINATION "lib"
  )
  set_target_properties(${name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set_target_properties(${name} PROPERTIES MACOSX_RPATH TRUE)
endmacro()


# cpptorch
file(GLOB_RECURSE cpptorch_src "include/*.h" "src/*.h" "src/*.h.inl" "src/*.cpp" "src/*.cc")
foreach(each_file ${cpptorch_src})
  get_filename_component(file_dir "${each_file}" PATH)
  # absolute path -> relative path
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" file_relative_dir "${file_dir}")

  # skip src or include directory
  string(REGEX REPLACE "(\\./)?(src)/?" "" group_name "${file_relative_dir}")
  # changes "/" to "\"
  string(REPLACE "/" "\\" group_name "${group_name}")

  # group into "Source Files" and "Header Files"
  if ("${each_file}" MATCHES ".*\\.cpp")
    set(group_name "Source Files\\${group_name}")
  elseif("${each_file}" MATCHES ".*\\.h")
    set(group_name "Header Files\\${group_name}")
  endif()

  source_group("${group_name}" FILES "${each_file}")
endforeach()
generate_project(cpptorch "${cpptorch_src}")


# cpptorch_cuda
if(WITH_CUDA)
  if(NOT CUDA_FOUND)
    find_package(CUDA 5.5 REQUIRED)
  endif()
  file(GLOB_RECURSE cpptorch_cuda_src "src_cuda/*.cpp" "src_cuda/*.cc")
  generate_project(cpptorch_cuda "${cpptorch_cuda_src}")
  target_include_directories(cpptorch_cuda PRIVATE ${CUDA_INCLUDE_DIRS} ${TH_INCLUDE_DIR}/../THC ${TH_INCLUDE_DIR}/../THCUNN)
  find_library(THC_LIBRARY NAMES THC PATHS ${TH_LIB_DIR})
  find_library(THCUNN_LIBRARY NAMES THCUNN PATHS ${TH_LIB_DIR})
  target_link_libraries(cpptorch_cuda cpptorch ${THC_LIBRARY} ${THCUNN_LIBRARY})
endif()


# install headers
file(GLOB_RECURSE cpptorch_headers "include/*.h")
foreach(file ${cpptorch_headers})
  get_filename_component(dir ${file} PATH)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/include" "" dir "${dir}")
  install(FILES ${file} DESTINATION "include/cpptorch/${dir}")
endforeach()


# create cmake file for find_package
set(CPPTORCH_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(CPPTORCH_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
configure_file(cpptorch.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/cpptorch-config.cmake")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/cpptorch-config.cmake"
  DESTINATION "share/cmake/cpptorch")


#############################################################


macro(generate_test_project name src)
  add_executable(${name} ${src})

  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(${name} cpptorch)

  set_target_properties(${name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set_target_properties(${name} PROPERTIES MACOSX_RPATH TRUE)

  install(TARGETS ${name} RUNTIME DESTINATION "bin")
endmacro()

generate_test_project(cpptorch_test "test/simple_test.cpp")

if(WITH_CUDA)
  generate_test_project(cpptorch_test_cuda "test/simple_test_cuda.cpp")
  target_link_libraries(cpptorch_test_cuda cpptorch cpptorch_cuda)
endif()
