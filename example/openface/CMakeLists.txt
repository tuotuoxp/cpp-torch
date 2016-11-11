CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(TestOpenFace)

FIND_PACKAGE(OpenCV REQUIRED)

find_package(cpptorch REQUIRED)


#ADD_DEFINITIONS(-mavx)
ADD_DEFINITIONS(-msse4)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

add_executable(test_openface main.cpp)
target_include_directories(test_openface PRIVATE ${CPPTORCH_INCLUDE_DIR})
target_link_libraries(test_openface ${OpenCV_LIBS} ${CPPTORCH_LIBRARIES})
