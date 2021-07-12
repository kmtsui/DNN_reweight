cmake_minimum_required(VERSION 2.8)

project(DNN_reweight)

#Use the compilers found in the path
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

#Set this to TRUE to enable build debugging messages
set(BUILD_DEBUG_MSGS TRUE)
include(${CMAKE_SOURCE_DIR}/cmake/cmessage.cmake)

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Debug)
elseif(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

include_directories(${CMAKE_BINARY_DIR})

###############################  TENSORFLOW  ####################################
if ( NOT DEFINED ENV{TENSORFLOW_C} )
  cmessage (FATAL_ERROR "$TENSORFLOW_C  is not defined, please set up TENSORFLOW_C first.")
else()
  cmessage(STATUS "Using tensorflow library installed at $ENV{TENSORFLOW_C}")
  set(CMAKE_TENSORFLOW $ENV{TENSORFLOW_C})
endif()
find_library(TENSORFLOW_LIB tensorflow HINT ${CMAKE_TENSORFLOW}/lib)
find_library(TENSORFLOW_FRAMEWORK_LIB tensorflow_framework HINT ${CMAKE_TENSORFLOW}/lib)
cmessage(STATUS "TENSORFLOW_LIB: ${TENSORFLOW_LIB}")
cmessage(STATUS "TENSORFLOW_FRAMEWORK_LIB: ${TENSORFLOW_FRAMEWORK_LIB}")
link_directories("${CMAKE_TENSORFLOW}/lib/")

##################################  ROOT  ######################################
include(${CMAKE_SOURCE_DIR}/cmake/ROOTSetup.cmake)

################################## COMPILER ####################################
if (NOT DEFINED CMAKE_CXX_STANDARD OR "${CMAKE_CXX_STANDARD} " STREQUAL " ")
  SET(CMAKE_CXX_STANDARD 11)
endif()

cmessage(STATUS "CMAKE CXX Standard: ${CMAKE_CXX_STANDARD}")

if(${CMAKE_VERSION} VERSION_LESS "3.1.0")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

add_compile_options(-Wall -Wno-unused-variable -Wno-sign-compare -Wno-unused-function -Wno-unused-but-set-variable -Wno-reorder)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")


# Target
add_executable(DNN_reweight DNN_reweight.cc)
target_include_directories(DNN_reweight PUBLIC cppflow/include ${CMAKE_TENSORFLOW}/include)
target_compile_options(DNN_reweight PUBLIC ${ROOT_CXX_FLAGS}) 
target_link_libraries (DNN_reweight -Wl,--whole-archive "${TENSORFLOW_LIB}" "${TENSORFLOW_FRAMEWORK_LIB}" "${ROOT_LIBS}" -Wl,--no-whole-archive)
