cmake_minimum_required(VERSION 3.20)

project(kniz
  VERSION 0.0.1
  DESCRIPTION "A Vulkan ICD implemented on top of OpenGL"
  HOMEPAGE_URL "https://github.com/virat013S/kniz"
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

option(KNIZ_FETCH_DEPS
  "Download dependencies with FetchContent when no system copy is found" ON)
option(KNIZ_BUILD_TESTS "Build the test suite" ON)
option(KNIZ_WERROR "Treat warnings as errors" OFF)

set(KNIZ_VULKAN_HEADERS_TAG "v1.4.363" CACHE STRING
  "Vulkan-Headers git tag used when fetching")

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Dependencies)

add_subdirectory(src)

if(KNIZ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

message(STATUS "")
message(STATUS "kniz ${PROJECT_VERSION}")
message(STATUS "  build type .......... ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler ............ ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Vulkan-Headers ...... ${KNIZ_VULKAN_HEADERS_SOURCE}")
message(STATUS "  FetchContent ........ ${KNIZ_FETCH_DEPS}")
message(STATUS "")
