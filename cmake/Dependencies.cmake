# Dependency resolution.
#
# Kniz is intended to build on any distribution without a package-manager
# specific recipe.  Every dependency therefore follows the same rule:
#
#   1. try to use the distribution's own copy (find_package / pkg-config);
#   2. if that fails and KNIZ_FETCH_DEPS is ON, download a pinned revision.
#
# That keeps distro packaging cheap (system libs win) while making a bare
# `git clone && cmake && ninja` work on a machine with no dev packages at all.

include(FetchContent)

find_package(PkgConfig REQUIRED)

# ---------------------------------------------------------------------------
# Vulkan-Headers
#
# Provides the Vulkan API declarations and, importantly, vulkan/vk_icd.h which
# defines the loader<->ICD interface.  Header-only.
#
#   Debian/Ubuntu   libvulkan-dev          (no CMake config -> falls back)
#   Fedora          vulkan-headers
#   Arch            vulkan-headers
#   Alpine          vulkan-headers
# ---------------------------------------------------------------------------

find_package(VulkanHeaders CONFIG QUIET)

if(VulkanHeaders_FOUND)
  # CMake has no ${VAR:+alternate} form -- that is shell syntax, and writing it
  # here is a configure-time syntax error on precisely those distributions whose
  # VulkanHeaders package is found (Fedora, Arch), while looking fine on any
  # distribution that falls through to FetchContent.
  if(VulkanHeaders_VERSION)
    set(KNIZ_VULKAN_HEADERS_SOURCE "system (${VulkanHeaders_VERSION})")
  else()
    set(KNIZ_VULKAN_HEADERS_SOURCE "system")
  endif()
elseif(KNIZ_FETCH_DEPS)
  set(KNIZ_VULKAN_HEADERS_SOURCE "fetched (${KNIZ_VULKAN_HEADERS_TAG})")

  FetchContent_Declare(Vulkan-Headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        ${KNIZ_VULKAN_HEADERS_TAG}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE)
  FetchContent_MakeAvailable(Vulkan-Headers)
else()
  message(FATAL_ERROR
    "Vulkan-Headers not found and KNIZ_FETCH_DEPS=OFF.\n"
    "Install your distribution's Vulkan development headers, or configure with "
    "-DKNIZ_FETCH_DEPS=ON.")
endif()

# ---------------------------------------------------------------------------
# EGL
#
# Used to create the OpenGL context that Kniz drives.  We deliberately do not
# link against libGL: every GL entry point is resolved through eglGetProcAddress
# at runtime (see src/icd/kniz_gl.cpp), which avoids binding to a specific GL
# loader or ABI at link time.
# ---------------------------------------------------------------------------

pkg_check_modules(EGL REQUIRED IMPORTED_TARGET egl)

# ---------------------------------------------------------------------------
# SPIRV-Cross: SPIR-V -> GLSL
#
# Kniz receives shaders as SPIR-V and has to hand OpenGL GLSL, so this is not
# optional.
#
# This is the one dependency where the distribution's copy is often *not* the
# useful one.  Arch and Fedora build SPIRV-Cross with SPIRV_CROSS_SHARED=ON,
# which produces a shared object exporting only the C API and installs no CMake
# package, so the C++ interface used here is simply absent.  The system tier is
# still tried first -- some distributions do install the static C++ targets --
# and the fetch tier is what usually runs.
#
# Arch ships the C API as `spirv-cross`, which cannot serve this purpose; the
# pinned tag is kept in step with KNIZ_VULKAN_HEADERS_TAG because SPIRV-Cross
# releases track Vulkan SDK versions.
# ---------------------------------------------------------------------------

set(KNIZ_SPIRV_CROSS_TAG "vulkan-sdk-1.4.363.0" CACHE STRING
  "SPIRV-Cross git tag used when fetching")

find_package(spirv_cross_glsl CONFIG QUIET)

if(TARGET spirv-cross-glsl)
  set(KNIZ_SPIRV_CROSS_SOURCE "system")
elseif(KNIZ_FETCH_DEPS)
  set(KNIZ_SPIRV_CROSS_SOURCE "fetched (${KNIZ_SPIRV_CROSS_TAG})")

  # Only the GLSL backend is wanted.  Everything else would be dead weight in
  # the build, and the test suite in particular pulls in Python and glslang.
  set(SPIRV_CROSS_STATIC     ON  CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_SHARED     OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_CLI        OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_TESTS   OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_GLSL    ON  CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_HLSL    OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_MSL     OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_CPP     OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_C_API   OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_UTIL    OFF CACHE BOOL "" FORCE)
  # Without this it installs itself into our prefix as a side effect.
  set(SPIRV_CROSS_SKIP_INSTALL   ON  CACHE BOOL "" FORCE)

  FetchContent_Declare(SPIRV-Cross
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
    GIT_TAG        ${KNIZ_SPIRV_CROSS_TAG}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE)
  FetchContent_MakeAvailable(SPIRV-Cross)
else()
  message(FATAL_ERROR
    "SPIRV-Cross (the C++ GLSL backend) was not found and KNIZ_FETCH_DEPS=OFF.\n"
    "Kniz needs the spirv-cross-glsl target; the C API alone is not enough.\n"
    "Install a package that provides it, or configure with -DKNIZ_FETCH_DEPS=ON.")
endif()

# The fetched build exposes its headers at the repository root, while an
# installed copy puts them under include/spirv_cross.  Whichever tier ran, name
# the result consistently and let the code handle both include layouts.
if(NOT TARGET Kniz::spirv-cross-glsl)
  add_library(Kniz::spirv-cross-glsl ALIAS spirv-cross-glsl)
endif()
