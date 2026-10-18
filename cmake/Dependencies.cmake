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
  set(KNIZ_VULKAN_HEADERS_SOURCE "system${VulkanHeaders_VERSION:+ (${VulkanHeaders_VERSION})}")
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
