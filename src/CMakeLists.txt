add_library(kniz SHARED
  icd/kniz_common.cpp
  icd/kniz_gl.cpp
  icd/kniz_icd.cpp
  icd/kniz_device.cpp
  icd/kniz_shader.cpp
  icd/kniz_memory.cpp
  icd/kniz_image.cpp
  icd/kniz_descriptor.cpp
  icd/kniz_pipeline.cpp
  icd/kniz_render.cpp
  icd/kniz_swapchain.cpp
  icd/kniz_command.cpp)

target_include_directories(kniz PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/icd")
target_link_libraries(kniz PRIVATE
  Vulkan::Headers
  PkgConfig::EGL
  Kniz::spirv-cross-glsl)

target_compile_definitions(kniz PRIVATE
  KNIZ_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  KNIZ_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  KNIZ_VERSION_PATCH=${PROJECT_VERSION_PATCH})

# The manifest at icd/kniz_icd.json expects a library called libkniz.so next to
# it, which is what the default naming gives us.
set_target_properties(kniz PROPERTIES
  OUTPUT_NAME kniz
  VERSION     ${PROJECT_VERSION}
  SOVERSION   ${PROJECT_VERSION_MAJOR}
  # Internal symbols must not be visible.  See the header comment in
  # icd/kniz_common.h for why this is load-bearing rather than cosmetic.
  CXX_VISIBILITY_PRESET      hidden
  VISIBILITY_INLINES_HIDDEN  ON)

if(KNIZ_WERROR)
  if(MSVC)
    target_compile_options(kniz PRIVATE /W4 /WX)
  else()
    target_compile_options(kniz PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
else()
  if(NOT MSVC)
    target_compile_options(kniz PRIVATE -Wall -Wextra)
  endif()
endif()

# ---------------------------------------------------------------------------
# Export control
#
# A Vulkan ICD shares an address space with the application and possibly with
# the loader.  Exporting anything other than the vk* entry points risks symbol
# clashes -- the loader specification calls this out explicitly for ICDs that
# also link OpenGL.  The version script restricts the dynamic symbol table to
# exactly the vk* namespace.
#
# Only meaningful for ELF targets; other platforms get default visibility.
# ---------------------------------------------------------------------------

include(CheckLinkerFlag)
check_linker_flag(CXX "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/kniz.map"
  KNIZ_LINKER_HAS_VERSION_SCRIPT)

if(KNIZ_LINKER_HAS_VERSION_SCRIPT)
  target_link_options(kniz PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/kniz.map")
  set_target_properties(kniz PROPERTIES LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/kniz.map")
else()
  message(STATUS "kniz: version script unsupported by linker; skipping export control")
endif()

# Refuse to emit a shared object with unresolved symbols.  Without this, the
# visibility/version-script combination above can produce a library that links
# successfully and then fails to load, which is the worst of both worlds.
if(UNIX AND NOT APPLE)
  target_link_options(kniz PRIVATE "-Wl,-z,defs")
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS kniz
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# The manifest points at the library by relative path, so they must be
# installed into the same directory to be usable via
# VK_ICD_FILENAMES=<dir>/kniz_icd.json .
install(FILES "${CMAKE_SOURCE_DIR}/icd/kniz_icd.json"
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Mirror that arrangement in the build tree.  The loader resolves library_path
# relative to the manifest's own directory, so placing the manifest next to the
# freshly built library is what makes
#   VK_ICD_FILENAMES=<build>/src/kniz_icd.json vulkaninfo
# work without installing first.  The test suite relies on it.
configure_file("${CMAKE_SOURCE_DIR}/icd/kniz_icd.json"
  "${CMAKE_CURRENT_BINARY_DIR}/kniz_icd.json" COPYONLY)
