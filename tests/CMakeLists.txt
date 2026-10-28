# Tests
#
# The test harness drives the ICD through the real Vulkan loader (see
# lifecycle.cpp for why), opening it with dlopen.  The only build requirement is
# the Vulkan headers, which the main build already provides.

add_executable(kniz_lifecycle lifecycle.cpp)

target_link_libraries(kniz_lifecycle PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_lifecycle PRIVATE -Wall -Wextra)

# The M1 milestone test: a compute dispatch whose output is compared against a
# CPU reference.  It reads its shader from a file, so the path is passed as an
# argument rather than baked into the binary.
add_executable(kniz_compute compute.cpp)

target_link_libraries(kniz_compute PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_compute PRIVATE -Wall -Wextra)

# The M2a milestone test: images, and the data movement to and from them.  It
# needs no shader, because a clear and a copy are the whole of what it exercises.
add_executable(kniz_image image.cpp)

target_link_libraries(kniz_image PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_image PRIVATE -Wall -Wextra)

# The M2b milestone test: drawing.  It needs four shaders -- a triangle through a
# vertex buffer and the same triangle asked for its fragments' own coordinates --
# so the directory holding them is passed as an argument, as with the compute
# test.
add_executable(kniz_draw draw.cpp)

target_link_libraries(kniz_draw PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_draw PRIVATE -Wall -Wextra)

# The M2c milestone test: uniform buffers and push constants, in both a compute
# and a graphics pipeline.  It needs its own shaders for the same reason the draw
# test does, and shares the shaders directory argument with it.
add_executable(kniz_uniform uniform.cpp)

target_link_libraries(kniz_uniform PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_uniform PRIVATE -Wall -Wextra)

# The M2d milestone test: samplers and textures, in both a compute and a graphics
# pipeline.  It needs four shaders, so it takes the shaders directory too.
add_executable(kniz_sampler sampler.cpp)

target_link_libraries(kniz_sampler PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_sampler PRIVATE -Wall -Wextra)

# The M3 milestone test: presentation.  It needs a window, which it makes itself
# through libX11 opened with dlopen -- so the build gains no X11 dependency, and a
# machine with no display skips the test rather than failing it.
add_executable(kniz_present present.cpp)

target_link_libraries(kniz_present PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_present PRIVATE -Wall -Wextra)

# The M3b milestone test: depth attachments, the depth test and depth writes.  It
# needs two shaders -- one triangle drawn at whatever depth a push constant names
# -- so it takes the shaders directory too.
add_executable(kniz_depth depth.cpp)

target_link_libraries(kniz_depth PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_depth PRIVATE -Wall -Wextra)

if(KNIZ_WERROR)
  target_compile_options(kniz_lifecycle PRIVATE -Werror)
  target_compile_options(kniz_compute PRIVATE -Werror)
  target_compile_options(kniz_image PRIVATE -Werror)
  target_compile_options(kniz_draw PRIVATE -Werror)
  target_compile_options(kniz_uniform PRIVATE -Werror)
  target_compile_options(kniz_sampler PRIVATE -Werror)
  target_compile_options(kniz_present PRIVATE -Werror)
  target_compile_options(kniz_depth PRIVATE -Werror)
endif()

add_test(NAME lifecycle COMMAND kniz_lifecycle)
add_test(NAME compute
  COMMAND kniz_compute "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scale.comp.spv")
add_test(NAME image COMMAND kniz_image)
add_test(NAME draw
  COMMAND kniz_draw "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
add_test(NAME uniform
  COMMAND kniz_uniform "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
add_test(NAME sampler
  COMMAND kniz_sampler "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
add_test(NAME present
  COMMAND kniz_present "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
add_test(NAME depth
  COMMAND kniz_depth "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

# The loader resolves the manifest's library_path relative to the manifest's own
# directory, so build/src is where both the library and its manifest live:
# choosing the manifest there makes the selection work straight from the build
# tree, with no install step.
#
# Both variables are set on purpose.  VK_ICD_FILENAMES is the name the loader has
# always used; VK_DRIVER_FILES is the newer one, and when both are present the
# loader prefers VK_DRIVER_FILES and ignores VK_ICD_FILENAMES entirely.  A shell
# that exports VK_DRIVER_FILES -- which is how a machine pins one vendor's driver
# -- would otherwise quietly redirect every test to the system's own Vulkan
# driver, where the tests would either fail or, worse, pass while proving nothing
# about Kniz.
set(kniz_test_environment
  "VK_ICD_FILENAMES=${CMAKE_BINARY_DIR}/src/kniz_icd.json;VK_DRIVER_FILES=${CMAKE_BINARY_DIR}/src/kniz_icd.json;KNIZ_LOG=2")

# Applied to every test in this directory rather than listed one by one, because
# listing them is a footgun that has already fired: the present test was added to
# add_test() and not to the list here, so it ran against the machine's own Vulkan
# driver.  It passed on a developer machine whose shell exported these variables
# and failed in CI, which is the worst way round.  Asking the directory for its
# tests means a new test cannot be forgotten.
get_property(kniz_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${kniz_tests} PROPERTIES ENVIRONMENT "${kniz_test_environment}")
