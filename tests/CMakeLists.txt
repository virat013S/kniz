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

if(KNIZ_WERROR)
  target_compile_options(kniz_lifecycle PRIVATE -Werror)
  target_compile_options(kniz_compute PRIVATE -Werror)
  target_compile_options(kniz_image PRIVATE -Werror)
  target_compile_options(kniz_draw PRIVATE -Werror)
endif()

add_test(NAME lifecycle COMMAND kniz_lifecycle)
add_test(NAME compute
  COMMAND kniz_compute "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scale.comp.spv")
add_test(NAME image COMMAND kniz_image)
add_test(NAME draw
  COMMAND kniz_draw "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

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

set_tests_properties(lifecycle PROPERTIES ENVIRONMENT "${kniz_test_environment}")
set_tests_properties(compute PROPERTIES ENVIRONMENT "${kniz_test_environment}")
set_tests_properties(image PROPERTIES ENVIRONMENT "${kniz_test_environment}")
set_tests_properties(draw PROPERTIES ENVIRONMENT "${kniz_test_environment}")
