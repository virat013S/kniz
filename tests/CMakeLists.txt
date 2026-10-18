# Tests
#
# The test harness drives the ICD through the real Vulkan loader (see
# lifecycle.cpp for why), opening it with dlopen.  The only build requirement is
# the Vulkan headers, which the main build already provides.

add_executable(kniz_lifecycle lifecycle.cpp)

target_link_libraries(kniz_lifecycle PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz_lifecycle PRIVATE -Wall -Wextra)

if(KNIZ_WERROR)
  target_compile_options(kniz_lifecycle PRIVATE -Werror)
endif()

add_test(NAME lifecycle COMMAND kniz_lifecycle)

# The loader resolves the manifest's library_path relative to the manifest's own
# directory, so build/src is where both the library and its manifest live:
# choosing the manifest there makes VK_ICD_FILENAMES work straight from the
# build tree, with no install step.
set_tests_properties(lifecycle PROPERTIES
  ENVIRONMENT "VK_ICD_FILENAMES=${CMAKE_BINARY_DIR}/src/kniz_icd.json;KNIZ_LOG=2")
