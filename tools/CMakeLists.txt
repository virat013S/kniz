# Tools
#
# kniz-info answers "is this thing working, and how much of Vulkan is real?"  It
# is built against the Vulkan headers only and opens the loader with dlopen, for
# the same reason the test suite does: no loader development package, and it
# exercises the same path an application takes.

include(GNUInstallDirs)

add_executable(kniz-info kniz-info.cpp)

target_link_libraries(kniz-info PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
target_compile_options(kniz-info PRIVATE -Wall -Wextra)

if(KNIZ_WERROR)
  target_compile_options(kniz-info PRIVATE -Werror)
endif()

install(TARGETS kniz-info RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# The launcher is a plain shell script.  Installed beside kniz-info, it finds
# both the tool and the manifest (which lands in the library directory) by
# searching relative to its own location.
install(PROGRAMS "${CMAKE_SOURCE_DIR}/kniz"
  DESTINATION ${CMAKE_INSTALL_BINDIR})
