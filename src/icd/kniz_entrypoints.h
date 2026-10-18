#pragma once

// Entry point resolution.
//
// The loader asks us for two tables:
//
//   vk_icdGetInstanceProcAddr     global and instance level commands
//   vkGetDeviceProcAddr           device level commands
//
// We keep a single flat table of everything we implement and let each entry
// point search it.  That is deliberately more permissive than the specification
// requires -- device level commands are reachable through the instance table --
// because applications and tooling in practice call vkGetInstanceProcAddr for
// device commands and expect a usable pointer.  Real ICDs behave the same way.

#include <vulkan/vulkan.h>

namespace kniz {

// Returns nullptr when the command is not implemented.
PFN_vkVoidFunction resolve_instance_proc(const char* name);
PFN_vkVoidFunction resolve_device_proc(const char* name);

}  // namespace kniz
