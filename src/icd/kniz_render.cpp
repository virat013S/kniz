#include "kniz_render.h"

#include "kniz_common.h"
#include "kniz_descriptor.h"
#include "kniz_image.h"
#include "kniz_memory.h"

#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace kniz {
namespace {

// VkClearValue is a union whose largest member is a colour, so every clear value
// in an array is this many bytes apart regardless of which member is live.  The
// recorded copy is a flat byte array for exactly that reason: at record time
// nothing knows whether the attachment is a colour or a depth target.
constexpr size_t kClearValueBytes = sizeof(VkClearValue);

bool device_can_draw(Device* device) {
  if (!device->instance->gl_ready) {
    KNIZ_ERR("no OpenGL context is available");
    return false;
  }
  if (!device->instance->gl.render_capable) {
    KNIZ_ERR("drawing needs framebuffer objects and vertex array objects, which "
             "this OpenGL context (version %d.%d) does not provide",
             device->instance->gl.major, device->instance->gl.minor);
    return false;
  }
  return true;
}

gl::Enum index_type_to_gl(VkIndexType type, bool* ok) {
  *ok = true;
  switch (type) {
    case VK_INDEX_TYPE_UINT16:
      return gl::kUnsignedShort;
    case VK_INDEX_TYPE_UINT32:
      return gl::kUnsignedInt;
    default:
      *ok = false;
      return 0;
  }
}

// The attachment index a subpass attachment refers to, given a GL draw buffer
// slot.  Returns false when GL_COLOR_ATTACHMENTslot is not one the subpass
// writes, which is the case that cannot be cleared because it is not in the
// draw buffer list.
bool subpass_slot_for_attachment(const RenderPass* pass, uint32_t attachment,
                                 uint32_t* slot) {
  for (size_t i = 0; i < pass->color_attachments.size(); ++i) {
    if (pass->color_attachments[i] == attachment) {
      *slot = static_cast<uint32_t>(i);
      return true;
    }
  }
  return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

void render_begin_pass(const RecordedCommand& command) {
  Framebuffer* framebuffer = command.framebuffer;
  if (framebuffer == nullptr)
    return;
  const RenderPass* pass = framebuffer->render_pass;

  gl::f.BindFramebuffer(gl::kFramebuffer, framebuffer->gl_framebuffer);

  // The load-op clear covers the render area and nothing else, which is the same
  // region the scissor will confine the draws to, so setting the scissor to the
  // render area here gets the clear right without a second pass over the
  // attachments.
  //
  // glClearBufferfv obeys the scissor test but not the viewport, the colour write
  // mask or dithering (GL 4.6, 17.4.3.3), so the clear covers exactly the scissor
  // rectangle whatever state the previous pass left behind.  The viewport is set
  // as well only so that the state a pass begins from is the state it describes.
  const VkRect2D& area = command.render_area;
  gl::f.Viewport(static_cast<gl::Int>(area.offset.x),
                 static_cast<gl::Int>(area.offset.y),
                 static_cast<gl::Size>(area.extent.width),
                 static_cast<gl::Size>(area.extent.height));
  gl::f.Scissor(static_cast<gl::Int>(area.offset.x),
                static_cast<gl::Int>(area.offset.y),
                static_cast<gl::Size>(area.extent.width),
                static_cast<gl::Size>(area.extent.height));
  gl::f.Enable(gl::kScissorTest);

  // A Vulkan load-op clear is not pipeline state: it happens before any pipeline
  // is in effect, so the pipeline's colour write mask cannot suppress or restrict
  // it.  GL has no such clean separation, and whether glClearBufferfv consults the
  // write mask is not something to rely on either way -- Gallium carries a
  // capability flag for exactly this ambiguity, because the hardware underneath
  // differs.  Setting the mask here costs one call per pass and makes the clear
  // independent of the answer.  Dithering is switched off for a related reason:
  // Vulkan has no dithering at all, and it perturbs a clear value in its least
  // significant bit rather than writing it exactly.
  //
  // Both settings stay in force for the pass, which is what draws want too: an
  // identity mask until a pipeline sets its own, which glClearBufferfv aside,
  // every draw does.
  gl::f.ColorMask(1, 1, 1, 1);
  gl::f.Disable(gl::kDither);

  for (size_t slot = 0; slot < pass->color_attachments.size(); ++slot) {
    const uint32_t attachment = pass->color_attachments[slot];
    if (attachment >= pass->attachments.size())
      continue;
    if (pass->attachments[attachment].loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
      continue;

    if ((attachment + 1) * kClearValueBytes > command.clear_values.size()) {
      KNIZ_ERR("the render pass clear values do not cover colour attachment %u",
               attachment);
      continue;
    }

    const auto* value = reinterpret_cast<const float*>(
        command.clear_values.data() + attachment * kClearValueBytes);
    // Every format Kniz can render to stores floating point or normalised
    // values, so the float entry point is the right one; the integer entry points
    // would be needed only for integer attachments, which are refused at render
    // pass creation.
    gl::f.ClearBufferfv(gl::kColor, static_cast<gl::Int>(slot), value);
  }

  gl::report_errors("after clearing the render pass's attachments");
}

void render_end_pass(const RecordedCommand& command) {
  (void)command;

  // Unbinding is not merely tidiness: GL declares the result of reading a texture
  // that is attached to the *currently bound* framebuffer to be undefined, and
  // the very next thing a recorded stream tends to do is read the image back.
  gl::f.BindFramebuffer(gl::kFramebuffer, 0);
  gl::f.Disable(gl::kScissorTest);

  // Make the attachment writes visible to everything that follows -- a readback,
  // a copy, or a shader that samples the result.
  gl::f.MemoryBarrier(gl::kAllBarrierBits);
}

void render_draw(const RecordedCommand& command) {
  Pipeline* pipeline = command.pipeline;
  if (pipeline == nullptr)
    return;

  gl::f.UseProgram(pipeline->program);

  // ---- descriptors --------------------------------------------------------
  //
  // Before the vertex state, because the vertex shader may read a uniform block
  // and the draw is about to use it.  This is the same call the dispatch path
  // makes: whether a recorded command is a draw or a dispatch, binding a
  // descriptor to GL is the same operation, and having one implementation is
  // what keeps the two from drifting apart.
  if (!bind_descriptor_sets(pipeline, command.sets, "draw")) {
    KNIZ_ERR("the draw was skipped because its descriptors could not be bound");
    return;
  }

  // ---- vertex input -------------------------------------------------------
  //
  // GL has exactly one GL_ARRAY_BUFFER binding at a time, but each attribute
  // pointer captures the buffer that was bound when it was set.  So a pipeline
  // whose attributes come from several buffers is handled by binding each in
  // turn and setting only the pointers that read from it.
  gl::f.BindVertexArray(pipeline->vao);
  for (const VertexBinding& binding : pipeline->vertex_bindings) {
    if (binding.binding >= command.vertex_buffers.size()) {
      KNIZ_ERR("the pipeline's vertex input names binding %u but the draw "
               "recorded no buffer there",
               binding.binding);
      continue;
    }
    const BoundVertexBuffer& bound = command.vertex_buffers[binding.binding];
    if (bound.buffer == nullptr) {
      KNIZ_ERR("no vertex buffer is bound at binding %u", binding.binding);
      continue;
    }

    gl::f.BindBuffer(gl::kArrayBuffer, bound.buffer->gl_name);
    for (const VertexAttribute& attribute : pipeline->vertex_attributes) {
      if (attribute.binding != binding.binding)
        continue;

      const uint64_t offset = bound.offset + attribute.offset;
      gl::f.VertexAttribPointer(
          attribute.location, attribute.components, attribute.type,
          attribute.normalized ? 1u : 0u, static_cast<gl::Size>(binding.stride),
          reinterpret_cast<const void*>(static_cast<uintptr_t>(offset)));
      gl::f.EnableVertexAttribArray(attribute.location);
    }
  }
  gl::f.BindBuffer(gl::kArrayBuffer, 0);

  // The element array binding is part of the vertex array object's state and is
  // deliberately not undone: unbinding it with the array object still bound would
  // erase the binding the draw is about to use.
  bool index_type_ok = true;
  gl::Enum index_gl_type = 0;
  if (command.indexed) {
    if (command.index_buffer == nullptr) {
      KNIZ_ERR("the draw is indexed but no index buffer was bound");
      return;
    }
    index_gl_type = index_type_to_gl(command.index_type, &index_type_ok);
    if (!index_type_ok) {
      KNIZ_ERR("index type %d is not implemented",
               static_cast<int>(command.index_type));
      return;
    }
    gl::f.BindBuffer(gl::kElementArrayBuffer, command.index_buffer->gl_name);
  }

  // ---- fixed state --------------------------------------------------------
  gl::f.Viewport(static_cast<gl::Int>(command.viewport.x),
                 static_cast<gl::Int>(command.viewport.y),
                 static_cast<gl::Size>(command.viewport.width),
                 static_cast<gl::Size>(command.viewport.height));
  gl::f.Scissor(static_cast<gl::Int>(command.scissor.offset.x),
                static_cast<gl::Int>(command.scissor.offset.y),
                static_cast<gl::Size>(command.scissor.extent.width),
                static_cast<gl::Size>(command.scissor.extent.height));
  gl::f.Enable(gl::kScissorTest);

  if (pipeline->cull_face == gl::kNone) {
    gl::f.Disable(gl::kCullFace);
  } else {
    gl::f.Enable(gl::kCullFace);
    gl::f.CullFace(pipeline->cull_face);
  }
  gl::f.FrontFace(pipeline->front_face);

  // Vulkan's colour write mask and GL's four arguments are the same four bits in
  // the same order.
  gl::f.ColorMask((pipeline->color_write & VK_COLOR_COMPONENT_R_BIT) ? 1u : 0u,
                  (pipeline->color_write & VK_COLOR_COMPONENT_G_BIT) ? 1u : 0u,
                  (pipeline->color_write & VK_COLOR_COMPONENT_B_BIT) ? 1u : 0u,
                  (pipeline->color_write & VK_COLOR_COMPONENT_A_BIT) ? 1u : 0u);

  // ---- the draw -----------------------------------------------------------
  if (command.indexed) {
    const auto* indices = reinterpret_cast<const void*>(
        static_cast<uintptr_t>(command.index_offset));
    if (command.vertex_offset != 0) {
      // Vulkan adds vertexOffset to every index; glDrawElementsBaseVertex is the
      // entry point that does the same thing.
      gl::f.DrawElementsBaseVertex(pipeline->draw_mode,
                                   static_cast<gl::Size>(command.index_count),
                                   index_gl_type, indices, command.vertex_offset);
    } else {
      gl::f.DrawElements(pipeline->draw_mode,
                         static_cast<gl::Size>(command.index_count),
                         index_gl_type, indices);
    }
  } else {
    gl::f.DrawArrays(pipeline->draw_mode,
                     static_cast<gl::Int>(command.first_vertex),
                     static_cast<gl::Size>(command.vertex_count));
  }

  if (gl::report_errors("after the draw"))
    return;

  // A draw writes to the framebuffer, and a later command may read those texels.
  gl::f.MemoryBarrier(gl::kAllBarrierBits);
}

// ---------------------------------------------------------------------------
// Render passes
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateRenderPass(
    VkDevice deviceHandle, const VkRenderPassCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass) {
  (void)pAllocator;
  if (pRenderPass == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pRenderPass = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!device_can_draw(device))
    return VK_ERROR_FEATURE_NOT_PRESENT;

  if (pCreateInfo->subpassCount != 1) {
    // Subpasses are the one construct GL cannot express at all: they exist so
    // that a second pass can read what the first wrote without leaving the chip,
    // and GL would have to materialise every one of them as a texture.  Refusing
    // is the honest answer; see the fidelity budget in docs/DESIGN.md.
    KNIZ_ERR("a render pass must have exactly one subpass, and this one has %u; "
             "subpasses cannot be expressed on OpenGL",
             pCreateInfo->subpassCount);
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->attachmentCount == 0) {
    KNIZ_ERR("a render pass needs at least one attachment");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  RenderPass* pass = new (std::nothrow) RenderPass();
  if (pass == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  pass->device = device;
  pass->attachments.assign(pCreateInfo->pAttachments,
                           pCreateInfo->pAttachments + pCreateInfo->attachmentCount);

  for (uint32_t i = 0; i < pCreateInfo->attachmentCount; ++i) {
    const VkAttachmentDescription& attachment = pCreateInfo->pAttachments[i];

    if (attachment.samples != VK_SAMPLE_COUNT_1_BIT) {
      KNIZ_ERR("attachment %u asks for %u samples, and multisampling is not "
               "implemented",
               i, static_cast<unsigned>(attachment.samples));
      delete pass;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    const ImageFormat* format = image_format(attachment.format);
    if (format == nullptr) {
      KNIZ_ERR("attachment %u has format %d, which has no OpenGL counterpart",
               i, static_cast<int>(attachment.format));
      delete pass;
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    if (format->depth) {
      // A depth attachment is not merely unimplemented: it implies depth testing
      // and depth writes in the pipeline, which are refused too, so accepting the
      // attachment would only move the failure somewhere less explicable.
      KNIZ_ERR("attachment %u has depth format %d, and depth attachments are not "
               "implemented yet",
               i, static_cast<int>(attachment.format));
      delete pass;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (attachment.initialLayout == VK_IMAGE_LAYOUT_PREINITIALIZED &&
        attachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) {
      KNIZ_ERR("attachment %u asks to load a preinitialised layout, whose "
               "contents are undefined",
               i);
      delete pass;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  const VkSubpassDescription& subpass = pCreateInfo->pSubpasses[0];
  if (subpass.pipelineBindPoint != VK_PIPELINE_BIND_POINT_GRAPHICS) {
    KNIZ_ERR("a render pass subpass must be a graphics subpass");
    delete pass;
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (subpass.inputAttachmentCount != 0) {
    KNIZ_ERR("input attachments are not implemented; they need a subpass to read "
             "from, and subpasses cannot be expressed on OpenGL");
    delete pass;
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (subpass.pDepthStencilAttachment != nullptr &&
      subpass.pDepthStencilAttachment->attachment != VK_ATTACHMENT_UNUSED) {
    KNIZ_ERR("a depth/stencil attachment is not implemented yet");
    delete pass;
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (subpass.pResolveAttachments != nullptr) {
    for (uint32_t i = 0; i < subpass.colorAttachmentCount; ++i) {
      if (subpass.pResolveAttachments[i].attachment != VK_ATTACHMENT_UNUSED) {
        KNIZ_ERR("multisampled resolve attachments are not implemented");
        delete pass;
        return VK_ERROR_FEATURE_NOT_PRESENT;
      }
    }
  }
  if (subpass.colorAttachmentCount == 0) {
    KNIZ_ERR("a subpass with no colour attachment has nothing to draw into, and "
             "default framebuffer rendering arrives with the swapchain (M3)");
    delete pass;
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  pass->color_attachments.reserve(subpass.colorAttachmentCount);
  for (uint32_t i = 0; i < subpass.colorAttachmentCount; ++i) {
    const VkAttachmentReference& reference = subpass.pColorAttachments[i];
    if (reference.attachment == VK_ATTACHMENT_UNUSED ||
        reference.attachment >= pCreateInfo->attachmentCount) {
      KNIZ_ERR("the subpass's colour attachment %u refers to attachment %u, which "
               "the render pass does not declare",
               i, reference.attachment);
      delete pass;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    pass->color_attachments.push_back(reference.attachment);
  }

  // An attachment that is declared but that the subpass never writes is legal in
  // Vulkan, and a load-op clear of one is legal too.  Such a clear has no GL
  // equivalent: glClearBufferfv names a slot in the draw buffer list, and an
  // unused attachment is not in it.  Refused here rather than at the first
  // vkCmdBeginRenderPass that would have to do it.
  for (uint32_t i = 0; i < pCreateInfo->attachmentCount; ++i) {
    if (pCreateInfo->pAttachments[i].loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
      continue;
    uint32_t slot = 0;
    if (!subpass_slot_for_attachment(pass, i, &slot)) {
      KNIZ_ERR("attachment %u is cleared by the render pass but is not one of the "
               "subpass's colour attachments, so the clear has no OpenGL "
               "equivalent",
               i);
      delete pass;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
  }

  *pRenderPass = to_handle(pass);
  KNIZ_INFO("render pass created: %u attachment(s), %zu colour attachment(s)",
            pCreateInfo->attachmentCount, pass->color_attachments.size());
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyRenderPass(
    VkDevice deviceHandle, VkRenderPass renderPassHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  delete from_handle(renderPassHandle);
}

// ---------------------------------------------------------------------------
// Framebuffers
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateFramebuffer(
    VkDevice deviceHandle, const VkFramebufferCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer) {
  (void)pAllocator;
  if (pFramebuffer == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pFramebuffer = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!device_can_draw(device))
    return VK_ERROR_FEATURE_NOT_PRESENT;

  RenderPass* pass = from_handle(pCreateInfo->renderPass);
  if (pass == nullptr) {
    KNIZ_ERR("the framebuffer references something that is not a Kniz render "
             "pass");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (pCreateInfo->attachmentCount != pass->attachments.size()) {
    KNIZ_ERR("the framebuffer has %u attachment(s) but its render pass declares "
             "%zu",
             pCreateInfo->attachmentCount, pass->attachments.size());
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (pCreateInfo->width == 0 || pCreateInfo->height == 0) {
    KNIZ_ERR("a framebuffer needs a non-zero width and height");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (pCreateInfo->layers != 1) {
    KNIZ_ERR("only one framebuffer layer is implemented (%u requested)",
             pCreateInfo->layers);
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  Framebuffer* framebuffer = new (std::nothrow) Framebuffer();
  if (framebuffer == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  framebuffer->device = device;
  framebuffer->render_pass = pass;
  framebuffer->width = pCreateInfo->width;
  framebuffer->height = pCreateInfo->height;
  framebuffer->layers = pCreateInfo->layers;
  framebuffer->attachments.resize(pCreateInfo->attachmentCount);

  for (uint32_t i = 0; i < pCreateInfo->attachmentCount; ++i) {
    ImageView* view = from_handle(pCreateInfo->pAttachments[i]);
    if (view == nullptr) {
      KNIZ_ERR("framebuffer attachment %u is not a Kniz image view", i);
      delete framebuffer;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (view->image == nullptr || !view->image->storage_allocated) {
      KNIZ_ERR("framebuffer attachment %u is a view of an image with no memory "
               "bound, so it has no OpenGL texture storage to render into",
               i);
      delete framebuffer;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if ((view->image->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
      KNIZ_ERR("framebuffer attachment %u is an image created without "
               "VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT",
               i);
      delete framebuffer;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (view->format != pass->attachments[i].format) {
      KNIZ_ERR("framebuffer attachment %u has format %d but the render pass "
               "declares %d for it",
               i, static_cast<int>(view->format),
               static_cast<int>(pass->attachments[i].format));
      delete framebuffer;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (pCreateInfo->width > view->image->extent.width ||
        pCreateInfo->height > view->image->extent.height) {
      KNIZ_ERR("the framebuffer is %ux%u but attachment %u is only %ux%u",
               pCreateInfo->width, pCreateInfo->height, i,
               view->image->extent.width, view->image->extent.height);
      delete framebuffer;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    framebuffer->attachments[i] = view;
  }

  gl::f.GenFramebuffers(1, &framebuffer->gl_framebuffer);
  if (framebuffer->gl_framebuffer == 0) {
    KNIZ_ERR("glGenFramebuffers failed");
    delete framebuffer;
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }

  gl::f.BindFramebuffer(gl::kFramebuffer, framebuffer->gl_framebuffer);
  for (uint32_t i = 0; i < pCreateInfo->attachmentCount; ++i) {
    gl::f.FramebufferTexture2D(
        gl::kFramebuffer, static_cast<gl::Enum>(gl::kColorAttachment0 + i),
        gl::kTexture2D, framebuffer->attachments[i]->image->gl_texture, 0);
  }

  // The draw buffer list is the subpass's colour attachment list, in the order
  // the subpass declares it.  That ordering is what makes subpass attachment k
  // both GL_COLOR_ATTACHMENTk in the shader's output location and draw buffer k
  // for glClearBufferfv.
  std::vector<gl::Enum> draw_buffers;
  draw_buffers.reserve(pass->color_attachments.size());
  for (uint32_t attachment : pass->color_attachments)
    draw_buffers.push_back(static_cast<gl::Enum>(gl::kColorAttachment0 + attachment));
  gl::f.DrawBuffers(static_cast<gl::Size>(draw_buffers.size()), draw_buffers.data());

  const gl::Enum status = gl::f.CheckFramebufferStatus(gl::kFramebuffer);
  gl::f.BindFramebuffer(gl::kFramebuffer, 0);

  if (gl::report_errors("after creating the framebuffer")) {
    gl::f.DeleteFramebuffers(1, &framebuffer->gl_framebuffer);
    delete framebuffer;
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (status != gl::kFramebufferComplete) {
    KNIZ_ERR("the framebuffer is incomplete (glCheckFramebufferStatus reported "
             "0x%x); the attachment formats are probably not renderable",
             status);
    gl::f.DeleteFramebuffers(1, &framebuffer->gl_framebuffer);
    delete framebuffer;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  *pFramebuffer = to_handle(framebuffer);
  KNIZ_TRACE("framebuffer %p: GL framebuffer %u, %ux%u, %u attachment(s)",
             static_cast<void*>(*pFramebuffer), framebuffer->gl_framebuffer,
             framebuffer->width, framebuffer->height,
             pCreateInfo->attachmentCount);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyFramebuffer(
    VkDevice deviceHandle, VkFramebuffer framebufferHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  Device* device = from_handle(deviceHandle);
  Framebuffer* framebuffer = from_handle(framebufferHandle);
  if (device == nullptr || framebuffer == nullptr)
    return;
  if (device_can_draw(device) && framebuffer->gl_framebuffer != 0)
    gl::f.DeleteFramebuffers(1, &framebuffer->gl_framebuffer);
  delete framebuffer;
}

}  // namespace kniz
