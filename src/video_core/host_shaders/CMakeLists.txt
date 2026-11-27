# Copyright Citra Emulator Project / Azahar Emulator Project
# Licensed under GPLv2 or any later version
# Refer to the license.txt file included.

if (CMAKE_SCRIPT_MODE_FILE)
    file(READ "${METAL_LIBRARY_INPUT}" shader_hex HEX)
    if (shader_hex STREQUAL "")
        message(FATAL_ERROR "Cannot embed an empty Metal library: ${METAL_LIBRARY_INPUT}")
    endif()
    get_filename_component(shader_name "${METAL_LIBRARY_INPUT}" NAME_WE)
    string(TOUPPER "${shader_name}" shader_symbol)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " shader_bytes "${shader_hex}")
    string(REPEAT "0x[0-9a-f][0-9a-f], " 12 byte_row)
    string(REGEX REPLACE "(${byte_row})" "\\1\n" shader_bytes "${shader_bytes}")
    get_filename_component(header_dir "${METAL_HEADER_OUTPUT}" DIRECTORY)
    file(MAKE_DIRECTORY "${header_dir}")
    file(WRITE "${METAL_HEADER_OUTPUT}.tmp"
        "#pragma once\n\nnamespace HostShaders::Metal {\n"
        "inline constexpr unsigned char ${shader_symbol}[] = {\n${shader_bytes}\n};\n}\n")
    file(RENAME "${METAL_HEADER_OUTPUT}.tmp" "${METAL_HEADER_OUTPUT}")
    return()
endif()

set(SHADER_FILES
    format_reinterpreter/d24s8_to_rgba8.frag
    format_reinterpreter/d24s8_to_rgba8_ms.frag
    format_reinterpreter/rgba4_to_rgb5a1.frag
    format_reinterpreter/rgba4_to_rgb5a1_ms.frag
    format_reinterpreter/vulkan_d24s8_to_rgba8.comp
    format_reinterpreter/vulkan_d24s8_to_rgba8_ms.comp
    texture_filtering/bicubic.frag
    texture_filtering/refine.frag
    texture_filtering/scale_force.frag
    texture_filtering/xbrz_freescale.frag
    texture_filtering/mmpx.frag
    texture_filtering/x_gradient.frag
    texture_filtering/y_gradient.frag
    full_screen_triangle.vert
    opengl_present.frag
    opengl_present.vert
    opengl_present_anaglyph.frag
    opengl_present_interlaced.frag
    vulkan_depth_to_buffer.comp
    vulkan_present.frag
    vulkan_present.vert
    vulkan_present_anaglyph.frag
    vulkan_present_interlaced.frag
    vulkan_blit_depth_stencil.frag
    vulkan_cursor.frag
    vulkan_cursor.vert
)

set(SHADER_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/include)
set(SHADER_DIR ${SHADER_INCLUDE}/video_core/host_shaders)
set(HOST_SHADERS_INCLUDE ${SHADER_INCLUDE} PARENT_SCOPE)

set(INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/source_shader.h.in)
set(HEADER_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/StringShaderHeader.cmake)

if (ENABLE_METAL)
    set(SHADER_FILES)
    find_program(METAL_XCRUN xcrun REQUIRED)
    set(metal_sdk "${CMAKE_OSX_SYSROOT}")
    if (NOT metal_sdk)
        set(metal_sdk macosx)
    endif()
    execute_process(
        COMMAND "${METAL_XCRUN}" --sdk "${metal_sdk}" --show-sdk-path
        OUTPUT_VARIABLE metal_sdk_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND "${METAL_XCRUN}" --sdk "${metal_sdk_path}" --find metal
        OUTPUT_QUIET
        COMMAND_ERROR_IS_FATAL ANY
    )

    function(add_metal_host_shader shader_name)
        if (NOT shader_name MATCHES "^[a-z][a-z0-9_]*$")
            message(FATAL_ERROR "Invalid Metal host shader name: ${shader_name}")
        endif()
        set(source "${CMAKE_CURRENT_SOURCE_DIR}/metal/${shader_name}.metal")
        if (NOT EXISTS "${source}")
            message(FATAL_ERROR "Missing registered Metal host shader: ${source}")
        endif()
        set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/metal")
        set(library "${output_dir}/${shader_name}.metallib")
        set(header "${SHADER_DIR}/metal/${shader_name}.h")
        add_custom_command(
            OUTPUT "${library}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
            COMMAND "${METAL_XCRUN}" --sdk "${metal_sdk_path}" metal
                -isysroot "${metal_sdk_path}" "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}"
                -std=metal4.0 "-fmodules-cache-path=${output_dir}/module-cache"
                -MD -MF "${library}.d" "${source}" -o "${library}"
            MAIN_DEPENDENCY "${source}"
            DEPFILE "${library}.d"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${header}"
            COMMAND "${CMAKE_COMMAND}" "-DMETAL_LIBRARY_INPUT=${library}"
                "-DMETAL_HEADER_OUTPUT=${header}" -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
            DEPENDS "${library}" "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
            VERBATIM
        )
        set(SHADER_HEADERS ${SHADER_HEADERS} "${header}" PARENT_SCOPE)
        set(SHADER_FILES ${SHADER_FILES} "${source}" PARENT_SCOPE)
    endfunction()

    # Register completed Metal shader sources here.
    add_metal_host_shader(d24s8_to_rgba8_comp)
    add_metal_host_shader(d24s8_to_rgba8_ms_comp)
    add_metal_host_shader(full_screen_triangle_vert)
    add_metal_host_shader(bicubic_frag)
    add_metal_host_shader(mmpx_frag)
    add_metal_host_shader(refine_frag)
    add_metal_host_shader(scale_force_frag)
    add_metal_host_shader(tex_coord_vert)
    add_metal_host_shader(x_gradient_frag)
    add_metal_host_shader(xbrz_freescale_frag)
    add_metal_host_shader(y_gradient_frag)
    add_metal_host_shader(blit_depth_stencil_frag)
    add_metal_host_shader(cursor_frag)
    add_metal_host_shader(cursor_vert)
    add_metal_host_shader(depth_to_buffer_comp)
    add_metal_host_shader(present_frag)
    add_metal_host_shader(present_vert)
    add_metal_host_shader(present_anaglyph_frag)
    add_metal_host_shader(present_interlaced_frag)
else()
    foreach(FILENAME IN ITEMS ${SHADER_FILES})
        string(REPLACE "." "_" SHADER_NAME ${FILENAME})
        set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME})
        set(SOURCE_HEADER_FILE ${SHADER_DIR}/${SHADER_NAME}.h)
        add_custom_command(
            OUTPUT
                ${SOURCE_HEADER_FILE}
            COMMAND
                ${CMAKE_COMMAND} -P ${HEADER_GENERATOR} ${SOURCE_FILE} ${SOURCE_HEADER_FILE} ${INPUT_FILE}
            MAIN_DEPENDENCY
                ${SOURCE_FILE}
            DEPENDS
                ${INPUT_FILE}
                # HEADER_GENERATOR should be included here but msbuild seems to assume it's always modified
        )
        set(SHADER_HEADERS ${SHADER_HEADERS} ${SOURCE_HEADER_FILE})
    endforeach()
endif()

set(SHADER_SOURCES ${SHADER_FILES})
list(APPEND SHADER_SOURCES ${GLSL_INCLUDES})

add_custom_target(host_shaders
    DEPENDS
        ${SHADER_HEADERS}
    SOURCES
        ${SHADER_SOURCES}
)
