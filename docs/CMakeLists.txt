# Helper that copies a documentation file from source to destination
set(dstdir ${CMAKE_BINARY_DIR}/docs)
set(collect_py ${CMAKE_CURRENT_SOURCE_DIR}/collect.py)
set(collect_args -I ${SILICON_BINARY_DIR} -I ${SILICON_SOURCE_DIR})
macro(collect_doc dst src)
  add_custom_command(
    OUTPUT ${dstdir}/${dst}
    COMMAND ${Python3_EXECUTABLE} ${collect_py} ${collect_args} ${src} ${dstdir}/${dst}
    DEPENDS ${collect_py} silicon-headers
    COMMENT "Collecting docs ${dst}"
  )
  list(APPEND collected_docs ${dstdir}/${dst})
endmacro()

# Dialects
collect_doc(internals/dialects/HIR.md include/silicon/HIR/Dialect.md)
collect_doc(internals/dialects/MIR.md include/silicon/MIR/Dialect.md)

# Passes
collect_doc(internals/passes/HIR.md include/silicon/HIR/Passes.md)
collect_doc(internals/passes/MIR.md include/silicon/MIR/Passes.md)
collect_doc(internals/passes/Conversion.md include/silicon/Conversion/Passes.md)
collect_doc(internals/passes/Transforms.md include/silicon/Transforms/Passes.md)

# Static documents
file(GLOB_RECURSE static_docs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.md)
list(FILTER static_docs EXCLUDE REGEX "^themes/")
foreach(doc ${static_docs})
  collect_doc(${doc} docs/${doc})
endforeach()

# Collect these documents as part of the silicon-docs target.
add_custom_target(SiliconDocFiles DEPENDS ${collected_docs})
add_dependencies(silicon-docs SiliconDocFiles)

# Build the website with Hugo. We copy hugo.toml and symlink the theme into
# the build docs directory, so Hugo runs entirely inside the build tree.
find_program(HUGO hugo)
if(HUGO)
  add_custom_target(silicon-website
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/hugo.toml ${dstdir}/hugo.toml
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/themes ${dstdir}/themes
    COMMAND ${HUGO} --source ${dstdir} --destination ${CMAKE_BINARY_DIR}/website
    COMMENT "Building website"
  )
  add_dependencies(silicon-website silicon-docs)
else()
  message(STATUS "Hugo not found; silicon-website target will not be available")
endif()
