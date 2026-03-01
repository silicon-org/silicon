# Helper that copies a documentation file from source to destination.
set(docdir ${CMAKE_BINARY_DIR}/docs)
set(collect_py ${CMAKE_CURRENT_SOURCE_DIR}/collect.py)
macro(collect_doc dst src)
  add_custom_command(
    OUTPUT ${docdir}/${dst}
    COMMAND ${Python3_EXECUTABLE} ${collect_py} ${src} ${docdir}/${dst}
    DEPENDS ${collect_py} ${src} silicon-headers
    COMMENT "Collecting docs ${dst}"
  )
  list(APPEND collected_docs ${docdir}/${dst})
endmacro()

# Dialects
collect_doc(internals/dialects/HIR.md ${SILICON_BINARY_DIR}/include/silicon/HIR/Dialect.md)
collect_doc(internals/dialects/MIR.md ${SILICON_BINARY_DIR}/include/silicon/MIR/Dialect.md)

# Passes
collect_doc(internals/passes/HIR.md ${SILICON_BINARY_DIR}/include/silicon/HIR/Passes.md)
collect_doc(internals/passes/MIR.md ${SILICON_BINARY_DIR}/include/silicon/MIR/Passes.md)
collect_doc(internals/passes/Conversions.md ${SILICON_BINARY_DIR}/include/silicon/Conversion/Passes.md)
collect_doc(internals/passes/Transforms.md ${SILICON_BINARY_DIR}/include/silicon/Transforms/Passes.md)

# Static documents
file(GLOB_RECURSE static_docs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.md)
list(FILTER static_docs EXCLUDE REGEX "^themes/")
foreach(doc ${static_docs})
  collect_doc(${doc} ${CMAKE_CURRENT_SOURCE_DIR}/${doc})
endforeach()

# Collect these documents as part of the silicon-docs target.
add_custom_target(SiliconDocFiles DEPENDS ${collected_docs})
add_dependencies(silicon-docs SiliconDocFiles)

# Build the website with Hugo. We copy hugo.toml and symlink the theme into
# the build docs directory, so Hugo runs entirely inside the build tree.
find_program(HUGO hugo)
if(HUGO)
  add_custom_target(silicon-website
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/website
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/hugo.toml ${docdir}/hugo.toml
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/themes ${docdir}/themes
    COMMAND ${HUGO} --source ${docdir} --destination ${CMAKE_BINARY_DIR}/website
    COMMENT "Building website"
  )
  add_dependencies(silicon-website silicon-docs)
else()
  message(STATUS "Hugo not found; silicon-website target will not be available")
endif()
