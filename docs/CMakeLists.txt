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
foreach(doc ${static_docs})
  collect_doc(${doc} docs/${doc})
endforeach()

# Collect these documents as part of the silicon-docs target.
add_custom_target(SiliconDocFiles DEPENDS ${collected_docs})
add_dependencies(silicon-docs SiliconDocFiles)
