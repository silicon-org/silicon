set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
mlir_tablegen(Passes.md -gen-pass-doc)

add_public_tablegen_target(SiliconTransformsHeaders)
add_dependencies(silicon-headers SiliconTransformsHeaders)
