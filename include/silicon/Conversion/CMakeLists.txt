set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
mlir_tablegen(Passes.md -gen-pass-doc)

add_public_tablegen_target(SiliconConversionHeaders)
add_dependencies(silicon-headers SiliconConversionHeaders)
