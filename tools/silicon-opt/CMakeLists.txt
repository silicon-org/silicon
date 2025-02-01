get_property(dialect_libs GLOBAL PROPERTY SILICON_DIALECT_LIBS)

set(libs
  ${dialect_libs}
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIROptLib
)

add_silicon_tool(silicon-opt silicon-opt.cpp DEPENDS ${libs})
target_link_libraries(silicon-opt PRIVATE ${libs})

llvm_update_compile_flags(silicon-opt)
mlir_check_all_link_libraries(silicon-opt)
