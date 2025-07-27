get_property(dialect_libs GLOBAL PROPERTY SILICON_DIALECT_LIBS)

set(libs
  ${dialect_libs}
  SiliconSyntax
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIROptLib
)

add_silicon_tool(silc silc.cpp DEPENDS ${libs})
target_link_libraries(silc PRIVATE ${libs})

llvm_update_compile_flags(silc)
mlir_check_all_link_libraries(silc)
