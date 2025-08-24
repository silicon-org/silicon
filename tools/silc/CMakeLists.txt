set(libs
  SiliconCodegen
  SiliconRegisterDialects
  SiliconRegisterPasses
  SiliconSyntax
)

add_silicon_tool(silc silc.cpp DEPENDS ${libs})
target_link_libraries(silc PRIVATE ${libs})

llvm_update_compile_flags(silc)
mlir_check_all_link_libraries(silc)
