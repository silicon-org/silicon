add_subdirectory(Codegen)
add_subdirectory(Dialect)
add_subdirectory(Syntax)

add_silicon_library(SiliconRegisterDialects
  RegisterDialects.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  SiliconHIR
)

add_silicon_library(SiliconRegisterPasses
  RegisterPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  SiliconHIRTransforms
)
