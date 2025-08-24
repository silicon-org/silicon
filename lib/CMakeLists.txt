add_subdirectory(Codegen)
add_subdirectory(Conversion)
add_subdirectory(HIR)
add_subdirectory(MIR)
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
  SiliconMIR
)

add_silicon_library(SiliconRegisterPasses
  RegisterPasses.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  SiliconConversion
  SiliconHIRTransforms
)
