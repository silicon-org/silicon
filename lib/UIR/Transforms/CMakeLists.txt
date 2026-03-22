add_silicon_dialect_library(SiliconUIRTransforms
  FlattenCF.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconUIR
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
