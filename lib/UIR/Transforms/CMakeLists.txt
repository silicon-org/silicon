add_silicon_dialect_library(SiliconUIRTransforms
  FlattenCF.cpp
  TestPhaseAnalysis2.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconUIR
  SiliconUIRAnalysis
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
