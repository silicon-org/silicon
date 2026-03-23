add_silicon_dialect_library(SiliconUIRTransforms
  FlattenCF.cpp
  SplitPhases2.cpp
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
