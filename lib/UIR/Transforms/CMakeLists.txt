add_silicon_dialect_library(SiliconUIRTransforms
  FlattenCF.cpp
  NormalizeExits.cpp
  SortPhases.cpp
  SortPhases2.cpp
  SplitPhases2.cpp
  TestPhaseAnalysis2.cpp
  ThreadPhases.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconUIR
  SiliconUIRAnalysis
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
