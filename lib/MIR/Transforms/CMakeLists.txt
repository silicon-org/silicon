add_silicon_dialect_library(SiliconMIRTransforms
  Interpret.cpp

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  SiliconMIR
  MLIRIR
  MLIRPass
)
