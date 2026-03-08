add_silicon_library(SiliconHIRAnalysis
  PhaseAnalysis.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  MLIRControlFlowDialect
  MLIRIR
  MLIRSideEffectInterfaces
)
