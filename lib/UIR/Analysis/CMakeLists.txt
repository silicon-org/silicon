add_silicon_library(SiliconUIRAnalysis
  PhaseAnalysis.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconUIR
  MLIRIR
  MLIRSideEffectInterfaces
)
