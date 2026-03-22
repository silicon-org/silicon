add_silicon_dialect_library(SiliconUIR
  Dialect.cpp
  Ops.cpp

  LINK_LIBS PUBLIC
  SiliconSupport
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
)
