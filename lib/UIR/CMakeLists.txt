add_silicon_dialect_library(SiliconUIR
  Dialect.cpp
  Ops.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconSupport
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
)

add_subdirectory(Transforms)
