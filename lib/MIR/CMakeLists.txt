add_subdirectory(Transforms)

add_silicon_dialect_library(SiliconMIR
  Attributes.cpp
  Dialect.cpp
  Ops.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  SiliconBase
  SiliconSupport
)
