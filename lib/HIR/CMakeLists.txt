add_silicon_dialect_library(SiliconHIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  SiliconBase
  SiliconSupport
  MLIRFuncDialect
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_subdirectory(Transforms)
