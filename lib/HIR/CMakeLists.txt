add_silicon_dialect_library(SiliconHIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  SiliconBase
  SiliconSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_subdirectory(Analysis)
add_subdirectory(Transforms)
