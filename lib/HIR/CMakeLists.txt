add_silicon_dialect_library(SiliconHIR
  Attributes.cpp
  Dialect.cpp
  Ops.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_subdirectory(Transforms)
