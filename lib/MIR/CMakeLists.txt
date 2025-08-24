add_silicon_dialect_library(SiliconMIR
  Attributes.cpp
  Dialect.cpp
  Ops.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
