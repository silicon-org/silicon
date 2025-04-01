add_silicon_dialect_library(SiliconHIR
  HIRAttributes.cpp
  HIRDialect.cpp
  HIROps.cpp
  HIRTypes.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_subdirectory(Transforms)
