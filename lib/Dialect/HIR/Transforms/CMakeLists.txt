add_silicon_dialect_library(SiliconHIRTransforms
  InferTypes.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
