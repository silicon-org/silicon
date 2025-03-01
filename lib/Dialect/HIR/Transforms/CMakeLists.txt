add_silicon_dialect_library(SiliconHIRTransforms
  EvalConsts.cpp
  InferTypes.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
