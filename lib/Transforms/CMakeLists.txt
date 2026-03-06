add_silicon_library(SiliconTransforms
  PhaseEvalLoop.cpp
  SpecializeFuncs.cpp

  DEPENDS
  SiliconTransformsHeaders

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  SiliconConversion
  SiliconHIR
  SiliconMIR
  SiliconMIRTransforms
)
