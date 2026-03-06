add_silicon_library(SiliconTransforms
  PhaseEvalLoop.cpp

  DEPENDS
  SiliconTransformsHeaders

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  SiliconConversion
  SiliconHIR
  SiliconHIRTransforms
  SiliconMIR
  SiliconMIRTransforms
)
