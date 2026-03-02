add_silicon_library(SiliconTransforms
  Interpret.cpp
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
)
