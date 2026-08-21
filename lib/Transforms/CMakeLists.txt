add_silicon_library(SiliconTransforms
  CheckCalls.cpp
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
  SiliconUIR
)
