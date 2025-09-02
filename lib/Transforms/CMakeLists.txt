add_silicon_library(SiliconTransforms
  Interpret.cpp

  DEPENDS
  SiliconTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  SiliconHIR
  SiliconMIR
)
