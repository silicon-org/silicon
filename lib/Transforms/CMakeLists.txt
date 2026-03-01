add_silicon_library(SiliconTransforms
  Interpret.cpp
  SpecializeFuncs.cpp

  DEPENDS
  SiliconTransformsHeaders

  LINK_LIBS PUBLIC
  MLIRPass
  SiliconHIR
  SiliconMIR
)
