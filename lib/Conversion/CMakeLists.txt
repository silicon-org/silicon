add_silicon_library(SiliconConversion
  HIRToMIR.cpp
  MIRToCIRCT.cpp

  DEPENDS
  SiliconConversionHeaders

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  MLIRTransformUtils
  SiliconHIR
  SiliconMIR
)
