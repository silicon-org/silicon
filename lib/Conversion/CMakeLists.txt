add_silicon_library(SiliconConversion
  HIRToMIR.cpp

  DEPENDS
  SiliconConversionHeaders

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  SiliconHIR
  SiliconMIR
)
