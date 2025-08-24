add_silicon_library(SiliconConversion
  HIRToMIR.cpp

  DEPENDS
  SiliconConversionIncGen

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconMIR
)
