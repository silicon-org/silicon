add_silicon_dialect_library(SiliconMIRTransforms
  Interpret.cpp

  LINK_LIBS PUBLIC
  SiliconMIR
  MLIRIR
  MLIRPass
)
