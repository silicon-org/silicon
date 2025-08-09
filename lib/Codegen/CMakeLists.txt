add_silicon_library(SiliconCodegen
  Codegen.cpp
  Expression.cpp
  Function.cpp
  Root.cpp
  Type.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  SiliconHIR
  SiliconSyntax
)
