add_silicon_dialect_library(SiliconHIR
  HIRAttributes.cpp
  HIRDialect.cpp
  HIROps.cpp
  HIRTypes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

add_subdirectory(Transforms)
