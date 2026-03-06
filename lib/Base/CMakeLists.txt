add_silicon_dialect_library(SiliconBase
  Attributes.cpp
  Dialect.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  SiliconSupport
)
