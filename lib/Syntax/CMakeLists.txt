add_silicon_library(SiliconSyntax
  Lexer.cpp
  Tokens.cpp
  Parser.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
