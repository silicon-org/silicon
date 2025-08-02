add_silicon_library(SiliconSyntax
  AST.cpp
  Lexer.cpp
  Names.cpp
  Tokens.cpp
  Parser.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
