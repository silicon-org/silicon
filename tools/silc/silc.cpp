//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Codegen.h"
#include "silicon/LLVM.h"
#include "silicon/MLIR.h"
#include "silicon/RegisterAll.h"
#include "silicon/Syntax/AST.h"
#include "silicon/Syntax/Lexer.h"
#include "silicon/Syntax/Names.h"
#include "silicon/Syntax/Parser.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace silicon;
namespace cl = llvm::cl;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

struct Opt {
  cl::opt<std::string> inputFilename{
      cl::Positional, cl::value_desc("filename"), cl::init("-"),
      cl::desc("Input filename (`-` for stdin)")};

  cl::opt<std::string> outputFilename{
      "o", cl::value_desc("filename"), cl::init("-"),
      cl::desc("Output filename (`-` for stdout)")};

  cl::opt<bool> verifyDiagnostics{
      "verify-diagnostics", cl::init(false), cl::Hidden,
      cl::desc("Check that emitted diagnostics match expected-* lines on the "
               "corresponding line")};

  cl::opt<bool> testLexer{"test-lexer", cl::init(false), cl::Hidden,
                          cl::desc("Print all tokens in the input")};

  cl::opt<bool> testParser{"test-parser", cl::init(false), cl::Hidden,
                           cl::desc("Print the AST after parsing")};

  cl::opt<bool> splitInputFile{
      "split-input-file", cl::init(false), cl::Hidden,
      cl::desc("Split the input file into chunks and process each separately")};
};
static Opt opt;

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

static LogicalResult process(MLIRContext *context, llvm::SourceMgr &sourceMgr,
                             llvm::raw_ostream &os) {
  // Create a lexer to tokenize the input.
  Lexer lexer(context, sourceMgr);

  // If we are only testing the lexer, print all tokens in the input and exit.
  if (opt.testLexer) {
    while (auto token = lexer.next()) {
      if (token.isError())
        return failure();
      os << lexer.getLoc(token) << ": " << token << "\n";
    }
    return success();
  }

  // Create a parser and parse the input into an AST.
  AST ast;
  Parser parser(lexer, ast);
  auto *root = parser.parseRoot();
  if (!root)
    return failure();
  ast.roots.push_back(root);

  // Resolve names in the AST.
  if (failed(resolveNames(ast)))
    return failure();

  // If we are only testing the parser, print the AST and exit.
  if (opt.testParser) {
    ast.print(os);
    return success();
  }

  // Convert the AST to MLIR.
  auto module = convertToIR(context, ast);
  if (!module)
    return failure();

  // Print the final MLIR.
  module->print(os);
  return success();
}

static LogicalResult executeCompiler(MLIRContext *context) {
  // Open the source file.
  std::string errorMessage;
  auto inputFile = mlir::openInputFile(opt.inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::WithColor::error()
        << "failed to open input file: " << errorMessage << "\n";
    return failure();
  }

  // Open the output file.
  auto outputFile = mlir::openOutputFile(opt.outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::WithColor::error()
        << "failed to open output file: " << errorMessage << "\n";
    return failure();
  }

  // Utility to call `process` with either the regular diagnostic handler, or,
  // if `--verify-diagnostics` is set, with the verifying handler.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> buffer,
                           llvm::raw_ostream &os) -> LogicalResult {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

    if (!opt.verifyDiagnostics) {
      mlir::SourceMgrDiagnosticHandler handler(sourceMgr, context);
      return process(context, sourceMgr, os);
    }

    mlir::SourceMgrDiagnosticVerifierHandler handler(sourceMgr, context);
    context->printOpOnDiagnostic(false);
    (void)process(context, sourceMgr, os);
    return handler.verify();
  };

  // Split the input file into chunks if the `--split-input-file` option is set.
  // Otherwise, process the entire input file as a single buffer.
  auto result = opt.splitInputFile
                    ? mlir::splitAndProcessBuffer(
                          std::move(inputFile), processBuffer, outputFile->os())
                    : processBuffer(std::move(inputFile), outputFile->os());

  // Keep the output file around if no errors occurred.
  if (succeeded(result))
    outputFile->keep();
  return result;
}

int main(int argc, char **argv) {
  // Register pass manager command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  // Register dialects and passes.
  mlir::DialectRegistry registry;
  silicon::registerAllDialects(registry);
  silicon::registerAllPasses();

  // Parse command line options.
  cl::ParseCommandLineOptions(argc, argv,
                              "Silicon hardware description language compiler");

  // Perform the actual work.
  MLIRContext context(registry);
  context.printOpOnDiagnostic(false);
  exit(failed(executeCompiler(&context)));
}
