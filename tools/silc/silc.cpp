//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/LLVM.h"
#include "silicon/MLIR.h"
#include "silicon/RegisterAll.h"
#include "silicon/Syntax/Lexer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"
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
};
static Opt opt;

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

static LogicalResult process(MLIRContext *context, llvm::SourceMgr &sourceMgr,
                             llvm::ToolOutputFile &outputFile) {
  // Create a lexer to tokenize the input.
  Lexer lexer(context, sourceMgr);

  // If we are only testing the lexer, print all tokens in the input and exit.
  if (opt.testLexer) {
    auto &os = outputFile.os();
    while (auto token = lexer.next()) {
      if (token.isError())
        return failure();
      os << lexer.getLoc(token) << ": " << token << "\n";
    }
    outputFile.keep();
    return success();
  }

  outputFile.keep();
  return success();
}

static LogicalResult executeCompiler(MLIRContext *context) {
  // Open the source file.
  std::string errorMessage;
  llvm::SourceMgr sourceMgr;
  auto inputFile = mlir::openInputFile(opt.inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::WithColor::error()
        << "failed to open input file: " << errorMessage << "\n";
    return failure();
  }
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());

  // Open the output file.
  auto outputFile = mlir::openOutputFile(opt.outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::WithColor::error()
        << "failed to open output file: " << errorMessage << "\n";
    return failure();
  }

  // Call `process` with either the regular diagnostic handler, or, if
  // `--verify-diagnostics` is set, with the verifying handler.
  if (!opt.verifyDiagnostics) {
    mlir::SourceMgrDiagnosticHandler handler(sourceMgr, context);
    return process(context, sourceMgr, *outputFile);
  }

  mlir::SourceMgrDiagnosticVerifierHandler handler(sourceMgr, context);
  context->printOpOnDiagnostic(false);
  (void)process(context, sourceMgr, *outputFile);
  return handler.verify();
}

int main(int argc, char **argv) {
  // Register pass manager command line options.
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
  exit(failed(executeCompiler(&context)));
}
