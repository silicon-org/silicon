// RUN: silicon-opt --lower-mir-to-circt --split-input-file --verify-diagnostics %s

// (No MIR-to-CIRCT conversion errors to test currently. The mir.if multi-result
// error was removed along with mir.if; CFG-to-dataflow conversion is deferred.)
