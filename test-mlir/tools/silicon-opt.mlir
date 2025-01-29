// RUN: silicon-opt --help | FileCheck %s --check-prefix=HELP
// RUN: silicon-opt --show-dialects | FileCheck %s --check-prefix=DIALECT

// HELP: OVERVIEW: Silicon modular optimizer driver

// DIALECT: Available Dialects:
// DIALECT-SAME: cf
// DIALECT-SAME: comb
// DIALECT-SAME: func
// DIALECT-SAME: hw
// DIALECT-SAME: seq
