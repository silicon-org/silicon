 // RUN: silc --help | FileCheck %s --check-prefix=HELP
 // RUN: silc --show-dialects | FileCheck %s --check-prefix=DIALECT

 // HELP: OVERVIEW: Silicon compiler driver

 // DIALECT: Available Dialects:
 // DIALECT-SAME: cf
 // DIALECT-SAME: comb
 // DIALECT-SAME: func
 // DIALECT-SAME: hir
 // DIALECT-SAME: hw
 // DIALECT-SAME: seq
