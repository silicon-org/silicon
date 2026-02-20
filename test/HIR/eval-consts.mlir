// RUN: silicon-opt --eval-consts %s | FileCheck %s

// CHECK-LABEL: func @UncondBranch
func.func @UncondBranch(%arg0: i42, %arg1: i42) {
  // CHECK-NEXT: call @use_i42(%arg0)
  // CHECK-NEXT: call @use_i42(%arg1)
  // CHECK-NEXT: return
  call @use_i42(%arg0) : (i42) -> ()
  hir.const_br ^bb1(%arg1 : i42)
^bb1(%0: i42):
  call @use_i42(%0) : (i42) -> ()
  return
}

// CHECK-LABEL: func @CondBranch
func.func @CondBranch(%arg0: i42, %arg1: i42) {
  // CHECK-NEXT: call @use_i42(%arg0)
  // CHECK-NEXT: call @use_i42(%arg1)
  // CHECK-NEXT: return
  %false = arith.constant false
  %true = arith.constant true
  hir.const_br ^bb1(%false, %arg0 : i1, i42)
^bb1(%0: i1, %1: i42):
  call @use_i42(%1) : (i42) -> ()
  hir.const_cond_br %0, ^bb2, ^bb1(%true, %arg1 : i1, i42)
^bb2:
  return
}

// CHECK-LABEL: func @Loop
func.func @Loop(%arg0: i42) -> i42 {
  // CHECK-NEXT: [[T0:%.+]] = call @process_i42(%arg0)
  // CHECK-NEXT: [[T1:%.+]] = call @process_i42([[T0]])
  // CHECK-NEXT: [[T2:%.+]] = call @process_i42([[T1]])
  // CHECK-NEXT: [[T3:%.+]] = call @process_i42([[T2]])
  // CHECK-NEXT: [[T4:%.+]] = call @process_i42([[T3]])
  // CHECK-NEXT: [[T5:%.+]] = call @process_i42([[T4]])
  // CHECK-NEXT: [[T6:%.+]] = call @process_i42([[T5]])
  // CHECK-NEXT: [[T7:%.+]] = call @process_i42([[T6]])
  // CHECK-NEXT: return [[T7]]
  %c1_index = arith.constant 1 : index
  %c8_index = arith.constant 8 : index
  hir.const_br ^bb1(%c1_index, %arg0 : index, i42)
^bb1(%0: index, %1: i42):
  %2 = call @process_i42(%1) : (i42) -> i42
  %3 = arith.addi %0, %c1_index : index
  %4 = arith.cmpi ult, %0, %c8_index : index
  hir.const_cond_br %4, ^bb1(%3, %2 : index, i42), ^bb2(%2 : i42)
^bb2(%5: i42):
  return %5 : i42
}

func.func private @use_i42(%x: i42)
func.func private @process_i42(%x: i42) -> i42
