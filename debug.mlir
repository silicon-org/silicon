func.func @Foo() {
  %0 = hir.const_region -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %1 = func.call @Bar(%c42_i32) : (i32) -> i32
    %2 = hir.const_region -> i32 {
      %c19_i32 = arith.constant 19 : i32
      %3 = func.call @Bar(%c19_i32) : (i32) -> i32
      hir.yield %3 : i32
    }
    %4 = arith.addi %1, %2 : i32
    hir.yield %4 : i32
  }
  return
}

func.func @Bar(%arg0: i32) -> i32 {
  %0 = hir.const_region -> i32 {
    %2 = func.call @Konst() : () -> i32
    hir.yield %2 : i32
  }
  %1 = arith.addi %arg0, %0 : i32
  return %1 : i32
}

func.func @Konst() -> i32 {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32 : i32
}
