func.func @reduce_xor(%arg0: i64) -> i1 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %false = arith.constant false
  cf.br ^bb1(%c0, %false : index, i1)
^bb1(%idx: index, %acc: i1):
  %0 = arith.cmpi ult, %idx, %c64 : index
  cf.cond_br %0, ^bb2, ^bb3
^bb2:
  %1 = call @extract.i1(%arg0, %idx) : (i64, index) -> i1
  %2 = arith.xori %acc, %1 : i1
  %3 = arith.addi %idx, %c1 : index
  cf.br ^bb1(%3, %2 : index, i1)
^bb3:
  return %acc : i1
}

func.func private @extract.i1(%value: i64, %offset: index) -> i1
