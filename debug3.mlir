// let x;
%c42 = hir.int 42
%yT = hir.uint_type %c42
%x = hir.let "x", %yT
// x = 1;
%c1 = hir.literal 1, %yT
hir.store %x, %c1, %yT
// let y: uint<_>;
%y = hir.let "y", %yT
// y = 2;
%c2 = hir.literal 2, %yT
hir.store %y, %c2, %yT
// let z: uint<42>;
%z = hir.let "z", %yT
// z = x + y;
%add2 = hir.load %x, %yT
%add5 = hir.load %y, %yT
%add7 = hir.add %add2, %add5, %yT
hir.store %z, %add7, %yT
