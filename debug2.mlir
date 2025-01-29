// let x;
%xT = hir.inferrable
%x = hir.let "x", %xT
// x = 1;
%c1T = hir.inferrable
%c1 = hir.literal 1, %c1T
%x0 = hir.type_of %x
%x1 = hir.unpack_ref %x0
%x2 = hir.type_of %c1
%x3 = hir.unify %x1, %x2
hir.store %x, %c1, %x3
// let y: uint<_>;
%yW = hir.inferrable
%yT = hir.uint_type %yW
%y = hir.let "y", %yT
// y = 2;
%c2T = hir.inferrable
%c2 = hir.literal 2, %c2T
%y0 = hir.type_of %y
%y1 = hir.unpack_ref %y0
%y2 = hir.type_of %c2
%y3 = hir.unify %y1, %y2
hir.store %y, %c2, %y3
// let z: uint<42>;
%c42 = hir.int 42
%zT = hir.uint_type %c42
%z = hir.let "z", %zT
// z = x + y;
%add0 = hir.type_of %x
%add1 = hir.unpack_ref %add0
%add2 = hir.load %x, %add1
%add3 = hir.type_of %y
%add4 = hir.unpack_ref %add3
%add5 = hir.load %y, %add4
%add6 = hir.unify %add1, %add4
%add7 = hir.add %add2, %add5, %add6
%z0 = hir.type_of %z
%z1 = hir.unpack_ref %z0
%z3 = hir.type_of %add7
%z4 = hir.unify %z1, %z3
hir.store %z, %add7, %z4
