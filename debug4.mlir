// fn foo<N: int>(a: uint<N>, b: uint<N>)
hir.func "foo" {
  %int = hir.int_type
  %N = hir.func_arg "N", %int
  %0 = hir.uint_type %N
  %a = hir.func_arg "a", %0
  %b = hir.func_arg "b", %0
  hir.return_signature
} {
  hir.return
}

// fn bar()
hir.func "bar" {
  hir.return_signature
} {
  // let x;
  // let y;
  %xT = hir.inferrable
  %yT = hir.inferrable
  %x = hir.let "x", %xT
  %y = hir.let "y", %yT
  // foo(x, y);
  %0 = hir.load %x, %xT
  %1 = hir.load %y, %yT
  %N = hir.inferrable
  hir.unlegalized_call @foo(%N, %0, %1)
  hir.return
}
