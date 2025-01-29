si.module @DerefRefInTuple {
  %c0_i0 = si.constant 0 : i0
  %c17_i15 = si.constant 17 : i15
  %c42_i15 = si.constant 42 : i15
  %c1337_i15 = si.constant 1337 : i15
  %c9001_i15 = si.constant 9001 : i15

  // let x = 42, y = 17
  %x = si.var_decl "x" : !si.ref<i15>
  %y = si.var_decl "y" : !si.ref<i15>
  si.assign %x, %c42_i15 : !si.ref<i15>
  si.assign %y, %c17_i15 : !si.ref<i15>

  // let z = (&x, 0)
  %z = si.var_decl "z" : !si.ref<!si.tuple<[!si.ref<i15>, i0]>>
  %0 = si.tuple_create %x, %c0_i0 : (!si.ref<i15>, i0) -> !si.tuple<[!si.ref<i15>, i0]>
  si.assign %z, %0 : !si.ref<!si.tuple<[!si.ref<i15>, i0]>>

  // read *z.0, x
  %z.0 = si.tuple_get_ref %z, #builtin.int<0> : !si.ref<!si.tuple<[!si.ref<i15>, i0]>> -> !si.ref<!si.ref<i15>>
  %1 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  %2 = si.deref %1 : !si.ref<i15>
  si.output "a0", %2 : i15
  %3 = si.deref %x : !si.ref<i15>
  si.output "a1", %3 : i15

  // *z.0 = 1337
  %4 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  si.assign %4, %c1337_i15 : !si.ref<i15>

  // read *z.0, x
  %5 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  %6 = si.deref %5 : !si.ref<i15>
  si.output "b0", %6 : i15
  %7 = si.deref %x : !si.ref<i15>
  si.output "b1", %7 : i15

  // z.0 = &y
  si.assign %z.0, %y : !si.ref<!si.ref<i15>>

  // read *z.0, y
  %8 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  %9 = si.deref %8 : !si.ref<i15>
  si.output "c0", %9 : i15
  %10 = si.deref %y : !si.ref<i15>
  si.output "c1", %10 : i15

  // *z.0 = 9001
  %11 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  si.assign %11, %c9001_i15 : !si.ref<i15>

  // read *z.0, y
  %12 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  %13 = si.deref %12 : !si.ref<i15>
  si.output "d0", %13 : i15
  %14 = si.deref %y : !si.ref<i15>
  si.output "d1", %14 : i15

  %15 = si.tuple_get %0, #builtin.int<1> : !si.tuple<[!si.ref<i15>, i0]> -> i0
  si.output "e", %15 : i0
}
