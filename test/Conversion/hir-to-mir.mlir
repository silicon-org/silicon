// RUN: silicon-opt --lower-hir-to-mir %s | FileCheck %s

// CHECK-LABEL: @Types
hir.func @Types {
  // CHECK: mir.constant #mir.type<!mir.int>
  %int_type = hir.int_type

  // CHECK: mir.constant #mir.int<42>
  // CHECK: mir.constant #mir.type<!mir.uint<42>>
  %c42_int = hir.constant_int 42
  %uint42_type = hir.uint_type %c42_int

  // CHECK: mir.constant #mir.type<!mir.anyfunc>
  %anyfunc_type = hir.anyfunc_type

  // CHECK: mir.constant #mir.type<() -> ()>
  hir.func_type () -> ()
  // CHECK: mir.constant #mir.type<(!mir.int) -> !mir.uint<42>>
  hir.func_type (%int_type) -> (%uint42_type)

  hir.return
}

// CHECK-LABEL: @Constants
hir.func @Constants {
  // CHECK: mir.constant #mir.int<42>
  hir.constant_int 42
  // CHECK: mir.constant #mir.unit
  hir.constant_unit
  // CHECK: mir.constant #mir.type
  // CHECK: mir.constant #mir.func<@foo> : () -> ()
  %0 = hir.func_type () -> ()
  hir.constant_func @foo : %0
  hir.return
}

// CHECK-LABEL: @Calls
hir.func @Calls {
  // CHECK: mir.call @foo() : () -> ()
  hir.call @foo() : () -> ()

  %int_type = hir.int_type
  // CHECK: [[C42:%.+]] = mir.constant #mir.int<42>
  %c42 = hir.constant_int 42
  // CHECK: mir.constant #mir.type<!mir.uint<42>>
  %uint42_type = hir.uint_type %c42

  // CHECK: mir.call @foo([[C42]]) : (!mir.int) -> !mir.uint<42>
  hir.call @foo(%c42) : (%int_type) -> (%uint42_type)

  hir.return
}

// CHECK-LABEL: @FunctionSpecialization
hir.func @FunctionSpecialization {
  // CHECK: [[TYPE:%.+]] = mir.constant #mir.type<!mir.int>
  %0 = hir.int_type
  // CHECK: [[VALUE:%.+]] = mir.constant #mir.int<42>
  %1 = hir.constant_int 42
  // CHECK: mir.constant #mir.type<() -> ()>
  %2 = hir.func_type () -> ()
  // CHECK: [[FUNC:%.+]] = mir.constant #mir.func<@bar>
  %3 = hir.constant_func @bar : %2
  // CHECK: [[SPEC:%.+]] = mir.specialize_func @foo([[TYPE]]) -> (), [[VALUE]], [[FUNC]]
  %4 = hir.specialize_func @foo(%0) -> (), %1, %3
  // CHECK: mir.return [[SPEC]]
  hir.return (%4) : (%0)
}

// CHECK-LABEL: @Casts
hir.func @Casts {
  // CHECK-NEXT: [[TMP1:%.+]] = mir.constant
  %a0 = mir.constant #mir.int<42>
  %a1 = builtin.unrealized_conversion_cast %a0 : !mir.int to !hir.any

  // CHECK-NEXT: [[TMP2:%.+]] = mir.constant
  %b0 = mir.constant #mir.func<@foo> : () -> ()
  %b1 = builtin.unrealized_conversion_cast %b0 : () -> () to !hir.any

  %ta = hir.int_type
  %tb = hir.anyfunc_type
  // CHECK: mir.return [[TMP1]], [[TMP2]]
  hir.return (%a1, %b1) : (%ta, %tb)
}
