// RUN: silicon-opt --lower-hir-to-mir %s | FileCheck %s

// CHECK-LABEL: @Types
hir.func @Types {
  // CHECK-NEXT: mir.constant #mir.type<!mir.int>
  %int_type = hir.int_type

  // CHECK-NEXT: mir.constant #mir.int<42>
  // CHECK-NEXT: mir.constant #mir.type<!mir.uint<42>>
  %c42_int = hir.constant_int 42
  %uint42_type = hir.uint_type %c42_int

  // CHECK-NEXT: mir.constant #mir.type<() -> ()>
  hir.func_type () -> ()
  // CHECK-NEXT: mir.constant #mir.type<(!mir.int) -> !mir.uint<42>>
  hir.func_type (%int_type : !hir.type) -> (%uint42_type : !hir.type)

  hir.return
}

// CHECK-LABEL: @Calls
hir.func @Calls {
  // CHECK: mir.call @foo() : () -> ()
  %a0 = hir.constant_func @foo
  %a1 = hir.func_type () -> ()
  hir.call %a0() : %a1, () -> ()

  // CHECK: [[TMP:%.+]] = mir.constant #mir.int<42>
  // CHECK: mir.call @foo([[TMP]]) : (!mir.int) -> !mir.uint<42>
  %b0 = hir.constant_func @foo
  %b1 = hir.int_type
  %b2 = hir.constant_int 42
  %b3 = hir.uint_type %b2
  %b4 = hir.func_type (%b1 : !hir.type) -> (%b3 : !hir.type)
  hir.call %b0(%b2) : %b4, (!hir.value) -> (!hir.value)

  hir.return
}

// CHECK-LABEL: @FunctionSpecialization
hir.func @FunctionSpecialization {
  // CHECK-NEXT: [[TYPE:%.+]] = mir.constant #mir.type<!mir.int>
  %0 = hir.int_type
  // CHECK-NEXT: [[VALUE:%.+]] = mir.constant #mir.int<42>
  %1 = hir.constant_int 42
  // CHECK-NEXT: [[FUNC:%.+]] = mir.constant #mir.func<@bar>
  %2 = hir.constant_func @bar
  // CHECK: [[SPEC:%.+]] = mir.specialize_func @foo([[TYPE]]) -> (), [[VALUE]], [[FUNC]]
  %3 = hir.specialize_func @foo(%0) -> (), %1, %2 : !hir.value, !hir.func
  // CHECK: mir.return [[SPEC]]
  hir.return %3 : !hir.func
}

// CHECK-LABEL: @Cast
hir.func @Cast {
  // CHECK-NEXT: [[TMP:%.+]] = mir.constant
  %0 = mir.constant #mir.int<42>
  %1 = builtin.unrealized_conversion_cast %0 : !mir.int to !hir.value
  // CHECK-NEXT: mir.return [[TMP]]
  hir.return %1 : !hir.value
}
