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
  %a0 = hir.func_type () -> ()
  %a1 = hir.constant_func @foo : %a0
  hir.call %a1() : () -> ()

  // CHECK: [[TMP:%.+]] = mir.constant #mir.int<42>
  // CHECK: mir.call @foo([[TMP]]) : (!mir.int) -> !mir.uint<42>
  %b0 = hir.int_type
  %b1 = hir.constant_int 42
  %b2 = hir.uint_type %b1
  %b3 = hir.func_type (%b0 : !hir.type) -> (%b2 : !hir.type)
  %b4 = hir.constant_func @foo : %b3
  hir.call %b4(%b1) : (!hir.value) -> (!hir.value)

  hir.return
}

// CHECK-LABEL: @FunctionSpecialization
hir.func @FunctionSpecialization {
  // CHECK-NEXT: [[TYPE:%.+]] = mir.constant #mir.type<!mir.int>
  %0 = hir.int_type
  // CHECK-NEXT: [[VALUE:%.+]] = mir.constant #mir.int<42>
  %1 = hir.constant_int 42
  // CHECK-NEXT: mir.constant #mir.type<() -> ()>
  %2 = hir.func_type () -> ()
  // CHECK-NEXT: [[FUNC:%.+]] = mir.constant #mir.func<@bar>
  %3 = hir.constant_func @bar : %2
  // CHECK: [[SPEC:%.+]] = mir.specialize_func @foo([[TYPE]]) -> (), [[VALUE]], [[FUNC]]
  %4 = hir.specialize_func @foo(%0) -> (), %1, %3 : !hir.value, !hir.func
  // CHECK: mir.return [[SPEC]]
  hir.return %4 : !hir.func
}

// CHECK-LABEL: @Cast
hir.func @Cast {
  // CHECK-NEXT: [[TMP:%.+]] = mir.constant
  %0 = mir.constant #mir.int<42>
  %1 = builtin.unrealized_conversion_cast %0 : !mir.int to !hir.value
  // CHECK-NEXT: mir.return [[TMP]]
  hir.return %1 : !hir.value
}
