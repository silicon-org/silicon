// RUN: silicon-opt --infer-types %s | FileCheck %s

// CHECK: [[T:%.+]] = hir.inferrable_type
// CHECK: [[X:%.+]] = hir.let "x" : [[T]]
// CHECK: [[Y:%.+]] = hir.let "y" : [[T]]
// CHECK: hir.store [[X]], [[Y]] : [[T]]
%0 = hir.inferrable_type
%1 = hir.let "x" : %0
%2 = hir.inferrable_type
%3 = hir.let "y" : %2
%4 = hir.unify_type %0, %2
hir.store %1, %3 : %4
