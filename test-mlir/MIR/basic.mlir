// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s

// Types
unrealized_conversion_cast to !mir.type
unrealized_conversion_cast to !mir.int
unrealized_conversion_cast to !mir.specialized_func

// Attributes
unrealized_conversion_cast to index {attr = #mir.type<!mir.int>}
unrealized_conversion_cast to index {attr = #mir.int<98765432109876543210987654321>}
unrealized_conversion_cast to index {attr = #mir.specialized_func<@foo, [!mir.int], [!mir.type], [#mir.int<42> : !mir.int]>}

// CHECK: %c42_int = mir.constant
%c42_int = mir.constant #mir.int<42>

// CHECK: %int_type = mir.constant
%int_type = mir.constant #mir.type<!mir.int>

mir.specialize_func @foo() -> ()
mir.specialize_func @foo(%int_type) -> (%int_type)
mir.specialize_func @foo(%int_type) -> (%int_type), %c42_int : !mir.int
