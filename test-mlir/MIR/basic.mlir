// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s

// Types
unrealized_conversion_cast to !mir.type
unrealized_conversion_cast to !mir.int

// Attributes
unrealized_conversion_cast to index {attr = #mir.type<!mir.int>}
unrealized_conversion_cast to index {attr = #mir.int<98765432109876543210987654321>}

// CHECK: %c42_int = mir.constant
mir.constant #mir.int<42>

// CHECK: %int_type = mir.constant
mir.constant #mir.type<!mir.int>
