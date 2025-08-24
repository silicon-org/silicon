// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s

// Types
unrealized_conversion_cast to !mir.int

// Attributes
unrealized_conversion_cast to index {attr = #mir.int<98765432109876543210987654321>}

// Ops
mir.constant #mir.int<42>
