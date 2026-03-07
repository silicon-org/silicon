---
weight: 100
---

# Open Questions

This document collects open design questions across the Silicon language.

## Modules

- **Clock domain crossings.** Multi-clock designs are supported by passing different `clock` values, but safe CDC (synchronizers, FIFOs) is left to library-level constructs for now.
- **Module interfaces / bundles.** Should there be a way to group related ports into a named interface or bundle type for reuse? This would reduce boilerplate for standard bus interfaces.
- **Multi-value return syntax.** How does a bare tuple return like `(ca, cb)` map to named output ports `(a: uint<8>, b: uint<8>)`? Is the mapping purely positional? Should we require named returns at the call site?
- **Bit indexing syntax.** The `r.q[N - 1]` syntax is used for bit extraction but is never formally introduced. Should `[]` be bit indexing, array indexing, or overloaded? How does it interact with slicing (e.g., `r.q[3:0]`)?
- **`as` cast precedence.** The precedence of `as` relative to comparison and arithmetic operators is not yet specified. For example, does `x != y as uint<N>` parse as `(x != y) as uint<N>` or `x != (y as uint<N>)`? We should define this clearly and consider whether typed `let` bindings are always sufficient to avoid `as` casts.

## Memories

- **Read latency.** Should memories support configurable read latency? CIRCT's `seq.firmem` supports this. A possible syntax is an additional `latency: 1` parameter.
- **Clock per-memory or per-port.** Per-memory is simpler; per-port allows dual-clock memories.

## Clock and Reset

- **Register initialization vs. reset.** FPGA designs often distinguish between the initial value loaded at configuration time and the value driven by a reset signal. Should `reg` support both?
