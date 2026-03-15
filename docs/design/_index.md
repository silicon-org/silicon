---
bookFlatSection: true
weight: 30
---

# Design

Design documents describing the goals and architecture of the Silicon language and compiler.

## Open Questions

This section collects open design questions across the Silicon language.

### Modules

See {{< page-link "/design/modules" >}}.

- **Clock domain crossings.** Multi-clock designs are supported by passing different `clock` values, but safe CDC (synchronizers, FIFOs) is left to library-level constructs for now.
- **Module interfaces / bundles.** Should there be a way to group related ports into a named interface or bundle type for reuse? This would reduce boilerplate for standard bus interfaces.
- **Multi-value return syntax.** How does a bare tuple return like `(ca, cb)` map to named output ports `(a: uint<8>, b: uint<8>)`? Is the mapping purely positional? Should we require named returns at the call site?
- **Bit indexing syntax.** The `r.q[N - 1]` syntax is used for bit extraction but is never formally introduced. Should `[]` be bit indexing, array indexing, or overloaded? How does it interact with slicing (e.g., `r.q[3:0]`)?
- **`as` cast precedence.** The precedence of `as` relative to comparison and arithmetic operators is not yet specified. For example, does `x != y as uint<N>` parse as `(x != y) as uint<N>` or `x != (y as uint<N>)`? We should define this clearly and consider whether typed `let` bindings are always sufficient to avoid `as` casts.

### Memories

See {{< page-link "/design/state" >}}.

- **Read latency.** Should memories support configurable read latency? CIRCT's `seq.firmem` supports this. A possible syntax is an additional `latency: 1` parameter.
- **Clock per-memory or per-port.** Per-memory is simpler; per-port allows dual-clock memories.

### Inference

See {{< page-link "/design/inference" >}} for the design document on type and value inference.
The open questions at the end of that document cover subtyping, coercion semantics, per-op type rules, generics, bitwidth inference, and the interaction between inference and phased execution.

### Clock and Reset

See {{< page-link "/design/state" >}}.

- **Register initialization vs. reset.** FPGA designs often distinguish between the initial value loaded at configuration time and the value driven by a reset signal. Should `reg` support both?

### Arrays and Slicing

See {{< page-link "/design/arbiter-example" >}} for a worked example that exercises arrays, slicing, and compile-time recursion.

- **Array type syntax.** `[T; N]` with const-dependent sizes, following Rust convention.
- **Bit vs. array indexing.** Should `v[i]` on a `uint<N>` extract a single bit (returning `bool`), while `a[i]` on `[T; N]` extracts an element? Or should the syntax be different for each?
- **Slicing semantics.** `v[lo..hi]` with exclusive upper bound for both bit slicing on integers and element slicing on arrays.
- **Array concatenation operator.** `a ++ b` to concatenate two arrays. How does this interact with bit concatenation (`cat`)?
