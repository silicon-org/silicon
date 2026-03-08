# Modules

This document describes the design of hardware modules in Silicon.
It builds on the phased execution model described in {{< page-link "/design/phase-splits" >}}.
It is aspirational and does not reflect the current state of the code base.

## Overview

A hardware module is declared with the `mod` keyword.
Syntactically, modules look like functions: inputs are arguments, outputs are return values.
Semantically, a `mod` works just like a `fn` — it goes through the same phased execution pipeline, the same `const`/`dyn` annotations apply, and the same specialization rules hold.
The `mod` keyword simply signals that the function describes hardware and will eventually be lowered to a hardware module rather than being fully evaluated away.

```silicon
mod counter(const WIDTH: int, clk: clock, rst: reset, en: bit) -> (count: uint<WIDTH>) {
  let zero: uint<WIDTH> = 0;
  let r = reg(clk, rst, zero);
  r.d = mux(en, r.q + 1, r.q);
  r.q
}
```

Arguments to a `mod` that are not marked `const` are not implicitly `dyn` — they simply follow the normal phase rules.
A non-`const` argument that receives a constant value at a call site can still be monomorphized and specialized just like in a regular function.
The key difference from `fn` is that the final-phase body of a `mod` is lowered to CIRCT hardware IR instead of being evaluated or compiled to machine code.

The phased execution pipeline handles `const` parameters as it does for regular functions: it evaluates them at compile time and specializes the module.
After all phases have been resolved, the remaining function body is lowered to CIRCT's `hw.module` and related ops.

## Ports

Ports follow functional style: arguments are inputs, return values are outputs.

```silicon
mod alu(a: uint<32>, b: uint<32>, op: uint<2>) -> (result: uint<32>) {
  // ...
}
```

This maps directly to an `hw.module` with three input ports and one output port.
Named return values become the output port names.

### Parameterized Modules

Since `const` arguments are compile-time parameters, parameterized modules are just modules with `const` args:

```silicon
mod fifo(const DEPTH: int, const WIDTH: int, push: bit, pop: bit, din: uint<WIDTH>)
    -> (dout: uint<WIDTH>, full: bit, empty: bit) {
  // DEPTH and WIDTH are resolved at compile time
  // push, pop, din, dout, full, empty are hardware ports
}
```

No separate `parameter` mechanism is needed.

## Instantiation

Module instantiation uses ordinary function call syntax.
The compiler knows that calling a `mod` creates a persistent hardware instance, not a one-shot computation.

```silicon
mod top(clk: clock, rst: reset) -> (out: uint<8>) {
  let en: bit = 1;
  let c = counter(8, clk, rst, en);
  c
}
```

The call `counter(8, clk, rst, en)` instantiates a `counter` module with `WIDTH = 8`.
The phased execution pipeline evaluates the `const` argument `8` at compile time and specializes the module, producing `counter$WIDTH8` (or similar) with concrete 8-bit types.
The remaining arguments become port connections on the instance.

Multiple calls to the same `mod` create distinct instances:

```silicon
mod dual_counter(clk: clock, rst: reset, en_a: bit, en_b: bit)
    -> (a: uint<8>, b: uint<8>) {
  let ca = counter(8, clk, rst, en_a);
  let cb = counter(8, clk, rst, en_b);
  (ca, cb)
}
```

## Interop between `mod` and `fn`

A `mod` can call a `fn`, and a `fn` can call a `mod`, without restriction.
Since both go through the same phased execution pipeline, the compiler handles the distinction transparently.
A `fn` called from a `mod` that survives to the final phase is inlined into the hardware module as combinational logic.
A `mod` called from a `fn` instantiates a hardware module as part of the enclosing design.

## State and Wires

Silicon provides dedicated constructs for hardware connectivity and state: wires, registers, latches, and memories.
See {{< page-link "/design/state" >}} for the full design, including the `reg` constructor signature and implicit hold behavior (a register retains its value when `.d` is not assigned).
All four follow a common pattern: they are expressions that return a first-class typed value with `.d` (input) and `.q` (output) accessors.

## Clock and Reset

Clock and reset are explicit values in Silicon, passed as arguments to modules and stateful constructs.

```silicon
mod my_module(clk: clock, rst: reset, data_in: uint<8>) -> (data_out: uint<8>) {
  let zero: uint<8> = 0;
  let r = reg(clk, rst, zero);
  r.d = data_in;
  r.q
}
```

This approach:

- Makes clock domain crossings explicit — there is no hidden implicit clock.
- Allows multi-clock designs naturally: different registers can use different clocks.
- Avoids the "implicit clock" problem where it's unclear which clock drives what.

The downside is verbosity: `clk` and `rst` must be threaded through every module and every `reg`.
A future extension could introduce clock domain blocks to reduce this boilerplate:

```silicon
// Possible future syntax — not part of the initial design.
mod my_module(clk: clock, rst: reset, data_in: uint<8>) -> (data_out: uint<8>) {
  clocked(clk, rst) {
    let zero: uint<8> = 0;
    let r = reg(zero);  // clock and reset inherited from block
    r.d = data_in;
    r.q
  }
}
```

For now, explicit clock and reset threading is the only supported approach.

## Lowering to CIRCT

After phased evaluation resolves all `const` parameters, the remaining module body is in MIR form with concrete types.
The compiler lowers this to CIRCT IR:

- `mod` becomes `hw.module` with typed input/output ports.
- Module instantiation becomes `hw.instance`.
- Combinational logic (arithmetic, muxes, etc.) becomes `comb` dialect ops.
- State and wire lowering is described in {{< page-link "/design/state" >}}.

The mapping is intentionally direct — Silicon's hardware constructs are designed to have a straightforward correspondence with CIRCT's ops, rather than requiring complex lowering.

## Examples

### Parameterized Shift Register

```silicon
mod shift_reg(const N: int, clk: clock, rst: reset, din: bit) -> (dout: bit) {
  let zero: uint<N> = 0;
  let din_wide: uint<N> = din;
  let r = reg(clk, rst, zero);
  r.d = (r.q << 1) | din_wide;
  r.q[N - 1]
}
```

### ALU

```silicon
mod alu(const WIDTH: int, a: uint<WIDTH>, b: uint<WIDTH>, op: uint<2>)
    -> (result: uint<WIDTH>) {
  mux(op, [a + b, a - b, a & b, a | b])
}
```

Note that the ALU is purely combinational — no `reg` or other state.
A `mod` without state is valid; it simply becomes an `hw.module` with only combinational logic.

### Counter with Saturation

```silicon
mod sat_counter(const WIDTH: int, clk: clock, rst: reset, en: bit, up: bit)
    -> (count: uint<WIDTH>) {
  let max: uint<WIDTH> = (1 << WIDTH) - 1;
  let zero: uint<WIDTH> = 0;
  let r = reg(clk, rst, zero);
  let inc = mux(r.q != max, r.q + 1, r.q);
  let dec = mux(r.q != zero, r.q - 1, r.q);
  let next = mux(up, inc, dec);
  r.d = mux(en, next, r.q);
  r.q
}
```
