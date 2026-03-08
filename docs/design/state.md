# State and Wires

This document describes the design of stateful elements and wires in Silicon.
It is aspirational and does not reflect the current state of the code base.

## Overview

Silicon provides four dedicated constructs for hardware connectivity and state: wires, registers, latches, and memories.
All four follow a common pattern: they are expressions that return a first-class typed value with `.d` (input) and `.q` (output) accessors.
They can be passed to functions, returned, and composed like any other value.

These constructs are not limited to modules — they can appear anywhere in the language, including inside regular functions.
Stateful elements must appear in a phase where their `.d` and `.q` ports are dynamic — they cannot be evaluated at compile time.
During phased evaluation, stateful ops are generated but not interpreted; if the interpreter encounters a stateful element (because it ended up in a phase that is being evaluated), it reports an error to the user.

## Wires

A `wire` is an unclocked, stateless connection point.
It has type `wire<T>` with `.d` and `.q` accessors — `.q` reflects `.d` combinationally within the same cycle, with no clock or storage involved.

```silicon
let w = wire<uint<8>>();  // w: wire<uint<8>>
w.d = some_expr;
let out = w.q;            // out == some_expr, combinationally
```

Wires are useful for building complex combinational logic where a value needs to be driven from one place and read from another — for example, when multiple `if` branches drive the same signal, or when connecting submodule ports.

```silicon
let result = wire<uint<8>>();
if sel {
  result.d = a;
} else {
  result.d = b;
}
```

Since `wire<T>` is a first-class type, wires follow the same expression and composition rules as registers and latches: they can be passed to functions, returned, and nested in expressions.

## Registers

A `reg` is a clocked storage element that updates on a clock edge.
It takes a clock, an optional asynchronous reset with initial value, and returns a value of type `reg<T>`.
The reset is always asynchronous — synchronous reset behavior can be expressed as a mux on the D input, just like a clock enable.
A register has two sides, accessed via `.d` (D input) and `.q` (Q output):

- `.q` reads the register's current value (the Q output, type `T`).
- `.d` is assigned to drive the register's next-cycle value (the D input).

```silicon
// Register with asynchronous reset and initial value.
let r = reg(clk, rst, 0 as uint<8>);  // r: reg<uint<8>>

// Register without reset.
let r = reg(clk) as reg<uint<8>>;
```

```silicon
r.d = r.q + 1;  // read current value, drive next value
```

On the next clock edge, `r.q` will reflect the value that was driven into `r.d`.
This matches the semantics of `seq.firreg` in CIRCT.

A clock-enabled register only updates when an enable signal is high:

```silicon
let r = reg(clk, rst, 0 as uint<8>);
if en { r.d = r.q + 1; }
// equivalent to: r updates to r.q+1 when en is high, holds otherwise
```

Since `reg<T>` is a first-class type, registers can be passed to and returned from functions:

```silicon
fn increment(r: reg<uint<8>>) {
  r.d = r.q + 1;
}

fn make_counter(clk: clock, rst: reset) -> reg<uint<8>> {
  let r = reg(clk, rst, 0 as uint<8>);
  r.d = r.q + 1;
  r
}
```

In hardware terms, passing a `reg<T>` gives the callee access to the register's ports — this is inherently pass-by-reference, since hardware values are not copied.

Since `reg<T>` is a proper type with `.d`/`.q` accessors, `reg(...)` is just an expression — no special declaration syntax is needed.
Registers can appear anywhere an expression can: bound to a variable, passed directly to a function, returned, or nested in other expressions.

## Latches

A `latch` is a level-sensitive storage element that returns a value of type `latch<T>`.
Like registers, latches have `.d` and `.q` accessors:

- `.q` reads the latch's current output (type `T`).
- `.d` drives the latch's data input.

When the enable signal is high, the latch is transparent and `.q` follows `.d`.
When the enable is low, the latch holds its last value.

```silicon
let l = latch(en, 0 as uint<8>);  // l: latch<uint<8>>
l.d = some_value;
let out = l.q;
```

Since `latch<T>` is a first-class type, latches can be passed to and returned from functions, just like registers.

Latches are generally discouraged in synchronous design but are sometimes necessary for specific use cases (e.g., clock gating cells).
The compiler could emit a warning when latches are used, unless explicitly silenced.

## Memories

A `mem` declares a multi-entry storage array with a fixed number of read, write, and read-write ports.
Unlike a raw array of registers, a memory's access is constrained to a declared set of ports, which maps to efficient hardware (SRAMs, block RAMs).

The number of each port type is specified upfront when the memory is created.
This makes the port structure part of the memory's type — `mem<T, DEPTH, R, W, RW>` — and ensures the hardware mapping is known at compile time.

```silicon
// 256-entry memory of uint<8>, with 1 read port, 1 write port, 0 read-write ports.
let m = mem<uint<8>>(256, clk, read: 1, write: 1);

// 2 read ports, 1 write port, 1 read-write port.
let m = mem<uint<8>>(1024, clk, read: 2, write: 1, read_write: 1);
```

Ports are accessed by index:

```silicon
// Read port 0: provide address, get data.
let data = m.read[0].q;
m.read[0].addr = addr;

// Write port 0: provide address, data, and enable.
m.write[0].addr = waddr;
m.write[0].data = wdata;
m.write[0].en = wen;

// Read-write port 0.
m.rw[0].addr = rwaddr;
m.rw[0].wdata = wdata;
m.rw[0].wen = wen;
let rd = m.rw[0].rdata;
```

The port indices are checked at compile time — accessing `m.read[1]` on a memory with only one read port is an error.

Since `mem<...>` is a first-class type, memories can be passed to and returned from functions.

## Lowering to CIRCT

- `wire` becomes an `hw.wire` or a direct SSA value connection.
- `reg` becomes `seq.firreg`.
- `latch` becomes the appropriate latch op in CIRCT.
- `mem` becomes `seq.firmem` with `seq.firmem.read_port` and `seq.firmem.write_port`.
