# Round-Robin Arbiter Example

This document sketches how a parameterized round-robin arbiter would look in Silicon.
The arbiter takes N ready-valid channels with request data on the valid side and response data on the ready side, and reduces them to a single ready-valid channel using a binary reduction tree.
Each tree level halves the number of channels by picking between pairs, and routes the response back along the same tree structure.

The example exercises several language features that are not yet implemented but follow naturally from Silicon's phased execution model:
arrays, bit/array slicing, compile-time recursion via `if const`, and dependent-type arithmetic on array sizes.
It is meant as a design target, not a currently compilable program.


## Proposed Language Extensions

The arbiter relies on a few constructs beyond what Silicon currently implements.

### Arrays

Fixed-size arrays with const-dependent sizes:

```silicon
let a: [uint<8>; 4] = [0, 0, 0, 0];   // array literal
let x = a[2];                           // element access (bool index or uint)
```

The size `N` in `[T; N]` may be any const expression, enabling dependent-sized arrays like `[bool; 1 << DEPTH]`.

### Bit and Array Slicing

Single-bit extraction on an unsigned integer returns a `bool`:

```silicon
let v: uint<8> = 0xA5;
let b: bool = v[3];         // extract bit 3
```

Range slicing on integers produces a narrower integer, and on arrays produces a smaller array.
Ranges use exclusive upper bounds, following Rust convention:

```silicon
let lo: uint<4> = v[0..4];  // bits 3 down to 0
let hi: uint<4> = v[4..8];  // bits 7 down to 4

let arr: [bool; 8] = /* ... */;
let first_half: [bool; 4] = arr[0..4];
let second_half: [bool; 4] = arr[4..8];
```

### Bit Concatenation

`cat(a, b)` concatenates two unsigned integers, placing `a` in the high bits:

```silicon
let hi: uint<1> = 1;
let lo: uint<3> = 5;
let result: uint<4> = cat(hi, lo);  // 0b1_101 = 13
```

### Array Concatenation

The `++` operator concatenates two arrays:

```silicon
let left:  [bool; 2] = [true, false];
let right: [bool; 3] = [true, true, false];
let all:   [bool; 5] = left ++ right;
```

### Compile-Time Recursion

A function with a `const` depth parameter and an `if const` base case naturally produces a compile-time recursive expansion.
The compiler unrolls each specialization at compile time, generating a flat circuit:

```silicon
fn tree<const DEPTH: int>(x: [bool; 1 << DEPTH]) -> bool {
    if const DEPTH == 0 {
        x[0]
    } else {
        const HALF: int = 1 << (DEPTH - 1);
        tree<DEPTH - 1>(x[0..HALF]) || tree<DEPTH - 1>(x[HALF..2*HALF])
    }
}
```


## Implementation

### Overview

The arbiter is split into three pieces:

1. **`arb_fwd`** -- a compile-time-recursive function that builds the forward reduction tree.
   It takes `2^DEPTH` valid-data pairs and reduces them to one, returning the index of the winner.

2. **`arb_rev`** -- a matching recursive function that builds the reverse expansion tree.
   It takes a single ready-response pair and fans it out to `2^DEPTH` channels, gating ready by the winner index.

3. **`arbiter`** -- the top-level module that wires the two trees together and holds the locking and priority state.


### Compile-Time Helpers

```silicon
/// Compute floor(log2(n)). Requires n to be a positive power of two.
const fn log2(n: int) -> int {
    if n == 1 { 0 } else { 1 + log2(n / 2) }
}
```

This is a pure `const fn`: it runs entirely at compile time and produces an `int` that can be used in type expressions like `uint<log2(N)>`.


### Forward Reduction Tree

The forward tree reduces N valid-data pairs to a single output.
At each level, it compares the valid signals of two children and picks one based on the priority derived from `last_idx`.
When the arbiter is locked, the tree is forced to follow the locked selection instead.

```silicon
//===----------------------------------------------------------------------===//
// Forward Reduction Tree
//
// Each level of the tree pairs up adjacent channels and picks between them.
// The `prio` bits control which side is preferred at each level: if the
// corresponding bit of `last_idx` is 0 the left side was last picked, so we
// prefer the right side, and vice versa.
//
// When `locked` is true, the tree ignores fresh arbitration and forces each
// level's pick to match the corresponding bit of `lock_idx`, so the same
// input stays selected until the handshake completes.
//
// The winner index is assembled from the pick bits: the top-level pick
// becomes the MSB, and recursive sub-winners form the lower bits.
//===----------------------------------------------------------------------===//

fn arb_fwd<const DEPTH: int, const DW: int>(
    valid: [bool; 1 << DEPTH],
    data:  [uint<DW>; 1 << DEPTH],
    prio:     uint<DEPTH>,
    lock_idx: uint<DEPTH>,
    locked:   bool,
) -> (
    out_valid: bool,
    out_data:  uint<DW>,
    winner:    uint<DEPTH>,
) {
    if const DEPTH == 0 {
        // Base case: single channel, pass through.
        (valid[0], data[0], 0)
    } else {
        const HALF: int = 1 << (DEPTH - 1);

        // Recurse into left and right halves.
        let (lv, ld, lw) = arb_fwd<DEPTH - 1, DW>(
            valid[0..HALF], data[0..HALF],
            prio[0..DEPTH - 1], lock_idx[0..DEPTH - 1], locked,
        );
        let (rv, rd, rw) = arb_fwd<DEPTH - 1, DW>(
            valid[HALF..2 * HALF], data[HALF..2 * HALF],
            prio[0..DEPTH - 1], lock_idx[0..DEPTH - 1], locked,
        );

        // Arbitrate between left and right at this level.  If the last
        // winner was on the left side (prio bit == 0), prefer right.
        let prefer_right = !prio[DEPTH - 1];
        let fresh_pick = mux(prefer_right,
            rv || !lv,    // prefer right: pick right unless only left is valid
            rv && !lv,    // prefer left:  pick right only if left is invalid
        );
        let pick_right = mux(locked, lock_idx[DEPTH - 1], fresh_pick);

        // Mux data and assemble the winner index.
        let out_valid = lv || rv;
        let out_data = mux(pick_right, rd, ld);
        let sub_winner = mux(pick_right, rw, lw);
        let winner = cat(pick_right, sub_winner);

        (out_valid, out_data, winner)
    }
}
```


### Reverse Expansion Tree

The reverse tree mirrors the forward tree's structure.
It takes the single ready-response pair at the output and routes it back to the winning input channel, using the same winner index that the forward tree produced.

```silicon
//===----------------------------------------------------------------------===//
// Reverse Expansion Tree
//
// At each level, the winner index's top bit tells us which child was
// selected.  Ready is gated so only the winning path receives it.
// Response data is broadcast to all channels; only the channel that sees
// ready asserted will consume it.
//===----------------------------------------------------------------------===//

fn arb_rev<const DEPTH: int, const RW: int>(
    ready:  bool,
    resp:   uint<RW>,
    winner: uint<DEPTH>,
) -> (
    out_ready: [bool; 1 << DEPTH],
    out_resp:  [uint<RW>; 1 << DEPTH],
) {
    if const DEPTH == 0 {
        ([ready], [resp])
    } else {
        let pick_right = winner[DEPTH - 1];

        // Route ready only to the selected subtree.
        let (lr, lresp) = arb_rev<DEPTH - 1, RW>(
            ready && !pick_right, resp,
            winner[0..DEPTH - 1],
        );
        let (rr, rresp) = arb_rev<DEPTH - 1, RW>(
            ready && pick_right, resp,
            winner[0..DEPTH - 1],
        );

        (lr ++ rr, lresp ++ rresp)
    }
}
```


### Top-Level Arbiter

The top-level module instantiates the forward and reverse trees and manages three registers:

- **`last_idx`** -- the index of the last input that completed a handshake (valid && ready).
  Its bits set the priority at each tree level: the side that contains the last winner has lowest priority.

- **`locked`** -- asserted when the arbiter has picked a valid input but the downstream consumer has not yet acknowledged with ready.
  While locked, the forward tree is forced to maintain the current selection so that the output data does not change mid-handshake.

- **`locked_idx`** -- captures the winner index on the cycle that locking engages, so the tree can be forced to the correct input on subsequent locked cycles.

```silicon
//===----------------------------------------------------------------------===//
// Top-Level Round-Robin Arbiter
//
// N must be a power of two.  The arbiter uses a binary reduction tree to
// pick one valid input per cycle.  A register tracks the last winner and
// shifts priority away from it at each tree level.  Once a valid input is
// selected, the arbiter locks onto it until the handshake completes
// (out_valid && out_ready), preventing output glitches.
//
// The tree-based priority is an approximation of strict round-robin:
// each level uses one bit of last_idx to flip its preference, which gives
// good fairness in practice but does not guarantee the exact rotation
// order of a true round-robin arbiter.  A strict implementation would
// require a rotating priority mask with O(N^2) comparators, whereas the
// tree approach is O(N log N).
//===----------------------------------------------------------------------===//

mod arbiter<const N: int, const DW: int, const RW: int>(
    clk: clock,
    rst: reset,

    // N input channels: valid + request data on the forward path.
    in_valid: [bool; N],
    in_data:  [uint<DW>; N],

    // Single output channel: ready + response data on the return path.
    out_ready: bool,
    out_resp:  uint<RW>,
) -> (
    // Arbitrated output: valid + request data.
    out_valid: bool,
    out_data:  uint<DW>,

    // Per-input ready + response data routed back.
    in_ready: [bool; N],
    in_resp:  [uint<RW>; N],
) {
    const DEPTH: int = log2(N);

    // State registers.
    let last_idx   = reg<uint<DEPTH>>(clk, rst, 0);
    let locked     = reg<bool>(clk, rst, false);
    let locked_idx = reg<uint<DEPTH>>(clk, rst, 0);

    // Forward tree: reduce N valid-data pairs to one.
    let (fwd_valid, fwd_data, winner) = arb_fwd<DEPTH, DW>(
        in_valid, in_data,
        last_idx.q, locked_idx.q, locked.q,
    );

    // Locking logic.  Lock whenever we present a valid output that has
    // not been acknowledged.  The winner index is captured on the
    // transition into the locked state; while locked, the tree is
    // forced to replay that same selection.
    let handshake = fwd_valid && out_ready;
    locked.d     = fwd_valid && !out_ready;
    locked_idx.d = winner;
    last_idx.d   = mux(handshake, winner, last_idx.q);

    // Forward outputs.
    out_valid = fwd_valid;
    out_data  = fwd_data;

    // Reverse tree: fan out ready and response to the winning input.
    let (rev_ready, rev_resp) = arb_rev<DEPTH, RW>(
        out_ready, out_resp, winner,
    );

    in_ready = rev_ready;
    in_resp  = rev_resp;
}
```


## Design Notes

### Approximate vs. Strict Round-Robin

The tree arbiter uses one bit of `last_idx` per tree level to flip the priority between left and right children.
This is a common hardware optimization (O(N log N) area and O(log N) delay), but it only approximates strict round-robin order.

Consider N=8 with `last_idx = 2` (binary `010`).
Strict round-robin would try inputs in order 3, 4, 5, 6, 7, 0, 1, 2.
The tree arbiter prefers right at level 2 (bit 1 = 1), left at level 1 (bit 0 = 0), and right at level 0 (bit 0 = 0).
Depending on which inputs are valid, this can skip an input that strict round-robin would select first.

A true round-robin would require rotating the priority across all N inputs, e.g. with a thermometer-coded mask and parallel priority logic.
That is straightforward in Silicon as well (a `for const` loop over all inputs with a comparator chain), but the tree version has better timing characteristics for large N.

### Phased Execution

The `const` parameters `N`, `DW`, `RW` are all resolved at compile time.
The entire tree structure -- every `if const DEPTH == 0` branch, every recursive call, every array slice -- is unrolled during phased evaluation.
What remains after phase evaluation is a flat combinational circuit with exactly N-1 two-input arbitration nodes, N-1 muxes, and the three output registers.
No runtime loops or recursion exist in the generated hardware.

This is the core value proposition of Silicon's phased execution for hardware design: structural parameters like tree depth, bus widths, and port counts are first-class compile-time values, and the language's existing `if const` / `for const` mechanisms generate the circuit topology without any special template or generate-block syntax.

### Locking Behavior

The arbiter locks onto a selected input for the duration of a single handshake.
While `out_valid` is high and `out_ready` is low, the `locked` register keeps the forward tree frozen on the same input.
This ensures that `out_data` does not glitch while the downstream consumer is sampling it.

The lock engages combinationally on the first cycle of a valid output (when `locked.q` is still low, the fresh arbitration result is used).
On the next cycle, if ready has not arrived, `locked.q` goes high and forces the tree via `locked_idx.q`.
When the handshake completes, `locked.d` drops and `last_idx` is updated, allowing the next arbitration to proceed with shifted priority.

### Ready-Valid Protocol Assumption

The arbiter assumes that each input channel follows the standard ready-valid protocol: once `in_valid[i]` is asserted, it must remain high and `in_data[i]` must remain stable until `in_ready[i]` is asserted.
The arbiter itself upholds this contract on the output side through the locking mechanism.
