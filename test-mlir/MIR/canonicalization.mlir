// RUN: silicon-opt --canonicalize %s

// Constants must be foldable.
mir.constant #mir.int<42>
