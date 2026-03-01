# Silicon

An experimental hardware language and compiler.

## Setup

Clone the repository and the submodules:

```
git clone --recurse-submodules --shallow-submodules git@github.com:silicon-org/silicon.git
```

Configure the build:

```
cmake -S circt/llvm/llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_EXTERNAL_PROJECTS="circt;silicon" \
  -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD/circt \
  -DLLVM_EXTERNAL_SILICON_SOURCE_DIR=$PWD \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Run the build:

```
ninja -C build check-silicon      # build and run all tests
ninja -C build silicon            # build everything
ninja -C build silicon-docs       # build documentation
ninja -C build silicon-tools      # build tools such as silc and silicon-opt
ninja -C build silicon-libraries  # build everything else
```
