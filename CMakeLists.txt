cmake_minimum_required(VERSION 3.13.4)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Generate a `compile_commands.json` file for use by various tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===------------------------------------------------------------------------===#
# Build Configuration
#===------------------------------------------------------------------------===#

# Configure either a standalone build where CIRCT, MLIR, and LLVM are dug up as
# external dependencies; or as a unified build where silicon is built as an
# external project to LLVM, with LLVM driving the overall build.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Standalone Build

  message(FATAL_ERROR "Standalone build not yet supported")
else()
  # Unified Build

  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  set(CIRCT_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(CIRCT_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../../include) # --includedir
  set(CIRCT_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/circt/include)
  include_directories(SYSTEM ${CIRCT_INCLUDE_DIR})
  include_directories(SYSTEM ${CIRCT_TABLEGEN_OUTPUT_DIR})
endif()

#===------------------------------------------------------------------------===#
# Silicon Configuration
#===------------------------------------------------------------------------===#

set(SILICON_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SILICON_MAIN_INCLUDE_DIR ${SILICON_MAIN_SRC_DIR}/include)

set(SILICON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SILICON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SILICON_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(SILICON_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddSilicon)

# Installing the headers and docs needs to depend on generating any public
# tablegen outputs.
add_custom_target(silicon-headers)
add_custom_target(silicon-doc)
add_custom_target(silicon-tools)
add_custom_target(silicon-libraries)

# Add CIRCT, MLIR, and LLVM headers to the include path.
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

# Add Silicon files to the include path
include_directories(${SILICON_MAIN_INCLUDE_DIR})
include_directories(${SILICON_INCLUDE_DIR})

#===------------------------------------------------------------------------===#
# Directory Setup
#===------------------------------------------------------------------------===#

add_subdirectory(include/silicon)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
