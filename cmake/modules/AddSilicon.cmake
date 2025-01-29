include_guard()

function(add_silicon_tool name)
  add_llvm_executable(${name} ${ARGN})
  add_dependencies(silicon-tools ${name})
endfunction()
