configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(deps
  FileCheck count not
  silicon-opt
)

add_lit_testsuite(
  check-silicon "Running the Silicon regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${deps}
)
