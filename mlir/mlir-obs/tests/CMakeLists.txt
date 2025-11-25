add_mlir_test_library(MLIROBSTest
  string_encryption.mlir
  symbol_encryption.mlir
)

add_lit_testsuite(check-mlir-obs
  "Running mlir-obs tests"
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS mlir-opt
  )
