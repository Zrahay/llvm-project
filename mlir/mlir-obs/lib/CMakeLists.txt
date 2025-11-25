add_mlir_library(MLIROBS
  Passes.cpp
  SymbolPass.cpp
  PassRegistrations.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)
