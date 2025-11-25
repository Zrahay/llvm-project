add_mlir_library(MLIROBS
  Passes.cpp
  SymbolPass.cpp
  PassRegistration.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)
