add_mlir_library(MLIROBS
  Passes.cpp
  SymbolPass.cpp

  DEPENDS
    MLIROBSIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)
