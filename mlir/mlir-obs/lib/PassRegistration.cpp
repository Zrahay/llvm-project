#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
  // Register string encryption pass for mlir-opt
  PassRegistration<StringEncryptPass> 
    stringReg("string-encrypt",
              "Encrypt string attributes using XOR");

  // Register symbol obfuscation pass for mlir-opt
  PassRegistration<SymbolObfuscatePass> 
    symbolReg("symbol-obfuscate",
              "Obfuscate symbol names randomly");
}
