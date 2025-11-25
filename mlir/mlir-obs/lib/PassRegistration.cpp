#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::obs;

namespace {
  PassRegistration<StringEncryptPass>
    stringReg("string-encrypt", "Encrypt string attributes using XOR");

  PassRegistration<SymbolObfuscatePass>
    symbolReg("symbol-obfuscate", "Obfuscate symbol names randomly");
}
