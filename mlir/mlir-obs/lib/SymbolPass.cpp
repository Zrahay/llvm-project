#include "Obfuscator/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <random>

using namespace mlir;
using namespace mlir::obs;

namespace {

static std::string generateObfuscatedName(std::mt19937 &rng) {
  std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
  uint32_t num = dist(rng);

  char buffer[16];
  snprintf(buffer, sizeof(buffer), "f_%08x", num);
  return std::string(buffer);
}

} // namespace


void SymbolObfuscatePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  SymbolTable symbolTable(module);

  // RNG seeded by key
  std::seed_seq seq(key.begin(), key.end());
  std::mt19937 rng(seq);

  llvm::StringMap<std::string> renameMap;

  module.walk([&](func::FuncOp func) {
    StringRef oldName = func.getName();
    std::string newName = generateObfuscatedName(rng);

    renameMap[oldName] = newName;
    symbolTable.setSymbolName(func, newName);
  });

  module.walk([&](Operation *op) {
    SmallVector<NamedAttribute> updatedAttrs;
    bool changed = false;

    for (auto &attr : op->getAttrs()) {
      if (auto symAttr = llvm::dyn_cast<SymbolRefAttr>(attr.getValue())) {
        StringRef old = symAttr.getRootReference();
        if (renameMap.count(old)) {
          auto newRef = SymbolRefAttr::get(ctx, renameMap[old]);
          updatedAttrs.emplace_back(attr.getName(), newRef);
          changed = true;
          continue;
        }
      }
      updatedAttrs.push_back(attr);
    }

    if (changed) {
      op->setAttrs(DictionaryAttr::get(ctx, updatedAttrs));
    }
  });
}

std::unique_ptr<Pass> mlir::obs::createSymbolObfuscatePass(llvm::StringRef key) {
  return std::make_unique<SymbolObfuscatePass>(key.str());
}
