// String test
module {
  func.func @hello() attributes { msg = "HELLO WORLD" } {
    return
  }

  // Symbol test
  func.func @main() {
    // Call with no results => NO %0
    func.call @hello() : () -> ()
    return
  }
}
