module {
  // String test
  func.func @hello() attributes { msg = "HELLO WORLD" } {
    return
  }

  // Symbol test
  func.func @main() {
    // call with no results
    func.call @hello() : () -> ()
    return
  }
}
