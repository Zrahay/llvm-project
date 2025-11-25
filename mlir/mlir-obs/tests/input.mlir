module {
  // String test
  func.func @hello() attributes { msg = "HELLO WORLD" } {
    return
  }

  // Symbol test
  func.func @main() {
    func.call @hello() : () -> ()
    return
  }
}
