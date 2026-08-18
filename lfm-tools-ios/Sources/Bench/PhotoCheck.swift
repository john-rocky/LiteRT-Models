// `--photocheck` verifies the photo tools' actual pixels, no model involved:
// apply each edit to the newest library photo, write a PNG per step into
// Documents, pull them back and look. The warmth direction in particular is
// only trustworthy after eyes have been on it (CITemperatureAndTint's two
// neutrals are easy to hold backwards).
import Foundation
import UIKit

@available(iOS 27.0, *)
@MainActor
enum PhotoCheck {
  static func run() async {
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    // Per-run names: `devicectl copy from` serves stale content for a path
    // it has copied before, and these files exist to be looked at.
    let runID = Int(Date().timeIntervalSince1970)
    func snap(_ name: String) {
      guard let image = PhotoEditBox.shared.currentRendered(),
        let data = image.pngData()
      else {
        print("PHOTOCHECK no image at \(name)")
        return
      }
      try? data.write(to: documents.appendingPathComponent("photocheck-\(runID)-\(name).png"))
      print("PHOTOCHECK wrote \(name)")
    }

    let box = PhotoEditBox.shared
    _ = try? await BrightnessPhotoTool().call(arguments: .init(amount: 40))
    snap("1-bright+40")
    _ = box.undo()
    _ = try? await WarmthPhotoTool().call(arguments: .init(amount: 80))
    snap("2-warm+80")
    _ = box.undo()
    _ = try? await WarmthPhotoTool().call(arguments: .init(amount: -80))
    snap("3-cool-80")
    _ = box.undo()
    _ = try? await CropPhotoTool().call(arguments: .init(aspect: "square"))
    snap("4-square")
    _ = try? await FilterPhotoTool().call(arguments: .init(look: "sepia"))
    snap("5-sepia-on-square")
    _ = box.undo()
    _ = box.undo()
    snap("6-original-again")
    try? Data("done\n".utf8).write(
      to: documents.appendingPathComponent("photocheck-\(runID).done"))
    print("PHOTOCHECK done")
  }
}
