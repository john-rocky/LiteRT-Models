// The photo-editing pack: "a bit brighter", "warmer", "crop it square" —
// steering a parameter space with vague words.
//
// Unlike edit_latest_photo (one filter, one save), these tools edit a
// session: the first edit loads the newest photo, every edit stacks on the
// last one, undo steps back, save writes the result to the library. The
// chain is the demo — "a little more" has to mean more on top of what the
// screen already shows.
//
// The tool names are deliberate near-neighbors (crop/resize/zoom,
// brightness/exposure/contrast): whether a model can tell them apart is a
// benchmark axis, not an accident.
import CoreImage
import CoreImage.CIFilterBuiltins
import FoundationModels
import Photos
import UIKit
import Vision

/// The edit chain. One shared instance: the model operates "the photo", and
/// which pixels that means is this class's problem, not the model's.
@available(iOS 27.0, *)
final class PhotoEditBox: @unchecked Sendable {
  static let shared = PhotoEditBox()
  private let lock = NSLock()
  private var current: CIImage?
  private var history: [CIImage] = []
  private let context = CIContext()

  enum Failure: LocalizedError {
    case noPhoto
    var errorDescription: String? { "no photo to edit (or permission was refused)" }
  }

  /// NSLock's lock()/unlock() are banned in async contexts (Swift 6); this
  /// synchronous wrapper is callable from them.
  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  /// The newest library photo on first touch; the working image after that.
  private func workingImage() async throws -> CIImage {
    if let existing = sync({ current }) { return existing }
    guard let cgImage = try await PhotoBox.latestImage() else { throw Failure.noPhoto }
    let loaded = CIImage(cgImage: cgImage)
    sync { current = loaded }
    return loaded
  }

  /// Apply one edit, remember the previous state, put the result on screen.
  func apply(_ caption: String, _ edit: (CIImage) -> CIImage?) async throws -> String {
    let image = try await workingImage()
    guard let edited = edit(image) else { return "could not \(caption)" }
    sync {
      history.append(image)
      current = edited
    }
    if let preview = render(edited) {
      ArtifactBox.shared.post(.photo(preview, caption: caption))
    }
    return "done: \(caption) — the result is on screen"
  }

  func undo() -> String {
    lock.lock()
    defer { lock.unlock() }
    guard let previous = history.popLast() else { return "nothing to undo" }
    current = previous
    if let preview = render(previous) {
      ArtifactBox.shared.post(.photo(preview, caption: "undid the last edit"))
    }
    return "undid the last edit"
  }

  /// Back to the untouched photo. `history.first` is the image before the
  /// first edit — the original, as loaded.
  func reset() -> String {
    lock.lock()
    defer { lock.unlock() }
    guard let original = history.first else { return "no edits to discard" }
    current = original
    history = []
    if let preview = render(original) {
      ArtifactBox.shared.post(.photo(preview, caption: "back to the original"))
    }
    return "discarded all edits — back to the original photo"
  }

  func save() async throws -> String {
    let (image, edits) = sync { (current, history.count) }
    guard let image, let rendered = render(image) else { return "no edits to save yet" }
    try await PHPhotoLibrary.shared().performChanges {
      PHAssetChangeRequest.creationRequestForAsset(from: rendered)
    }
    return "saved a copy with \(edits) edit\(edits == 1 ? "" : "s") to the photo library"
  }

  private func render(_ image: CIImage) -> UIImage? {
    guard let cgImage = context.createCGImage(image, from: image.extent) else { return nil }
    return UIImage(cgImage: cgImage)
  }

  /// The working image as pixels, for `--photocheck`'s eyes-on verification
  /// and the stage's always-on photo.
  func currentRendered() -> UIImage? {
    sync { current }.flatMap { render($0) }
  }

  /// Load the newest photo into the chain without editing it — the stage
  /// shows the photo from the first frame, before any tool has run.
  func preload() async throws {
    _ = try await workingImage()
  }
}

// MARK: - Geometry (crop / resize / zoom are near-neighbors on purpose)

@available(iOS 27.0, *)
struct CropPhotoTool: Tool {
  let name = "crop_photo"
  let description = "Crop the photo to an aspect ratio."

  @Generable struct Arguments {
    @Guide(description: "Target aspect.", .anyOf(["square", "4:3", "3:2", "16:9", "9:16"]))
    var aspect: String
  }

  func call(arguments: Arguments) async throws -> String {
    let ratios: [String: CGFloat] = [
      "square": 1, "4:3": 4.0 / 3.0, "3:2": 3.0 / 2.0, "16:9": 16.0 / 9.0, "9:16": 9.0 / 16.0,
    ]
    guard let ratio = ratios[arguments.aspect.lowercased()] else {
      return "unknown aspect; try \(ratios.keys.sorted().joined(separator: ", "))"
    }
    return try await PhotoEditBox.shared.apply("crop to \(arguments.aspect)") { image in
      let extent = image.extent
      var size = extent.size
      if size.width / size.height > ratio {
        size.width = size.height * ratio
      } else {
        size.height = size.width / ratio
      }
      let origin = CGPoint(
        x: extent.midX - size.width / 2, y: extent.midY - size.height / 2)
      return image.cropped(to: CGRect(origin: origin, size: size))
    }
  }
}

@available(iOS 27.0, *)
struct ResizePhotoTool: Tool {
  let name = "resize_photo"
  let description = "Scale the photo to a percentage of its current size."

  @Generable struct Arguments {
    @Guide(description: "New size as a percentage, 10 to 200. 50 halves it.")
    var percent: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let percent = min(200, max(10, arguments.percent))
    let scale = CGFloat(percent) / 100
    return try await PhotoEditBox.shared.apply("resize to \(percent)%") { image in
      image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }
  }
}

@available(iOS 27.0, *)
struct ZoomPhotoTool: Tool {
  let name = "zoom_photo"
  let description = "Zoom into the center of the photo."

  @Generable struct Arguments {
    @Guide(description: "Zoom factor, 1.2 to 4. 2 shows the middle half.")
    var factor: Double
  }

  func call(arguments: Arguments) async throws -> String {
    let factor = min(4, max(1.2, arguments.factor))
    return try await PhotoEditBox.shared.apply(
      String(format: "zoom %.1fx", factor)
    ) { image in
      let extent = image.extent
      let size = CGSize(width: extent.width / factor, height: extent.height / factor)
      let origin = CGPoint(x: extent.midX - size.width / 2, y: extent.midY - size.height / 2)
      let cropped = image.cropped(to: CGRect(origin: origin, size: size))
      return cropped.transformed(by: CGAffineTransform(scaleX: factor, y: factor))
    }
  }
}

@available(iOS 27.0, *)
struct RotatePhotoTool: Tool {
  let name = "rotate_photo"
  let description = "Rotate the photo."

  @Generable struct Arguments {
    @Guide(description: "Clockwise degrees.", .anyOf(["90", "180", "270"]))
    var degrees: String
  }

  func call(arguments: Arguments) async throws -> String {
    let orientations: [String: CGImagePropertyOrientation] = [
      "90": .right, "180": .down, "270": .left,
    ]
    guard let orientation = orientations[arguments.degrees] else {
      return "rotation must be 90, 180 or 270"
    }
    return try await PhotoEditBox.shared.apply("rotate \(arguments.degrees)°") { image in
      image.oriented(orientation)
    }
  }
}

@available(iOS 27.0, *)
struct FlipPhotoTool: Tool {
  let name = "flip_photo"
  let description = "Mirror the photo."

  @Generable struct Arguments {
    @Guide(description: "Mirror axis.", .anyOf(["horizontal", "vertical"]))
    var direction: String
  }

  func call(arguments: Arguments) async throws -> String {
    switch arguments.direction.lowercased() {
    case "horizontal":
      return try await PhotoEditBox.shared.apply("flip horizontally") { $0.oriented(.upMirrored) }
    case "vertical":
      return try await PhotoEditBox.shared.apply("flip vertically") { $0.oriented(.downMirrored) }
    default:
      return "direction must be horizontal or vertical"
    }
  }
}

// MARK: - Light (brightness / exposure / contrast are near-neighbors on purpose)

@available(iOS 27.0, *)
struct BrightnessPhotoTool: Tool {
  let name = "adjust_photo_brightness"
  let description = "Make the photo brighter or darker."

  @Generable struct Arguments {
    @Guide(description: "-100 (darker) to 100 (brighter).")
    var amount: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let amount = min(100, max(-100, arguments.amount))
    return try await PhotoEditBox.shared.apply("brightness \(amount > 0 ? "+" : "")\(amount)") {
      image in
      let filter = CIFilter.colorControls()
      filter.inputImage = image
      filter.brightness = Float(amount) / 250  // ±0.4 keeps highlights alive
      return filter.outputImage
    }
  }
}

@available(iOS 27.0, *)
struct ExposurePhotoTool: Tool {
  let name = "adjust_photo_exposure"
  let description = "Adjust the photo's exposure in stops."

  @Generable struct Arguments {
    @Guide(description: "Exposure stops, -2 to 2.")
    var stops: Double
  }

  func call(arguments: Arguments) async throws -> String {
    let stops = min(2, max(-2, arguments.stops))
    return try await PhotoEditBox.shared.apply(
      String(format: "exposure %+.1f stops", stops)
    ) { image in
      let filter = CIFilter.exposureAdjust()
      filter.inputImage = image
      filter.ev = Float(stops)
      return filter.outputImage
    }
  }
}

@available(iOS 27.0, *)
struct ContrastPhotoTool: Tool {
  let name = "adjust_photo_contrast"
  let description = "Adjust the photo's contrast."

  @Generable struct Arguments {
    @Guide(description: "-100 (flatter) to 100 (punchier).")
    var amount: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let amount = min(100, max(-100, arguments.amount))
    return try await PhotoEditBox.shared.apply("contrast \(amount > 0 ? "+" : "")\(amount)") {
      image in
      let filter = CIFilter.colorControls()
      filter.inputImage = image
      filter.contrast = 1 + Float(amount) / 250  // 0.6..1.4
      return filter.outputImage
    }
  }
}

// MARK: - Color

@available(iOS 27.0, *)
struct SaturationPhotoTool: Tool {
  let name = "adjust_photo_saturation"
  let description = "Adjust how vivid the photo's colors are."

  @Generable struct Arguments {
    @Guide(description: "-100 (grayscale) to 100 (very vivid).")
    var amount: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let amount = min(100, max(-100, arguments.amount))
    return try await PhotoEditBox.shared.apply("saturation \(amount > 0 ? "+" : "")\(amount)") {
      image in
      let filter = CIFilter.colorControls()
      filter.inputImage = image
      filter.saturation = 1 + Float(amount) / 100
      return filter.outputImage
    }
  }
}

@available(iOS 27.0, *)
struct WarmthPhotoTool: Tool {
  let name = "adjust_photo_warmth"
  let description = "Make the photo warmer (orange) or cooler (blue)."

  @Generable struct Arguments {
    @Guide(description: "-100 (cooler) to 100 (warmer).")
    var amount: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let amount = min(100, max(-100, arguments.amount))
    return try await PhotoEditBox.shared.apply("warmth \(amount > 0 ? "+" : "")\(amount)") {
      image in
      let filter = CIFilter.temperatureAndTint()
      filter.inputImage = image
      filter.neutral = CIVector(x: 6500, y: 0)
      // A LOWER target temperature warms the image (verified by pixel: with
      // +30/unit the "warm" result measured R-B ≈ -21, i.e. blue). Backwards
      // is the natural way to hold this filter; --photocheck is the guard.
      filter.targetNeutral = CIVector(x: 6500 - CGFloat(amount) * 30, y: 0)
      return filter.outputImage
    }
  }
}

@available(iOS 27.0, *)
struct FilterPhotoTool: Tool {
  let name = "apply_photo_filter"
  let description = "Apply a named look to the photo."

  @Generable struct Arguments {
    @Guide(description: "Which look.", .anyOf(["mono", "sepia", "noir", "vivid", "fade"]))
    var look: String
  }

  func call(arguments: Arguments) async throws -> String {
    let names = [
      "mono": "CIPhotoEffectMono", "sepia": "CISepiaTone", "noir": "CIPhotoEffectNoir",
      "vivid": "CIPhotoEffectChrome", "fade": "CIPhotoEffectFade",
    ]
    guard let filterName = names[arguments.look.lowercased()] else {
      return "unknown look; try \(names.keys.sorted().joined(separator: ", "))"
    }
    return try await PhotoEditBox.shared.apply("\(arguments.look) look") { image in
      guard let filter = CIFilter(name: filterName) else { return nil }
      filter.setValue(image, forKey: kCIInputImageKey)
      return filter.outputImage
    }
  }
}

@available(iOS 27.0, *)
struct BlurPhotoTool: Tool {
  let name = "blur_photo"
  let description = "Blur the whole photo."

  @Generable struct Arguments {
    @Guide(description: "Blur radius in pixels, 2 to 50.")
    var radius: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let radius = min(50, max(2, arguments.radius))
    return try await PhotoEditBox.shared.apply("blur radius \(radius)") { image in
      image.clampedToExtent()
        .applyingGaussianBlur(sigma: Double(radius))
        .cropped(to: image.extent)
    }
  }
}

// MARK: - Whole-photo actions

@available(iOS 27.0, *)
struct AutoEnhancePhotoTool: Tool {
  let name = "auto_enhance_photo"
  let description = "Automatically improve the photo."

  func call(arguments: NoArguments) async throws -> String {
    try await PhotoEditBox.shared.apply("auto-enhance") { image in
      var output = image
      for filter in image.autoAdjustmentFilters(options: [.redEye: false]) {
        filter.setValue(output, forKey: kCIInputImageKey)
        if let next = filter.outputImage { output = next }
      }
      return output
    }
  }
}

@available(iOS 27.0, *)
struct UndoPhotoEditTool: Tool {
  let name = "undo_photo_edit"
  let description = "Undo the last photo edit."

  func call(arguments: NoArguments) async throws -> String {
    PhotoEditBox.shared.undo()
  }
}

@available(iOS 27.0, *)
struct CutOutSubjectTool: Tool {
  // "Cut out the person" routed to flip_photo on the 1.2B even with
  // `person` in this tool's name. remove_background is the word the world
  // uses for this job; the beat says exactly that.
  let name = "remove_background"
  let description = "Remove the background, keeping the person or subject."

  func call(arguments: NoArguments) async throws -> String {
    try await PhotoEditBox.shared.apply("remove the background") { image in
      let request = VNGenerateForegroundInstanceMaskRequest()
      let handler = VNImageRequestHandler(ciImage: image)
      try? handler.perform([request])
      guard let observation = request.results?.first,
        let maskBuffer = try? observation.generateScaledMaskForImage(
          forInstances: observation.allInstances, from: handler)
      else { return nil }
      let filter = CIFilter.blendWithMask()
      filter.inputImage = image
      // Transparent background: on the stage's black it reads as the person
      // lifted clean off the photo.
      filter.backgroundImage = CIImage.empty()
      filter.maskImage = CIImage(cvPixelBuffer: maskBuffer)
      return filter.outputImage?.cropped(to: image.extent)
    }
  }
}

@available(iOS 27.0, *)
struct ResetPhotoEditsTool: Tool {
  // Named away from its neighbors the hard way: "undo everything" routed to
  // undo_photo_edit (name match), and as reset_photo_edits the beat routed
  // to resize_photo — `res-` was enough prefix for a 1.2B. No shared stem
  // with any other tool, and the words the user will say are all in it.
  let name = "revert_to_original"
  let description = "Throw away all edits and show the original photo."

  func call(arguments: NoArguments) async throws -> String {
    PhotoEditBox.shared.reset()
  }
}

@available(iOS 27.0, *)
struct SavePhotoTool: Tool {
  let name = "save_edited_photo"
  let description = "Save the edited photo to the library."

  func call(arguments: NoArguments) async throws -> String {
    try await PhotoEditBox.shared.save()
  }
}
