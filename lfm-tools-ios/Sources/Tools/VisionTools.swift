// The photo tools, the way Foundation Models does vision natively.
//
// The picture goes into the prompt as an `Attachment` with a label; the model
// looks at the pixels and decides; and when it calls a tool it names the
// picture it means with an `ImageReference` argument, which the tool resolves
// against the session transcript. Nothing here tells the model what is wrong
// with the photo — that is the point. Same names and descriptions as the
// plain photo pack, so the stage reads the same; only the `image` argument is
// new, and the bodies delegate to the plain tools once the referenced image
// is "the photo".
import CoreGraphics
import CoreImage
import Foundation
import FoundationModels
import Vision

/// Where an ImageReference gets resolved: the transcript of whichever session
/// is live. Set when a session is created; tools read it during a call, when
/// the model is waiting on them anyway.
@available(iOS 27.0, *)
final class TranscriptBox: @unchecked Sendable {
  static let shared = TranscriptBox()
  private let lock = NSLock()
  private var provider: (@Sendable () -> Transcript?)?

  func attach(_ session: LanguageModelSession) {
    lock.lock()
    provider = { [weak session] in session?.transcript }
    lock.unlock()
  }

  func current() -> Transcript? {
    lock.lock()
    let provider = self.provider
    lock.unlock()
    return provider?()
  }
}

@available(iOS 27.0, *)
enum SeenPhoto {
  /// The label every attachment carries when there is only one photo in
  /// play — the model refers back to it by this name. "image", not "photo":
  /// Apple's model wrote `image` as the label whatever the attachment was
  /// called (2026-08-19), and its documentation labels examples image-0,
  /// image-1 — so the label the model reaches for is the label we use.
  static let singleLabel = "image"

  /// Make the referenced picture "the photo" the editing chain works on.
  /// A reference to the label already on stage is left alone: the same
  /// photo, attached again after an edit so the model can see its own work,
  /// must not reset the chain it is looking at. A label that resolves to
  /// nothing falls back to the newest image in the transcript — with one
  /// photo in play there is only one thing the model can mean, and a demo
  /// should not die on the spelling of a label. Returns the label of what
  /// could not be resolved at all, or nil on success.
  static func select(_ reference: ImageReference) -> String? {
    if PhotoEditBox.shared.loadedLabel == reference.attachmentLabel { return nil }
    guard let transcript = TranscriptBox.shared.current() else { return reference.attachmentLabel }
    if let attachment = reference.resolved(in: transcript) {
      PhotoEditBox.shared.load(attachment.cgImage, label: reference.attachmentLabel)
      return nil
    }
    if let (label, attachment) = newestAttachment(in: transcript) {
      // Already on stage under its real label: leave the chain alone.
      if PhotoEditBox.shared.loadedLabel == label { return nil }
      PhotoEditBox.shared.load(attachment.cgImage, label: label)
      return nil
    }
    return reference.attachmentLabel
  }

  /// The last image attached to any prompt in the transcript, with its label.
  private static func newestAttachment(in transcript: Transcript) -> (String, Transcript.ImageAttachment)? {
    for entry in transcript.reversed() {
      guard case .prompt(let prompt) = entry else { continue }
      for segment in prompt.segments.reversed() {
        guard case .attachment(let attachment) = segment,
          case .image(let image) = attachment.content
        else { continue }
        return (attachment.label ?? singleLabel, image)
      }
    }
    return nil
  }

  static func unresolved(_ label: String) -> String {
    "no image called \"\(label)\" in this conversation"
  }

  /// Vague amounts land on the rail: asked for 0–100, the model answered 100
  /// for "warmer", "more contrast" and "make it look its best" alike. The
  /// vision tools take the steps the words already have and map them to
  /// numbers a photo can survive.
  static func percent(_ strength: String) -> Int {
    switch strength.lowercased() {
    case "a_lot": return 60
    case "some": return 35
    default: return 15
    }
  }

  static func stops(_ strength: String) -> Double {
    switch strength.lowercased() {
    case "a_lot": return 1.2
    case "some": return 0.7
    default: return 0.3
    }
  }

  static let strengths = ["a_little", "some", "a_lot"]
}

@available(iOS 27.0, *)
struct SeenBrightnessTool: Tool {
  let name = "adjust_photo_brightness"
  let description = "Make the photo brighter or darker."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "Which way.", .anyOf(["brighter", "darker"])) var direction: String
    @Guide(description: "How much.", .anyOf(SeenPhoto.strengths)) var strength: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    let sign = arguments.direction.lowercased() == "darker" ? -1 : 1
    return try await BrightnessPhotoTool().call(
      arguments: .init(amount: sign * SeenPhoto.percent(arguments.strength)))
  }
}

@available(iOS 27.0, *)
struct SeenExposureTool: Tool {
  let name = "adjust_photo_exposure"
  let description = "Adjust the photo's exposure in stops."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "Which way.", .anyOf(["up", "down"])) var direction: String
    @Guide(description: "How much.", .anyOf(SeenPhoto.strengths)) var strength: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    let sign: Double = arguments.direction.lowercased() == "down" ? -1 : 1
    return try await ExposurePhotoTool().call(
      arguments: .init(stops: sign * SeenPhoto.stops(arguments.strength)))
  }
}

@available(iOS 27.0, *)
struct SeenContrastTool: Tool {
  let name = "adjust_photo_contrast"
  let description = "Adjust the photo's contrast."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "Which way.", .anyOf(["more", "less"])) var direction: String
    @Guide(description: "How much.", .anyOf(SeenPhoto.strengths)) var strength: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    let sign = arguments.direction.lowercased() == "less" ? -1 : 1
    return try await ContrastPhotoTool().call(
      arguments: .init(amount: sign * SeenPhoto.percent(arguments.strength)))
  }
}

@available(iOS 27.0, *)
struct SeenSaturationTool: Tool {
  let name = "adjust_photo_saturation"
  let description = "Adjust how vivid the photo's colors are."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "Which way.", .anyOf(["more_vivid", "more_muted"])) var direction: String
    @Guide(description: "How much.", .anyOf(SeenPhoto.strengths)) var strength: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    let sign = arguments.direction.lowercased() == "more_muted" ? -1 : 1
    return try await SaturationPhotoTool().call(
      arguments: .init(amount: sign * SeenPhoto.percent(arguments.strength)))
  }
}

@available(iOS 27.0, *)
struct SeenWarmthTool: Tool {
  let name = "adjust_photo_warmth"
  let description = "Make the photo warmer (orange) or cooler (blue)."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "Which way.", .anyOf(["warmer", "cooler"])) var direction: String
    @Guide(description: "How much.", .anyOf(SeenPhoto.strengths)) var strength: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    let sign = arguments.direction.lowercased() == "cooler" ? -1 : 1
    return try await WarmthPhotoTool().call(
      arguments: .init(amount: sign * SeenPhoto.percent(arguments.strength)))
  }
}

@available(iOS 27.0, *)
struct SeenRotateTool: Tool {
  let name = "rotate_photo"
  let description = "Rotate the photo."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "Clockwise degrees.", .anyOf(["90", "180", "270"])) var degrees: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    return try await RotatePhotoTool().call(arguments: .init(degrees: arguments.degrees))
  }
}

@available(iOS 27.0, *)
struct SeenCropTool: Tool {
  let name = "crop_photo"
  let description = "Crop the photo to an aspect ratio."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "Target aspect.", .anyOf(["square", "4:3", "3:2", "16:9", "9:16"]))
    var aspect: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    return try await CropPhotoTool().call(arguments: .init(aspect: arguments.aspect))
  }
}

@available(iOS 27.0, *)
struct SeenFilterTool: Tool {
  let name = "apply_photo_filter"
  let description = "Apply a named look to the photo."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "Which look.", .anyOf(["mono", "sepia", "noir", "vivid", "fade"]))
    var look: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    return try await FilterPhotoTool().call(arguments: .init(look: arguments.look))
  }
}

@available(iOS 27.0, *)
struct SeenAutoEnhanceTool: Tool {
  let name = "auto_enhance_photo"
  let description = "Automatically improve the photo."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    return try await AutoEnhancePhotoTool().call(arguments: NoArguments())
  }
}

@available(iOS 27.0, *)
struct SeenRemoveBackgroundTool: Tool {
  let name = "remove_background"
  let description = "Remove the background, keeping the person or subject."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    return try await CutOutSubjectTool().call(arguments: NoArguments())
  }
}

@available(iOS 27.0, *)
struct SeenReadTextTool: Tool {
  let name = "read_text_in_photo"
  let description = "Read the text in the photo, exactly as written."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    return try await ReadPhotoTextTool().call(arguments: NoArguments())
  }
}

@available(iOS 27.0, *)
struct SeenRedactTool: Tool {
  let name = "redact_photo"
  let description = "Black out faces, readable text, or both — so the photo can be shared without the private parts."
  @Generable struct Arguments {
    @Guide(description: "The attached photo, by its label.") var image: ImageReference
    @Guide(description: "What to hide.", .anyOf(["faces", "text", "both"])) var hide: String
  }
  func call(arguments: Arguments) async throws -> String {
    if let missing = SeenPhoto.select(arguments.image) { return SeenPhoto.unresolved(missing) }
    guard let cgImage = PhotoEditBox.shared.currentCGImage() else {
      return "no photo to redact"
    }
    // Detect first, so "nothing found" is an honest answer, not a fake edit.
    let hide = arguments.hide.lowercased()
    let handler = VNImageRequestHandler(cgImage: cgImage)
    var normalized: [CGRect] = []
    var found: [String] = []
    if hide != "text" {
      let faces = VNDetectFaceRectanglesRequest()
      try? handler.perform([faces])
      let boxes = (faces.results ?? []).map(\.boundingBox)
      if !boxes.isEmpty { found.append("\(boxes.count) face\(boxes.count == 1 ? "" : "s")") }
      normalized += boxes
    }
    if hide != "faces" {
      let texts = VNRecognizeTextRequest()
      texts.recognitionLevel = .fast
      try? handler.perform([texts])
      let boxes = (texts.results ?? []).map(\.boundingBox)
      if !boxes.isEmpty { found.append("\(boxes.count) piece\(boxes.count == 1 ? "" : "s") of text") }
      normalized += boxes
    }
    guard !normalized.isEmpty else {
      return "nothing to hide — no \(hide == "both" ? "faces or text" : hide) in the photo"
    }
    let caption = "redact \(found.joined(separator: " and "))"
    return try await PhotoEditBox.shared.apply(caption) { image in
      let width = image.extent.width
      let height = image.extent.height
      var out = image
      for box in normalized {
        let rect = CGRect(
          x: image.extent.minX + box.minX * width, y: image.extent.minY + box.minY * height,
          width: box.width * width, height: box.height * height
        ).insetBy(dx: -width * 0.005, dy: -height * 0.005)
        out = CIImage(color: .black).cropped(to: rect).composited(over: out)
      }
      return out.cropped(to: image.extent)
    }
  }
}

@available(iOS 27.0, *)
struct SeenRevertTool: Tool {
  let name = "revert_to_original"
  let description = "Throw away all edits and show the original photo."
  func call(arguments: NoArguments) async throws -> String {
    PhotoEditBox.shared.reset()
  }
}

@available(iOS 27.0, *)
struct SeenSaveTool: Tool {
  let name = "save_edited_photo"
  let description = "Save the edited photo to the library."
  func call(arguments: NoArguments) async throws -> String {
    try await PhotoEditBox.shared.save()
  }
}
