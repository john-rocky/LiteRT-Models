// See and choose, split: for the vision models too small to route.
//
// A 450M vision bundle sees ("Mountains") and never calls a tool — every beat
// comes back as words (measured 2026-08-19). Tool calling asks one model to
// do two things at once: look, and pick from a menu of schemas. This file
// takes the second job away from it. The model is asked one question about
// the photo and answers with a `@Generable` enum — the constrained-decoding
// path every backend here has — and the app maps that answer to the tool
// call and makes it. The model looks; the app chooses what to call.
//
// The same split runs on Apple's model too, which is how the stage can show
// both: `--scenario choose --backend apple`, or with `--model VL-450M`.
import CoreGraphics
import Foundation
import FoundationModels

/// The one adjustment a photo needs most, as a menu the model picks from.
@available(iOS 27.0, *)
@Generable
enum PhotoNeed {
  case brighter, darker, warmer, cooler, more_contrast, straighten, nothing
  var word: String { String(describing: self).replacingOccurrences(of: "_", with: " ") }
}

@available(iOS 27.0, *)
@Generable
struct NeedVerdict {
  @Guide(description: "The one adjustment this photo needs most, or nothing if it looks fine.")
  var need: PhotoNeed
}

@available(iOS 27.0, *)
@Generable
struct YesNoVerdict {
  @Guide(description: "true if yes, false if no.")
  var yes: Bool
  @Guide(description: "Why, in a few words.")
  var because: String
}

/// The app's side of the split: each question the stage asks, what the model
/// answers with, and which tool the app calls for each answer.
@available(iOS 27.0, *)
enum SeeAndChoose {
  static let instructions = """
    You can see the photo attached to the message. Look at it and answer the
    question about it. Answer only what is asked.
    """

  /// A beat: the question, and how the answer becomes a call.
  struct Step: Sendable {
    let question: String
    let kind: Kind
  }
  enum Kind: Sendable {
    case need  // NeedVerdict → one adjustment tool
    case person  // YesNoVerdict → remove_background if yes
    case text  // YesNoVerdict → read_text_in_photo → write_note if yes
    case save  // no question: the app saves
  }

  static let steps: [Step] = [
    Step(question: "What does this photo need most?", kind: .need),
    Step(question: "Is there a person in this photo?", kind: .person),
    Step(question: "Is there readable text in this photo?", kind: .text),
    Step(question: "", kind: .save),
  ]

  /// Ask, then act. Returns what happened for the stage: the model's answer
  /// as text, the tool the app called (if any) with its arguments and
  /// result, and the sentence to show.
  struct Outcome: Sendable {
    let answer: String
    let call: (name: String, arguments: String, returned: String)?
    let sentence: String
  }

  static func run(_ step: Step, session: LanguageModelSession, photo: CGImage) async -> Outcome {
    switch step.kind {
    case .need:
      do {
        let verdict = try await session.respond(
          to: Prompt {
            step.question
            Attachment(photo).label(SeenPhoto.singleLabel)
          }, generating: NeedVerdict.self
        ).content
        let need = verdict.need
        guard need != .nothing else {
          return Outcome(answer: "nothing", call: nil, sentence: "It looks fine as it is — nothing to change.")
        }
        let (name, arguments, returned) = try await apply(need)
        return Outcome(
          answer: need.word, call: (name, arguments, returned),
          sentence: "It needed to be \(need.word) — done.")
      } catch {
        return Outcome(answer: "error: \(error.localizedDescription)", call: nil, sentence: error.localizedDescription)
      }
    case .person:
      do {
        let verdict = try await session.respond(
          to: Prompt {
            step.question
            Attachment(photo).label(SeenPhoto.singleLabel)
          }, generating: YesNoVerdict.self
        ).content
        guard verdict.yes else {
          return Outcome(answer: "no — \(verdict.because)", call: nil, sentence: "No person in it (\(verdict.because)); left as is.")
        }
        let returned = try await CutOutSubjectTool().call(arguments: NoArguments())
        return Outcome(
          answer: "yes — \(verdict.because)", call: ("remove_background", "{}", returned),
          sentence: "There is a person (\(verdict.because)) — cut out from the background.")
      } catch {
        return Outcome(answer: "error: \(error.localizedDescription)", call: nil, sentence: error.localizedDescription)
      }
    case .text:
      do {
        let verdict = try await session.respond(
          to: Prompt {
            step.question
            Attachment(photo).label(SeenPhoto.singleLabel)
          }, generating: YesNoVerdict.self
        ).content
        guard verdict.yes else {
          return Outcome(answer: "no — \(verdict.because)", call: nil, sentence: "No text in it (\(verdict.because)).")
        }
        let read = try await ReadPhotoTextTool().call(arguments: NoArguments())
        var returned = read
        if !read.hasPrefix("no text") {
          returned += "\n" + (try await WriteNoteTool().call(arguments: .init(text: read)))
        }
        return Outcome(
          answer: "yes — \(verdict.because)", call: ("read_text_in_photo → write_note", "{}", returned),
          sentence: "There is text — read it and kept it as a note.")
      } catch {
        return Outcome(answer: "error: \(error.localizedDescription)", call: nil, sentence: error.localizedDescription)
      }
    case .save:
      do {
        let returned = try await PhotoEditBox.shared.save()
        return Outcome(answer: "", call: ("save_edited_photo", "{}", returned), sentence: returned)
      } catch {
        return Outcome(answer: "", call: nil, sentence: error.localizedDescription)
      }
    }
  }

  /// The menu answer becomes the call: the same tool bodies the vision pack
  /// uses, with the strength the polish pack settled on ("some").
  private static func apply(_ need: PhotoNeed) async throws -> (String, String, String) {
    switch need {
    case .brighter:
      return ("adjust_photo_brightness", "{brighter, some}",
        try await BrightnessPhotoTool().call(arguments: .init(amount: SeenPhoto.percent("some"))))
    case .darker:
      return ("adjust_photo_brightness", "{darker, some}",
        try await BrightnessPhotoTool().call(arguments: .init(amount: -SeenPhoto.percent("some"))))
    case .warmer:
      return ("adjust_photo_warmth", "{warmer, some}",
        try await WarmthPhotoTool().call(arguments: .init(amount: SeenPhoto.percent("some"))))
    case .cooler:
      return ("adjust_photo_warmth", "{cooler, some}",
        try await WarmthPhotoTool().call(arguments: .init(amount: -SeenPhoto.percent("some"))))
    case .more_contrast:
      return ("adjust_photo_contrast", "{more, some}",
        try await ContrastPhotoTool().call(arguments: .init(amount: SeenPhoto.percent("some"))))
    case .straighten:
      return ("auto_enhance_photo", "{}", try await AutoEnhancePhotoTool().call(arguments: NoArguments()))
    case .nothing:
      return ("none", "{}", "nothing to do")
    }
  }
}
