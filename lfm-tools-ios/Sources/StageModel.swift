// Drives the stage: one beat at a time, each with a visible phase.
//
// Unlike the chat model this keeps no history on screen. What matters is the
// moment in front of you; the transcript still exists inside the session.
import Foundation
import FoundationModels
#if canImport(LiteRTLM)
  import LiteRTLM
#endif
#if canImport(LiteRTLMFoundationModels)
  import LiteRTLMFoundationModels
#endif
import Observation
import UIKit

@available(iOS 27.0, *)
@MainActor
@Observable
final class StageModel {
  enum Phase {
    case idle
    /// The instruction being typed into the composer, before it is sent.
    case typing
    case thinking
    case calling(name: String, arguments: String, returned: String?)
    case result(text: String, artifact: Artifact?)
  }

  /// Six beats, chosen for what a viewer can see happening rather than for
  /// coverage. Steps, tilt, arithmetic and calendar all work; none of them is a
  /// surprise, and a demo that shows everything shows nothing.
  // Four beats, not six: on the 1.2B the translate and speak beats fail the
  // same way across five prompt/description variants — the model translates
  // by itself and apologizes that it "can't speak out loud" — so the recorded
  // cut keeps only what the model actually does. The 2.6B runs all six.
  static let beats = [
    "Where am I?",
    "Find a coffee shop near me.",
    "Read the text in my latest photo.",
    // Last on purpose: this one sends the app to the background, and iOS stops
    // generating there — nothing after it would run.
    // Named, not referred to: the shop was found turns ago and the history
    // window does not reach that far, so "that coffee shop" sent the model
    // searching again instead of opening the map.
    "Open CAFE LA in Apple Maps.",
  ]

  /// The photo-editing cut: edits stacking on edits, a mistake talked back
  /// out of existence, and a save. Undo by voice is the surprise beat.
  // Runs against ToolBox.photoStage (no one-step undo): with undo present,
  // every "undo everything / revert / reset" wording routed to it on the
  // 1.2B, and "Reset it" routed to resize_photo (res- prefix). The mistake
  // beat now reverts the whole chain, visibly back to the untouched photo.
  static let photoBeats = [
    "Make the photo a bit brighter.",
    "A bit warmer, too.",
    "Crop it square.",
    "Undo everything — back to the original.",
    "Give it a sepia look.",
    "Remove the background.",
    "Save it.",
  ]

  /// The focus cut: one sentence at a time steering notifications, a timer
  /// and the screen itself. The dim beat is the visible one; the last beat is
  /// the compound — two tools out of one sentence, written in call order
  /// because the models that chain follow the sentence.
  static let focusBeats = [
    "Set a timer for 25 minutes.",
    "Remind me to stretch in half an hour.",
    "What notifications are coming up?",
    "Remember this: I stopped at page 128.",
    "Dim the screen — I need to focus.",
    "Silence all my notifications and set a one-hour focus timer.",
  ]

  /// The field-report cut: the gauge photo read out, kept, and turned into
  /// tomorrow's obligation. "Save that as a note" is the anaphora beat — the
  /// 1.2B is expected to fail it (references don't become arguments at that
  /// size); the recorded cut decides whether that beat stays.
  static let reportBeats = [
    "Read the text in my latest photo.",
    "Save that as a note.",
    "Remind me tomorrow at 9 to file the report.",
    "What's still on my reminder list?",
  ]

  /// Three quick packs for trying the phone's other capabilities on Apple's
  /// model: what it knows about right now, what it can sense, what it can
  /// leave behind. Beats are what a viewer sees happen, not coverage.
  static let briefingBeats = [
    "What time is it, and how's my battery?",
    "What's on my calendar this week?",
    "Anything left on my reminder list?",
    "How many steps have I walked today?",
    "Chart my steps for the last 7 days.",
  ]
  static let sensorBeats = [
    "Where am I?",
    "What's this place called?",
    "Which way am I facing?",
    "Am I moving right now?",
    "How loud is it here?",
    "How high up am I?",
  ]
  static let handoffBeats = [
    "Turn the flashlight on.",
    "Turn it off and play the mail sound.",
    "Put a 3 on the app icon.",
    "Copy 'meeting moved to 4pm' to the clipboard.",
    "Remind me in 30 seconds to check the oven.",
    "Note this: the gate code is 2281.",
  ]

  /// One sentence, several calls. Written so the second call depends on
  /// nothing the first returned (torch + battery) or on exactly what it
  /// returned (cafe → Maps, OCR → translate): both kinds of chain.
  static let chainBeats = [
    "Turn the flashlight on and tell me how much battery I have.",
    "Where am I, and how loud is it here?",
    "What time is it in Tokyo, and in London?",
    "Read the text in my latest photo and translate it into Japanese.",
    "Find a cafe near me and open the first one in Maps.",
  ]

  /// One sentence, one call, several things happen — the compound tools.
  static let compoundBeats = [
    "I need to focus for 20 minutes.",
    "Give me my morning briefing.",
    "Copy where I am so I can send it to someone.",
    "Read my latest photo and keep the text as a note.",
  ]

  /// The vision cut: the photo goes in with every beat, and what the model
  /// does depends on what it sees. No beat names a tool or an adjustment —
  /// "fix it" on a dark photo should become brightness, on a tilted one
  /// rotate, on a menu OCR. The conditional beats ("if there's text…") are
  /// routing decided by pixels: the right answer may be no call at all.
  // Conditionals before the makeover: on the first run "what would you fix"
  // removed the background of a portrait, and the later "if there's a person,
  // remove the background" had nothing left to do and rotated the photo
  // instead. The model is shown its own work, so beat order is beat logic.
  static let visionBeats = [
    "What's in this photo?",
    "If there's any text in it, save that text as a note.",
    "If there's a person in it, cut them out from the background.",
    "Is there anything you'd fix about it? If so, do it.",
    "Now make it look its best, and save it.",
  ]

  /// The demo: hand the phone a photo and say nothing. An empty beat sends
  /// the attachment alone; the vision instructions say what that means —
  /// make it look its best. Then three words each, spoken if `--voice`.
  static let polishBeats = [
    "",
    "A little more.",
    "Now cut her out from the background.",
    "Save it.",
  ]

  /// Sight alone, no tools: does the model see the picture at all? Three
  /// questions a blind model cannot answer from the words. Run this before
  /// reading anything into what the vision pack routes.
  static let lookBeats = [
    "Describe this photo in one sentence.",
    "Is there a person in this photo? Answer yes or no, then say why.",
    "Is this photo too dark, too bright, or about right?",
    "Is there any readable text in it? If so, what does it say?",
  ]

  /// The video cut: a CapCut's menu, said out loud, on the newest library
  /// video. The model never sees a frame — every beat carries the app's
  /// state (clips, selection, playhead, frame size) at the top of the
  /// message, and the tools take the numbers it names. Beat 1 is the
  /// headline: three menu items out of one sentence. Beat 2 is the state
  /// test — "the playhead" is a number in the message, and the model has to
  /// copy it. Beat 3 is a two-call chain (select → speed). Export last: it
  /// writes to the library and takes the longest.
  static let videoBeats = [
    "Cut the first two seconds, make it vertical, and fade out at the end.",
    "Split it at the playhead.",
    "Make the second clip slow motion.",
    "Caption it 'Tokyo, August' at the bottom for the first three seconds.",
    "Mute it, and put some calm music under it.",
    "Export it.",
  ]

  /// The store cut: a Shopify admin's menu over canned products and orders.
  /// Filter → act is the shape: "their" in beat 2 is the selection beat 1
  /// made, resolved by the app. Beat 6 is a two-call chain (filter →
  /// fulfil); beat 7 is a report, no records change.
  static let storeBeats = [
    "Which products have fewer than 5 in stock?",
    "Cut their prices by 10%.",
    "Tag them 'clearance'.",
    "Show me the orders that haven't been paid.",
    "Send them a payment reminder.",
    "Fulfil all the paid orders that haven't shipped yet.",
    "How were sales this week?",
  ]

  /// The audio cut: a GarageBand mixer, four synthesized tracks looping.
  /// Play first so every change is heard; "a bit quieter" is a number the
  /// model reads from the state and lowers; beat 5 is a two-call chain.
  static let audioBeats = [
    "Play it from the top.",
    "The keys are a bit loud — turn them down a bit.",
    "Put some echo on the lead.",
    "Solo the drums.",
    "Un-solo the drums, and pan the bass a little left.",
    "Fade out at the end, then export it.",
  ]

  /// The documents cut: an Acrobat / Goodnotes menu on a real PDF. Pages
  /// are named in the state by their first line, so "the cover" and "the
  /// rules page" are numbers the model reads; beat 2 chains on beat 1's
  /// answer; beat 6 needs the page count.
  static let docsBeats = [
    "Which pages mention the deposit?",
    "Go to the first of those and highlight every 'deposit' in yellow.",
    "Add a note: check this with the landlord.",
    "Delete the cover page.",
    "Move the house rules page to the end.",
    "Sign the last page.",
    "Save it as 'lease-signed'.",
  ]

  /// The shopping cut: the buyer's side. Beat 3 is the state test ("the
  /// second one" is a number in the message); beat 4 acts on the cart by
  /// name; checkout closes it.
  static let shoppingBeats = [
    "Find wireless earbuds.",
    "Sort them by price, cheapest first.",
    "Add the second one to the cart — two of them.",
    "Actually, make it just one.",
    "Use the coupon SAVE10.",
    "Check out.",
  ]

  /// The money cut: report → triage → fix → the two reports. Beat 3 is the
  /// anaphora beat: "they" is the selection beat 2 made.
  static let moneyBeats = [
    "What did I spend this week?",
    "Show me the uncategorized ones.",
    "They're all groceries — categorize them.",
    "Find my subscriptions.",
    "Set the eating-out budget to 25,000 yen.",
    "How am I doing against my budgets this month?",
  ]

  /// The inbox cut: triage. Beat 2 and beat 5 are find-then-act chains; the
  /// reply beat writes a draft and sends nothing.
  static let inboxBeats = [
    "What's new in my inbox?",
    "Archive all the newsletters.",
    "Snooze the invoice from Kanda Goods until tomorrow.",
    "Flag the one from the property company.",
    "Reply to Hana: I'll send the report by Friday.",
    "Archive everything I've already read.",
  ]

  /// See and choose, split (Tools/ChooseTools.swift): the model answers a
  /// question about the photo with a @Generable enum, the app makes the
  /// call. For the vision bundles too small to route; runs on Apple's model
  /// too, so the two can be compared on the same beats.
  static let chooseBeats = SeeAndChoose.steps.map { $0.question.isEmpty ? "(the app saves the result)" : $0.question }

  /// `--scenario photo|focus|report|briefing|sensors|handoff|chains|compound|vision|look|polish|video|store|audio|docs|choose`
  /// swaps the stage to that pack; default stays the coffee run. Beats and
  /// tools travel together, same as the bench.
  static var scenarioBeats: [String] {
    switch scenarioName {
    case "photo": return photoBeats
    case "focus": return focusBeats
    case "report": return reportBeats
    case "briefing": return briefingBeats
    case "sensors": return sensorBeats
    case "handoff": return handoffBeats
    case "chains": return chainBeats
    case "compound": return compoundBeats
    case "vision": return visionBeats
    case "look": return lookBeats
    case "polish": return polishBeats
    case "video": return videoBeats
    case "store": return storeBeats
    case "audio": return audioBeats
    case "docs": return docsBeats
    case "shopping": return shoppingBeats
    case "money": return moneyBeats
    case "inbox": return inboxBeats
    case "choose": return chooseBeats
    default: return beats
    }
  }

  /// Packs that keep a photo on stage from the first frame.
  static var scenarioShowsPhoto: Bool {
    scenarioIsPhoto || scenarioAttachesPhoto || scenarioIsChoose
  }
  static var scenarioIsChoose: Bool { scenarioName == "choose" }
  /// Packs where the photo goes into the prompt as an image attachment, so
  /// the model routes on what it sees rather than on what it is told.
  static var scenarioAttachesPhoto: Bool {
    ["vision", "look", "polish"].contains(scenarioName)
  }
  /// Packs where the app's state — not a picture — is the input: the model
  /// is told the timeline (video) or the store and its selection (store)
  /// and asked to operate it.
  static var scenarioSendsState: Bool {
    ["video", "store", "audio", "docs", "shopping", "money", "inbox"].contains(scenarioName)
  }
  static var scenarioIsVideo: Bool { scenarioName == "video" }
  static var scenarioIsDocs: Bool { scenarioName == "docs" }

  /// The state block for this beat, from whichever box owns the pack.
  static func currentState() -> String {
    switch scenarioName {
    case "store": return StoreBox.shared.describe()
    case "audio": return AudioBox.shared.describe()
    case "docs": return DocBox.shared.describe()
    case "shopping": return ShopBox.shared.describe()
    case "money": return MoneyBox.shared.describe()
    case "inbox": return InboxBox.shared.describe()
    default: return VideoEditBox.shared.describe()
    }
  }
  static var stateInstructions: String {
    switch scenarioName {
    case "store": return ToolBox.storeInstructions
    case "audio": return ToolBox.audioInstructions
    case "docs": return ToolBox.docsInstructions
    case "shopping": return ToolBox.shoppingInstructions
    case "money": return ToolBox.moneyInstructions
    case "inbox": return ToolBox.inboxInstructions
    default: return ToolBox.videoInstructions
    }
  }

  static var scenarioName: String {
    guard let flag = CommandLine.arguments.firstIndex(of: "--scenario"),
      CommandLine.arguments.indices.contains(flag + 1)
    else { return "coffee" }
    return CommandLine.arguments[flag + 1].lowercased()
  }

  static var scenarioIsPhoto: Bool { scenarioName == "photo" }

  /// `--voice`: the beats come from the microphone instead of the script.
  /// The scenario still sets the tool set and the timeline length; what is
  /// said is up to the person holding the phone. Speech is the vaguest
  /// interface there is, and the whole point of the recording.
  static var voiceDriven: Bool { CommandLine.arguments.contains("--voice") }
  private(set) var listening = false
  private let voice = VoiceInput()



  private(set) var question = ""
  /// Who the bubble belongs to. "YOU" for a person's beat; the see-and-choose
  /// scenario is the app asking the model, and the label says so.
  private(set) var speaker = "YOU"
  /// What is in the composer right now, while the instruction is being typed.
  private(set) var typed = ""
  private(set) var live = ""
  private(set) var phase: Phase = .idle
  private(set) var rate: Double = 0
  private(set) var beatIndex = 0
  private(set) var toolCount = 0
  /// What the HUD claims. The demo runs on whichever backend is selected below;
  /// the tools, the cards and the stage are identical either way.
  private(set) var backendName = ""

  enum Backend { case system, liteRT }
  /// `--backend apple` runs the same stage on Apple's model — the way to try
  /// a new pack in a minute instead of a quarter of an hour. Default stays
  /// LiteRT, which is what every recording so far was made on.
  static var backend: Backend {
    #if !canImport(LiteRTLM)
      // No LiteRT engine in the Mac Catalyst build: the stage always runs on
      // Apple's model there.
      return .system
    #else
      guard let flag = CommandLine.arguments.firstIndex(of: "--backend"),
        CommandLine.arguments.indices.contains(flag + 1)
      else { return .liteRT }
      return CommandLine.arguments[flag + 1].lowercased() == "apple" ? .system : .liteRT
    #endif
  }
  /// Bumped on every phase change so the view animates the swap rather than
  /// diffing an enum with associated values.
  private(set) var phaseID = 0

  /// The photo scenario's star: the edit chain's current image, on screen
  /// from the first frame to the last, updated after every edit.
  private(set) var stageImage: UIImage?
  private(set) var stageImageID = 0

  private func refreshStageImage() {
    guard Self.scenarioShowsPhoto else { return }
    stageImage = PhotoEditBox.shared.currentRendered()
    stageImageID += 1
  }

  /// The video pack's stage: the frame at the moment the last edit touched,
  /// rendered through the composition, and the timeline as blocks under it.
  /// Async because the frame is decoded on demand — a few hundred ms, off
  /// the actor.
  private(set) var stageTimeline: VideoEditBox.Snapshot?
  /// The documents pack's: the open page above, the pages as a strip below.
  private(set) var stagePages: DocBox.Snapshot?
  /// The audio pack's: the mixer, faders and all.
  private(set) var stageMixer: AudioBox.Snapshot?
  /// The records packs': the app's list, always on stage.
  private(set) var stageTable: TablePanel?

  private func refreshStagePanel() {
    switch Self.scenarioName {
    case "audio":
      stageMixer = AudioBox.shared.snapshot()
      stageImageID += 1
    case "store":
      stageTable = StoreBox.shared.snapshot()
      stageImageID += 1
    case "shopping":
      stageTable = ShopBox.shared.snapshot()
      stageImageID += 1
    case "money":
      stageTable = MoneyBox.shared.snapshot()
      stageImageID += 1
    case "inbox":
      stageTable = InboxBox.shared.snapshot()
      stageImageID += 1
    default:
      break
    }
  }

  private func refreshStageDoc() {
    guard Self.scenarioIsDocs else { return }
    if let page = DocBox.shared.currentPageImage() { stageImage = page }
    stagePages = DocBox.shared.snapshot()
    stageImageID += 1
  }

  private func refreshStageVideo() async {
    guard Self.scenarioIsVideo else { return }
    let frame = await Task.detached(priority: .userInitiated) {
      await VideoEditBox.shared.currentFrame()
    }.value
    if let frame { stageImage = frame }
    stageTimeline = VideoEditBox.shared.snapshot()
    stageImageID += 1
  }

  var beatCount: Int { Self.scenarioBeats.count }

  /// The bundle to run: `--model <substring>` pins one by filename
  /// (case-insensitive); otherwise the newest bundle wins. Newest-first bit a
  /// whole day of measurements when a pushed 2.6B silently became "the model".
  static func chosenModel() -> URL? {
    let models = ChatModel.availableModels()
    if let flag = CommandLine.arguments.firstIndex(of: "--model"),
      CommandLine.arguments.indices.contains(flag + 1)
    {
      let needle = CommandLine.arguments[flag + 1].lowercased()
      if let match = models.first(where: { $0.lastPathComponent.lowercased().contains(needle) }) {
        return match
      }
    }
    return models.first
  }

  private var session: LanguageModelSession?
  private let pending = TokenBuffer()
  private var pump: Task<Void, Never>?

  func start() async {
    guard session == nil else { return }
    // Tokens arrive faster than a screen can usefully change. They are collected
    // off the actor and published on a timer; one hop and one redraw per token
    // was competing with the generation itself.
    #if canImport(LiteRTLM)
      LiteRTFMTrace.onChunk = { [weak self] piece in
        self?.pending.append(piece)
        RunLog.stream(piece)
      }
      LiteRTFMTrace.onTiming = { line in RunLog.write("TIME \(line)") }
      LiteRTFMTrace.onRate = { [weak self] rate in
        Task { @MainActor in self?.rate = rate }
      }
    #endif
    pump = Task { @MainActor [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .milliseconds(60))
        guard let self, let drained = self.pending.drain() else { continue }
        self.live += drained
      }
    }
    RunLog.startNewRun()
    // A take is minutes of nobody touching the screen; auto-lock would end it.
    UIApplication.shared.isIdleTimerDisabled = true
    let tools: [any FoundationModels.Tool]
    switch Self.scenarioName {
    case "photo": tools = ToolBox.photoStage
    case "focus": tools = ToolBox.focus
    case "report": tools = ToolBox.fieldReport
    case "briefing": tools = ToolBox.briefing
    case "sensors": tools = ToolBox.sensors
    case "handoff": tools = ToolBox.handoff
    case "chains": tools = ToolBox.chains
    // The compound pack carries the single tools too: the point on screen is
    // that the model picks the one call over the three, when one exists.
    case "compound": tools = ToolBox.compound + ToolBox.focus + ToolBox.briefing
    case "vision": tools = ToolBox.vision
    case "look": tools = []
    case "polish": tools = ToolBox.vision
    case "video": tools = ToolBox.video
    case "store": tools = ToolBox.store
    case "audio": tools = ToolBox.audio
    case "docs": tools = ToolBox.docs
    case "shopping": tools = ToolBox.shopping
    case "money": tools = ToolBox.money
    case "inbox": tools = ToolBox.inbox
    case "choose": tools = []  // the model answers; the app calls
    default: tools = ToolBox.demo
    }
    // The vision packs get their own instructions: the stock ones push tools
    // ("prefer a tool over guessing"), which is the wrong bias for a model
    // that can see — it removed the background from a mountain range because
    // a beat said "if there's a person…". The state packs get theirs: the
    // model is told everything and asked to copy numbers, not look them up.
    let instructions =
      Self.scenarioIsChoose
      ? SeeAndChoose.instructions
      : Self.scenarioAttachesPhoto
        ? ToolBox.visionInstructions
        : (Self.scenarioSendsState ? Self.stateInstructions : ToolBox.instructions)
    toolCount = tools.count
    switch Self.backend {
    case .system:
      // Apple's on-device model. Same session, same tools, same cards — the
      // only thing that changes is who generates. It answers in about a second
      // where the LiteRT path spends ten on prefill alone.
      let model = SystemLanguageModel.default
      guard model.availability == .available else {
        question = "the system model is not available on this device"
        return
      }
      backendName = "Apple on-device"
      session = LanguageModelSession(tools: tools, instructions: instructions)
    case .liteRT:
      #if !canImport(LiteRTLM)
        question = "LiteRT is not in this build — launch with --backend apple"
        return
      #else
      // Same process, same bundle, same backend as the beats that follow. The
      // standalone benchmark reported 252 tok/s of prefill where a turn in this
      // app measured about 46; running both here says whether that gap belongs
      // to the conversation path or to the environment it runs in.
      if let url = Self.chosenModel() {
        let bench = await Task.detached(priority: .userInitiated) { () -> String in
          do {
            let info = try await LiteRTLM.benchmark(
              modelPath: url.path, backend: .cpu(), prefillTokens: 256, decodeTokens: 64)
            return String(describing: info)
          } catch {
            return "failed: \(error)"
          }
        }.value
        RunLog.write("BENCH in-process \(bench.prefix(400))")
      }
      guard let url = Self.chosenModel() else {
        question = "no .litertlm bundle in the app container"
        return
      }
      do {
        // A big bundle plus a big KV is what iOS kills the app for: the 2.6B's
        // 1.55 GB of weights left nothing for a 4096-token cache and the
        // process went away during load.
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        // The KV is allocated at this size and every prefill step works against
        // it: the benchmark that reported 252 tok/s used 544, this app used
        // 4096 and measured about 50.
        let context = bytes > 1_000_000_000 ? 1024 : 2048
        // The bundle's own name, not a hardcoded label. `availableModels` is
        // newest-first, and a hardcoded "LFM2.5-1.2B" here spent a day fronting
        // for the 2.6B — every in-app number got attributed to the wrong model.
        backendName = url.deletingPathExtension().lastPathComponent
        // `.bare` writes the tool list in LFM2's trained format — no OpenAI
        // envelope — which is both what the model expects and ~180 fewer
        // prefilled characters across the demo's six tools.
        // The thinking budget is tight on purpose: LFM2.5's reasoning is
        // invisible (a metadata channel the runtime strips from the stream),
        // so on stage it is pure silence — up to 40s of it, sometimes ending
        // in an empty answer. 32 tokens caps the silence at about two seconds.
        let model = try LiteRTLanguageModel(
          modelPath: url.path, backend: .cpu(),
          visionBackend: url.lastPathComponent.uppercased().contains("-VL-") ? .cpu() : nil,
          maxTokens: context, toolListStyle: .bare,
          thinkingTokenBudget: 32)
        session = LanguageModelSession(
          model: model, tools: tools, instructions: instructions)
      } catch {
        question = "could not load the model: \(error.localizedDescription)"
        return
      }
      #endif
    }
    RunLog.write("BACKEND \(backendName)")
    if Self.backend == .system {
      let caps = SystemLanguageModel.default.capabilities
      RunLog.write("CAPS vision=\(caps.contains(.vision)) tools=\(caps.contains(.toolCalling)) reasoning=\(caps.contains(.reasoning))")
    }
    if let session { TranscriptBox.shared.attach(session) }
    if Self.scenarioShowsPhoto {
      // The photo is on stage before the first word is typed; a permission
      // prompt, if any, fires here rather than mid-beat.
      try? await PhotoEditBox.shared.preload(label: SeenPhoto.singleLabel)
      refreshStageImage()
    }
    if Self.scenarioIsVideo {
      // Same for the video: newest in the library, on stage from the first
      // frame, and the state line the model will read is in the log.
      do {
        try await VideoEditBox.shared.preload()
        RunLog.write("VIDEO loaded — \(VideoEditBox.shared.describe())")
      } catch {
        RunLog.write("VIDEO load failed: \(error)")
      }
      await refreshStageVideo()
    }
    if Self.scenarioIsDocs {
      DocBox.shared.preload()
      RunLog.write("DOC loaded — \(DocBox.shared.describe())")
      refreshStageDoc()
    }
    refreshStagePanel()
    await run()
  }

  private func run() async {
    guard let session else { return }
    if Self.scenarioIsChoose {
      await runChoose(session)
      return
    }
    for (index, scripted) in Self.scenarioBeats.enumerated() {
      RunLog.flushStream()  // beat N's tail was landing at the head of beat N+1
      beatIndex = index
      question = ""
      typed = ""
      set(.typing)
      let prompt: String
      if Self.voiceDriven && !scripted.isEmpty {
        // The person speaks the beat. (A silent beat — photo only — stays
        // silent in voice mode too; the silence is the point.) An empty take (a cough, a false start)
        // just listens again; a broken recognizer ends the run — looping on
        // it would record a phone listening to nothing.
        var spoken = ""
        while spoken.isEmpty {
          guard let heard = await listenForOneUtterance() else {
            RunLog.write("VOICE stopped: \(voice.problem ?? "no transcript")")
            return
          }
          spoken = heard
        }
        prompt = spoken
        RunLog.write("BEAT \(index + 1) (spoken) \(prompt)")
      } else {
        RunLog.write("BEAT \(index + 1) \(scripted.isEmpty ? "(photo only)" : scripted)")
        // Typed out rather than announced: on screen this has to read as
        // somebody instructing the phone, not as a chapter heading.
        for character in scripted {
          typed.append(character)
          try? await Task.sleep(for: .milliseconds(28))
        }
        prompt = scripted
      }
      try? await Task.sleep(for: .milliseconds(450))
      // A silent beat still needs a sent message on screen: the photo itself.
      question = prompt.isEmpty ? "📷" : prompt
      typed = ""
      live = ""
      set(.thinking)
      _ = ArtifactBox.shared.take()

      let before = session.transcript.count
      let answer: String
      do {
        // Off the main actor. `respond` is `nonisolated(nonsending)`, so it runs
        // on the caller's executor — and this class is @MainActor. Left alone,
        // the model generates on the main thread while the same thread is being
        // asked to redraw for every token it produces.
        // The vision pack sends the photo as it is now — edits included, so
        // "now make it look its best" is judged on what beat 2 produced.
        let attached = Self.scenarioAttachesPhoto ? PhotoEditBox.shared.currentCGImage() : nil
        if attached != nil { RunLog.write("ATTACH photo \(attached!.width)x\(attached!.height)") }
        // The state packs send the timeline as it is now, ahead of the words:
        // the model is told what beat N-1 did before it is asked for beat N.
        // On screen only the words show; the log has the whole message.
        var message = prompt
        if Self.scenarioSendsState {
          let state = Self.currentState()
          RunLog.write("STATE \(state)")
          message = AppState.compose(state: state, request: prompt)
        }
        answer = try await Task.detached(priority: .userInitiated) {
          if let attached {
            if prompt.isEmpty {
              return try await session.respond(
                to: Prompt { Attachment(attached).label(SeenPhoto.singleLabel) }
              ).content
            }
            return try await session.respond(
              to: Prompt {
                prompt
                Attachment(attached).label(SeenPhoto.singleLabel)
              }
            ).content
          }
          return try await session.respond(to: message).content
        }.value
      } catch {
        RunLog.write("ERROR \(error)")
        answer = error.localizedDescription
      }

      // Replay what the session did, so the tool call gets its own moment on
      // screen instead of being something that already happened.
      let tail = Array(session.transcript).dropFirst(before)
      var pendingCall: (name: String, arguments: String)?
      for entry in tail {
        switch entry {
        case .toolCalls(let calls):
          if let call = calls.first {
            pendingCall = (call.toolName, Self.pretty(call.arguments))
            set(.calling(name: call.toolName, arguments: Self.pretty(call.arguments), returned: nil))
            try? await Task.sleep(for: .milliseconds(900))
          }
        case .toolOutput(let output):
          // What the tool gave back, on screen. The badge alone showed that
          // something was reached; this shows what it said.
          let returned = output.segments.compactMap { segment -> String? in
            if case .text(let text) = segment { return text.content } else { return nil }
          }.joined()
          RunLog.write("TOOL \(output.toolName) -> \(returned.prefix(300))")
          refreshStageImage()
          await refreshStageVideo()
          refreshStageDoc()
          refreshStagePanel()
          if let call = pendingCall {
            set(.calling(name: call.name, arguments: call.arguments, returned: returned))
            try? await Task.sleep(for: .milliseconds(1400))
          }
        default:
          break
        }
      }

      // Everything the model actually produced this beat, so a silent turn can
      // be told apart from a turn that produced something unusable.
      RunLog.write("STREAM[\(live.count)] \(live.replacingOccurrences(of: "\n", with: "⏎").prefix(400))")
      RunLog.write("ANSWER[\(answer.count)] \(answer.prefix(200))")
      LastAnswer.shared.set(answer)
      set(.result(text: answer, artifact: ArtifactBox.shared.take()))
      beatIndex = index + 1
      try? await Task.sleep(for: .seconds(3))
    }
  }

  /// The see-and-choose loop: the app asks, the model answers with an enum,
  /// the app calls the tool. Same screen phases as the tool-calling loop —
  /// the badge shows the call the app made and why — so the recording reads
  /// the same whichever model is under it.
  private func runChoose(_ session: LanguageModelSession) async {
    speaker = "APP → MODEL"
    for (index, step) in SeeAndChoose.steps.enumerated() {
      RunLog.flushStream()
      beatIndex = index
      question = ""
      typed = ""
      set(.typing)
      let shown = Self.chooseBeats[index]
      RunLog.write("BEAT \(index + 1) (ask) \(shown)")
      for character in shown {
        typed.append(character)
        try? await Task.sleep(for: .milliseconds(28))
      }
      try? await Task.sleep(for: .milliseconds(450))
      question = shown
      typed = ""
      live = ""
      set(.thinking)
      _ = ArtifactBox.shared.take()
      guard let photo = PhotoEditBox.shared.currentCGImage() else {
        RunLog.write("ERROR no photo on stage")
        set(.result(text: "no photo to look at", artifact: nil))
        return
      }
      RunLog.write("ATTACH photo \(photo.width)x\(photo.height)")
      let outcome = await Task.detached(priority: .userInitiated) {
        await SeeAndChoose.run(step, session: session, photo: photo)
      }.value
      if !outcome.answer.isEmpty { RunLog.write("CHOSE \(outcome.answer)") }
      if let call = outcome.call {
        RunLog.write("TOOL \(call.name) -> \(call.returned.prefix(300))")
        set(.calling(name: call.name, arguments: call.arguments, returned: nil))
        try? await Task.sleep(for: .milliseconds(900))
        refreshStageImage()
        set(.calling(name: call.name, arguments: call.arguments, returned: call.returned))
        try? await Task.sleep(for: .milliseconds(1400))
      }
      RunLog.write("STREAM[\(live.count)] \(live.replacingOccurrences(of: "\n", with: "⏎").prefix(400))")
      RunLog.write("ANSWER[\(outcome.sentence.count)] \(outcome.sentence.prefix(200))")
      LastAnswer.shared.set(outcome.sentence)
      set(.result(text: outcome.sentence, artifact: ArtifactBox.shared.take()))
      beatIndex = index + 1
      try? await Task.sleep(for: .seconds(3))
    }
  }

  /// One spoken beat: listen until the words stop changing. There is no cap on
  /// the silence before the first word — the speaker decides when the beat
  /// starts — and 1.6 s of unchanged text after it ends the utterance. Returns
  /// nil only when the recognizer itself failed.
  private func listenForOneUtterance() async -> String? {
    listening = true
    defer { listening = false }
    await voice.start()
    guard voice.problem == nil else { return nil }
    var last = ""
    var changedAt = Date()
    while voice.problem == nil {
      try? await Task.sleep(for: .milliseconds(100))
      if voice.heard != last {
        last = voice.heard
        changedAt = Date()
        typed = last  // the words land in the composer as they are said
      }
      if !last.isEmpty, Date().timeIntervalSince(changedAt) > 1.6 { break }
    }
    let text = await voice.stop()
    typed = text
    return voice.problem == nil ? text : nil
  }

  private func set(_ next: Phase) {
    phase = next
    phaseID += 1
  }

  private static func pretty(_ arguments: GeneratedContent) -> String {
    let raw = String(describing: arguments)
    return raw.count > 90 ? String(raw.prefix(90)) + "…" : raw
  }
}


/// Collects stream chunks off the main actor so the UI can take them in batches.
final class TokenBuffer: @unchecked Sendable {
  private let lock = NSLock()
  private var text = ""

  func append(_ piece: String) {
    lock.lock()
    text += piece
    lock.unlock()
  }

  func drain() -> String? {
    lock.lock()
    defer { lock.unlock() }
    guard !text.isEmpty else { return nil }
    let out = text
    text = ""
    return out
  }
}


/// A run log in the app's Documents, pulled back with `devicectl device copy
/// from`. The console stream needs a session attached, and attaching one means
/// something can kill the app when that session ends — which is what happened
/// to a take.
enum RunLog {
  private static let lock = NSLock()

  /// A fresh name per run. Pulling a fixed path with `devicectl copy from`
  /// kept returning a cached, stale copy — which is how three "the log says
  /// nothing happened" reports got made about runs that had happened.
  nonisolated(unsafe) private static var name = "run.log"

  static func startNewRun() {
    lock.lock()
    name = "run-\(Int(Date().timeIntervalSince1970)).log"
    lock.unlock()
  }

  static var url: URL {
    AppFiles.documents
      .appendingPathComponent(name)
  }

  /// Raw tokens as they arrive, coalesced into lines. The thinking is the most
  /// interesting thing on screen during a turn and it was going nowhere.
  nonisolated(unsafe) private static var streamBuffer = ""

  /// Drop whatever is half-collected, so one beat's tail does not open the next.
  static func flushStream() {
    lock.lock()
    streamBuffer = ""
    lock.unlock()
  }

  static func stream(_ piece: String) {
    lock.lock()
    streamBuffer += piece
    let flush = streamBuffer.count > 120 || piece.contains("\n")
    let out = flush ? streamBuffer : ""
    if flush { streamBuffer = "" }
    lock.unlock()
    guard flush, !out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
    write("THINK \(out.replacingOccurrences(of: "\n", with: "⏎"))")
  }

  static func write(_ line: String) {
    lock.lock()
    defer { lock.unlock() }
    let stamped = "\(Date().formatted(date: .omitted, time: .standard))  \(line)\n"
    if let handle = try? FileHandle(forWritingTo: url) {
      defer { try? handle.close() }
      try? handle.seekToEnd()
      try? handle.write(contentsOf: Data(stamped.utf8))
    } else {
      try? Data(stamped.utf8).write(to: url)
    }
  }

  static func reset() {
    lock.lock()
    defer { lock.unlock() }
    try? FileManager.default.removeItem(at: url)
  }
}
