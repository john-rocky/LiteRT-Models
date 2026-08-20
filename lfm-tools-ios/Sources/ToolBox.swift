// Every tool the demo offers, in one list.
//
// The model sees this list as a name + description + argument schema per tool,
// and picks by name. That is the whole contract: adding a capability to the
// agent is adding one struct here, not touching the loop.
import Foundation
import FoundationModels

/// Run a tool's body with a deadline. CoreLocation, CoreMotion and EventKit all
/// answer through callbacks that can simply never arrive — a denied permission,
/// a released manager, a sensor that is not there. Without this the whole agent
/// stops on one of them, which on a phone looks exactly like a slow model.
@available(iOS 27.0, *)
func withDeadline(
  _ seconds: Double = 8, _ what: String, _ body: @escaping @Sendable () async throws -> String
) async throws -> String {
  do {
    return try await firstToFinish(within: seconds, body)
  } catch is DeadlinePassed {
    return "\(what) did not answer within \(Int(seconds))s"
  }
}

struct DeadlinePassed: Error, CustomStringConvertible {
  let seconds: Double
  var description: String { "timed out after \(Int(seconds))s" }
}

/// Whichever finishes first: `body`, or a timer that throws `DeadlinePassed`.
///
/// Not a task group on purpose. A `withThrowingTaskGroup` race looks right
/// and is dead code against a real hang: the group awaits *every* child
/// before it returns, even after the timer child throws, and a child stuck
/// in a callback that never comes — or awaiting a detached task's `.value`,
/// which cancellation does not interrupt — never finishes. The bench watched
/// a hung engine for five minutes past a 180 s "timeout" this way. This
/// version resumes the caller the moment the timer fires and lets the hung
/// work stay hung on its own thread; the caller decides what to do about a
/// process that now has one wedged task in it.
@available(iOS 27.0, *)
func firstToFinish<T: Sendable>(
  within seconds: Double, _ body: @escaping @Sendable () async throws -> T
) async throws -> T {
  let gate = OnceGate<T>()
  return try await withCheckedThrowingContinuation { continuation in
    gate.arm(continuation)
    Task.detached {
      do { gate.finish(.success(try await body())) } catch { gate.finish(.failure(error)) }
    }
    Task.detached {
      try? await Task.sleep(for: .seconds(seconds))
      gate.finish(.failure(DeadlinePassed(seconds: seconds)))
    }
  }
}

/// Resumes a continuation exactly once, from whichever racer gets there first.
final class OnceGate<T: Sendable>: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: CheckedContinuation<T, Error>?

  func arm(_ continuation: CheckedContinuation<T, Error>) {
    lock.lock()
    self.continuation = continuation
    lock.unlock()
  }

  func finish(_ result: Result<T, Error>) {
    lock.lock()
    let pending = continuation
    continuation = nil
    lock.unlock()
    pending?.resume(with: result)
  }
}

@available(iOS 27.0, *)
enum ToolBox {
  /// Tools that need no permission prompt. Start here when checking whether a
  /// model calls anything at all — a refusal dialog in the middle of a demo
  /// looks like the model failed when it did its part.
  static let ambient: [any FoundationModels.Tool] = [
    CurrentTimeTool(), DeviceInfoTool(), BatteryTool(), StorageTool(),
    PowerStateTool(), LocaleTool(), NetworkTool(), CalculateTool(),
    ReadBrightnessTool(), VolumeTool(), OrientationTool(), AudioRouteTool(),
    ScreenInfoTool(), AccessibilitySettingsTool(), TimeZoneTool(),
    ListDocumentsTool(), ReadNotesTool(), PendingNotificationsTool(),
    DetectLanguageTool(), SentimentTool(), TranslateTool(),
  ]

  /// Tools that change something the user can see, hear or feel.
  static let actions: [any FoundationModels.Tool] = [
    BrightnessTool(), TorchTool(), SpeakTool(), HapticTool(),
    ReadClipboardTool(), WriteClipboardTool(), OpenURLTool(), OpenMapsTool(),
    NotificationTool(),
    SystemSoundTool(), BadgeTool(), WriteNoteTool(), CancelNotificationsTool(),
    TimerTool(),
  ]

  /// Tools behind a permission prompt.
  static let personal: [any FoundationModels.Tool] = [
    LocationTool(), PlaceNameTool(), SearchPlacesTool(),
    HeadingTool(), AltitudeTool(), StepsTool(), AttitudeTool(), MotionActivityTool(),
    SoundLevelTool(), ReadPhotoTextTool(), ClassifyPhotoTool(), EditPhotoTool(),
    StepChartTool(),
    ListEventsTool(), CreateEventTool(), ListRemindersTool(), CreateReminderTool(),
    SearchContactsTool(), PhotoLibraryTool(),
  ]

  /// The photo-editing scenario pack. Session-stateful: edits stack, undo
  /// steps back, save writes out. The near-neighbor names (crop/resize/zoom,
  /// brightness/exposure/contrast) are a benchmark axis.
  static let photoEditing: [any FoundationModels.Tool] = [
    CropPhotoTool(), ResizePhotoTool(), ZoomPhotoTool(), RotatePhotoTool(), FlipPhotoTool(),
    BrightnessPhotoTool(), ExposurePhotoTool(), ContrastPhotoTool(),
    SaturationPhotoTool(), WarmthPhotoTool(), FilterPhotoTool(), BlurPhotoTool(),
    AutoEnhancePhotoTool(), CutOutSubjectTool(),
    UndoPhotoEditTool(), ResetPhotoEditsTool(), SavePhotoTool(),
  ]

  /// The stage cut of the photo pack: no undo_photo_edit. Measured, not
  /// cosmetic — every "undo everything / revert / reset" wording routed to
  /// the one-step undo on the 1.2B while it was present. With it gone,
  /// revert_to_original owns the going-back words.
  static let photoStage: [any FoundationModels.Tool] = photoEditing.filter {
    $0.name != "undo_photo_edit"
  }

  /// The focus scenario pack: one vague sentence fanning out into
  /// notifications, a timer and the screen. The discrimination axes are the
  /// `set_` prefix neighbors (timer/brightness/torch), the get/set brightness
  /// pair, and remind-vs-remember (schedule_notification vs write_note).
  static let focus: [any FoundationModels.Tool] = [
    TimerTool(), BrightnessTool(), ReadBrightnessTool(),
    NotificationTool(), CancelNotificationsTool(), PendingNotificationsTool(),
    WriteNoteTool(), ReadNotesTool(), TorchTool(), HapticTool(),
  ]

  /// The field-report scenario pack: a photographed gauge becoming a note and
  /// a next-morning reminder, fully offline. Discrimination axes: the
  /// read/identify photo pair, reminder-vs-calendar, and a date argument
  /// ("tomorrow") the model can only fill by asking the phone what today is.
  static let fieldReport: [any FoundationModels.Tool] = [
    ReadPhotoTextTool(), ClassifyPhotoTool(), PhotoLibraryTool(),
    WriteNoteTool(), ReadNotesTool(),
    CreateReminderTool(), ListRemindersTool(),
    CreateEventTool(), ListEventsTool(), CurrentTimeTool(),
  ]

  /// Quick packs for trying the rest of the phone on Apple's model. Not
  /// benchmarked yet; each becomes a scenario pack the day it earns cases.
  static let briefing: [any FoundationModels.Tool] = [
    CurrentTimeTool(), BatteryTool(), PowerStateTool(),
    ListEventsTool(), ListRemindersTool(), StepsTool(), StepChartTool(),
  ]
  static let sensors: [any FoundationModels.Tool] = [
    LocationTool(), PlaceNameTool(), HeadingTool(), MotionActivityTool(),
    SoundLevelTool(), AltitudeTool(), OrientationTool(), AttitudeTool(),
  ]
  static let handoff: [any FoundationModels.Tool] = [
    TorchTool(), SystemSoundTool(), HapticTool(), BadgeTool(),
    WriteClipboardTool(), ReadClipboardTool(), NotificationTool(), WriteNoteTool(),
    BrightnessTool(),
  ]

  /// Tools that are several tools (Tools/CompoundTools.swift). One name to
  /// say, one call, the app walks the steps — for the jobs that are always
  /// the same three steps, and for the models that cannot chain.
  static let compound: [any FoundationModels.Tool] = [
    FocusSessionTool(), MorningBriefingTool(), ShareLocationTool(), PhotoTextToNoteTool(),
  ]

  /// The chains pack: every beat wants two or three calls out of one
  /// sentence, on tools that have nothing to do with each other. Apple FM
  /// chains; the 1.2B stops after one; this is where that shows.
  static let chains: [any FoundationModels.Tool] = [
    TorchTool(), BatteryTool(), LocationTool(), SearchPlacesTool(), OpenMapsTool(),
    CurrentTimeTool(), ReadPhotoTextTool(), TranslateTool(), SoundLevelTool(),
    WriteNoteTool(),
  ]

  /// The vision pack: the photo is in the prompt as a labelled attachment,
  /// and every tool takes an `ImageReference` — Foundation Models' native
  /// vision: the model looks, decides, and names the picture it means
  /// (Tools/VisionTools.swift). Nothing here needs the user to name the
  /// problem.
  static let vision: [any FoundationModels.Tool] = [
    SeenBrightnessTool(), SeenExposureTool(), SeenContrastTool(), SeenSaturationTool(),
    SeenWarmthTool(), SeenRotateTool(), SeenCropTool(), SeenFilterTool(),
    SeenAutoEnhanceTool(), SeenRemoveBackgroundTool(), SeenReadTextTool(), SeenRedactTool(),
    SeenRevertTool(), SeenSaveTool(), WriteNoteTool(),
  ]

  /// The video-editing pack (Tools/VideoEditTools.swift): a CapCut's menu,
  /// said out loud. The model never sees a frame — the app's state (clips,
  /// selection, playhead, frame size) rides at the top of every message and
  /// the tools take the numbers it names. Trim / split / speed / crop 9:16 /
  /// caption / fade / stabilise / volume / export, in the menu's own words.
  static let video: [any FoundationModels.Tool] = [
    TrimClipTool(), SplitClipTool(), SelectClipTool(), DeleteClipTool(), ClipSpeedTool(),
    CropVideoTool(), AddCaptionTool(), AutoCaptionsTool(), AddFadeTool(), StabilizeVideoTool(),
    VideoVolumeTool(), AddMusicTool(), RemoveMusicTool(), MakeReelTool(),
    UndoLastTool(target: .video), RevertVideoTool(), ExportVideoTool(), AskUserTool(),
  ]

  /// The store pack (Tools/StoreTools.swift): a Shopify admin's menu over
  /// canned products and orders — the Commerce pack of the business wing,
  /// extended to the spec on 2026-08-20. A filter or search makes a
  /// selection; the bulk actions act on it. Discrimination axes:
  /// adjust_product_price (relative) vs set_price (absolute); search vs
  /// filter vs find_low_stock vs get_product; fulfil vs refund vs cancel
  /// (the status-verb triangle, both gated); customers vs their orders;
  /// and a selection that has to exist before a bulk action.
  static let store: [any FoundationModels.Tool] = [
    SearchProductsTool(), FilterProductsTool(), LowStockTool(), GetProductTool(),
    AdjustProductPriceTool(), SetPriceTool(), AddTagTool(), SetProductStatusTool(), AdjustInventoryTool(),
    SearchOrdersTool(), FilterOrdersTool(), FulfillOrdersTool(), SendInvoiceTool(),
    RefundOrderTool(), CancelOrderTool(), OrderNoteTool(),
    SearchCustomersTool(), CreateDiscountTool(),
    SalesSummaryTool(), ExportProductsTool(),
    UndoLastTool(target: .store), AskUserTool(),
  ]

  /// The audio pack (Tools/AudioTools.swift): a GarageBand mixer over four
  /// synthesized tracks. Volume and pan are numbers the model reads from
  /// the state and moves; mute vs solo, add vs remove effect, and a track
  /// named by what the user calls it are the axes.
  static let audio: [any FoundationModels.Tool] = [
    TrackVolumeTool(), TrackPanTool(), MuteTrackTool(), SoloTrackTool(),
    AddEffectTool(), RemoveEffectTool(), DuplicateTrackTool(), DeleteTrackTool(), RenameTrackTool(),
    SetTempoTool(), SetBarsTool(), SongFadeTool(), PlaySongTool(), StopSongTool(),
    ExportSongTool(), UndoLastTool(target: .audio), RevertSongTool(), AskUserTool(),
  ]

  /// The documents pack (Tools/DocTools.swift): an Acrobat / Goodnotes menu
  /// on a real PDF through PDFKit. Pages are named in the state by their
  /// first line, so "the cover" and "the rules page" are page numbers the
  /// model reads; "the last page" is the page count.
  static let docs: [any FoundationModels.Tool] = [
    GoToPageTool(), DeletePageTool(), MovePageTool(), RotatePageTool(), InsertBlankPageTool(),
    HighlightTextTool(), RemoveHighlightsTool(), RedactTextTool(), AddNoteTool(), SignPageTool(),
    WatermarkTool(), FillFieldTool(), SearchDocumentTool(), ExtractPagesTool(), SavePDFTool(),
    UndoLastTool(target: .docs), RevertDocumentTool(), AskUserTool(),
  ]

  /// The shopping pack (Tools/ShoppingTools.swift): the buyer's side of the
  /// counter — search → sort → "the second one" → cart → coupon → checkout.
  /// Numbers come from the numbered results in the state; the cart maths is
  /// the app's.
  static let shopping: [any FoundationModels.Tool] = [
    SearchCatalogTool(), FilterResultsTool(), SortResultsTool(), ShowProductTool(),
    AddToCartTool(), ChangeQuantityTool(), RemoveFromCartTool(),
    ApplyCouponTool(), CheckoutTool(), TrackOrderTool(),
    UndoLastTool(target: .shopping), AskUserTool(),
  ]

  /// The money pack (Tools/MoneyTools.swift): a household-budget app over a
  /// month of canned spending. Finders select; categorize/flag act on the
  /// selection; the reports do the arithmetic in the app.
  static let money: [any FoundationModels.Tool] = [
    ListTransactionsTool(), FilterTransactionsTool(), SearchPayeeTool(),
    CategorizeTool(), FlagTransactionsTool(),
    SetBudgetTool(), SpendingReportTool(), BudgetReportTool(), FindSubscriptionsTool(),
    UndoLastTool(target: .money), AskUserTool(),
  ]

  /// The CRM pack (Tools/CRMTools.swift): a Salesforce's pipeline over a
  /// frozen quarter. Finders select; every action takes its record as an id
  /// argument — no bulk tools, so nothing needs a selection to act. The
  /// similar-tool triple (update stage / update amount / assign) is a
  /// deliberate axis, and the frozen today in the state makes relative
  /// dates scorable as absolute ones.
  static let crm: [any FoundationModels.Tool] = [
    SearchOpportunitiesTool(), GetOpportunityTool(),
    UpdateOpportunityStageTool(), UpdateOpportunityAmountTool(), AssignOpportunityTool(),
    CrmSearchContactsTool(), CrmGetContactTool(), CrmSearchCompaniesTool(),
    CreateFollowUpTaskTool(), CrmAddNoteTool(),
    UndoLastTool(target: .crm), AskUserTool(),
  ]

  /// The PM pack (Tools/PMTools.swift): a Jira board over a frozen sprint.
  /// One finder makes the selection; every action takes its issue id. The
  /// four change_* verbs (assign / status / priority / due date) and
  /// close-vs-status are deliberately similar — the pack is dense with the
  /// bench's central question on purpose.
  static let pm: [any FoundationModels.Tool] = [
    SearchIssuesTool(), GetIssueTool(), CreateIssueTool(),
    AssignIssueTool(), ChangeIssueStatusTool(), ChangeIssuePriorityTool(), ChangeDueDateTool(),
    AddCommentTool(), CloseIssueTool(),
    UndoLastTool(target: .pm), AskUserTool(),
  ]

  /// The inbox pack (Tools/InboxTools.swift): mail triage over fifteen canned
  /// messages. list/search select; archive/snooze/flag act on the selection;
  /// draft_reply writes, never sends.
  static let inbox: [any FoundationModels.Tool] = [
    ListInboxTool(), SearchMailTool(), ReadMessageTool(),
    ArchiveTool(), MarkReadTool(), FlagMailTool(), SnoozeTool(), DeleteMessageTool(),
    DraftReplyTool(), UnsubscribeTool(),
    UndoLastTool(target: .inbox), AskUserTool(),
  ]

  static let all: [any FoundationModels.Tool] =
    ambient + actions + personal + photoEditing + compound

  /// The set the scripted run uses.
  ///
  /// Not a size optimization — a correctness one. Routing across all 54 is past
  /// what a 1.2B can do: "read the text in my photo" went to `get_audio_route`,
  /// "what is my compass heading" to `get_orientation`. Six distinct jobs
  /// it can tell apart. The whole set stays in the tool sheet.
  static let demo: [any FoundationModels.Tool] = [
    LocationTool(), SearchPlacesTool(), ReadPhotoTextTool(), TranslateTool(),
    SpeakLastAnswerTool(), OpenMapsTool(),
  ]

  /// The system prompt. Kept short on purpose: it is prefilled on every turn,
  /// and on a 1–3B model running on the phone every sentence here is paid for
  /// twice — once in latency, once in the attention it takes from the question.
  // Positive imperative, no "you cannot" list: told it "cannot speak out
  // loud yourself", the 1.2B apologized that it can't speak instead of
  // calling the speak tool. Small models quote their limitations back.
  static let instructions = """
    You are running on the user's iPhone and can operate it through tools.
    Prefer a tool over guessing: you cannot know the time, the battery level or
    where the phone is without calling one. When a tool matches the request,
    call it instead of answering yourself. Call one tool at a time. When a
    tool has returned, answer the user in one short sentence using its result.
    """

  static let shoppingInstructions = """
    You are operating the user's shopping app through tools. Every message
    starts with the app's current state: the numbered search results and the
    cart with its total. "The second one" means result 2 in that list — take
    numbers from the state, never guess them. The results in the state are
    live: sort_results and add_to_cart act on them directly — search only
    when the state has no results for what the user wants. track_order
    answers where an order is. Quantity is 1 unless the user says otherwise. When a request lists
    several steps, call the tools one after another in that order. When the
    request does not say which item at all, call ask_user instead of
    guessing — but never ask about a detail the request or the state
    already gives. Do
    only what was asked: after a search returns, report the results and
    stop. Answer questions about prices or the cart from the
    state without a tool. When a tool has returned, answer the user in one
    short sentence using its result.
    """

  static let moneyInstructions = """
    You are operating the user's budgeting app through tools. Every message
    starts with the app's current state: the month's totals, the budgets, and
    the current selection — the transactions the last list, search or filter
    found. Categorize and flag apply to that selection: find first, then act.
    Amounts are yen. When a request lists several steps, call the tools one
    after another in that order. When something required is truly absent — "set a budget" with no
    category or amount — call ask_user; never ask about a detail the
    request or the state already gives. Do only what was asked:
    after a finder returns, report what it found and stop. Answer questions the state already answers
    without a tool. When a tool has returned, answer the user in one short
    sentence using its result.
    """

  static let crmInstructions = """
    You are operating the user's CRM through tools. Every message starts with
    the app's current state: today's date, the pipeline by stage, the owners
    and companies, and the current selection — the records the last search
    found. Tools that take an id (get_opportunity, get_contact, update stage,
    update amount, assign, add_note, a task's record) act straight on an id
    from the request or the selection in the state — no search call first.
    Showing or listing records is a search call — the counts in the state
    only answer how-many questions. Amounts are yen. Dates in arguments are
    absolute, YYYY-MM-DD: count them from the today the state names. When a
    request lists several steps, call the tools one after another in that
    order. When something required is truly absent — an update with no record
    named anywhere — call ask_user; never ask about a detail the request or
    the state already gives. Do only what was asked: after a search returns,
    report the results and stop. Answer questions the state already answers
    without a tool. When a tool has returned, answer the user in one short
    sentence using its result.
    """

  static let pmInstructions = """
    You are operating the user's issue tracker through tools. Every message
    starts with the board's current state: today's date, the projects, the
    issue counts by status and priority, the assignees, and the current
    selection — the issues the last search found. Tools that take an issue
    id (get_issue, assign, change status, change priority, change due date,
    add_comment, close) act straight on an id from the request or the
    selection in the state — no search call first. Showing or listing
    issues is a search call — the counts in the state only answer how-many
    questions. Dates in arguments are absolute, YYYY-MM-DD: count them from
    the today the state names. When a request lists several steps, call the
    tools one after another in that order. When something required is truly
    absent — a change with no issue named anywhere — call ask_user; never
    ask about a detail the request or the state already gives. Do only what
    was asked: after a search returns, report the results and stop. Answer
    questions the state already answers without a tool. When a tool has
    returned, answer the user in one short sentence using its result.
    """

  static let inboxInstructions = """
    You are operating the user's mail app through tools. Every message starts
    with the app's current state: the newest messages by number, unread
    counts, and the current selection — the messages the last list or search
    found. Tools that take a message number (read, reply, unsubscribe,
    delete) act straight on a number from the state — no finder call first.
    The bulk tools (archive, snooze, flag, mark-read) act on the selection:
    list_inbox or search_mail makes the selection, the bulk call acts on it —
    make both calls. list_inbox slices the inbox (unread, read, flagged,
    newsletters); search_mail is for a name or words — prefer list_inbox when
    the slice has a name, and call a finder once, not repeatedly. Message
    numbers come from the state, never guess one — call ask_user when the
    request does not say which message at all, but never ask about a detail
    the request or the state already gives. Do only what was asked, nothing
    more. Drafting a reply never sends anything. When a tool has returned,
    answer the user in one short sentence using its result.
    """

  /// For packs where the photo is in the prompt. The stock instructions tell
  /// the model to prefer a tool over guessing, which is right when it cannot
  /// see and wrong when it can: it should look first, and a conditional
  /// ("if there's a person…") is answered by the pixels, not by the tool list.
  static let visionInstructions = """
    You are running on the user's iPhone and can see the photo attached to each
    message. Look at the photo first and describe what is actually there when
    asked. Call a photo tool only when the photo itself calls for it — an edit
    the picture needs, text that is really in it, a person who is really in
    it. When the condition is not met, say so and call nothing. Pass the
    photo's label as the image argument. A photo sent with no words means:
    make it look its best — judge what this picture needs (exposure,
    brightness, warmth, contrast, color, straightening) and apply those edits
    one after another, gently, then say in one sentence what you changed and
    why. When a tool has returned, answer in one short sentence.
    """

  /// For packs where the app's state is in the message. The stock line
  /// "you cannot know X without calling a tool" is wrong here — the model
  /// is told everything it needs at the top of each message and its job is
  /// to copy those numbers into arguments, not to look anything up.
  static let videoInstructions = """
    You are operating the user's video editor through tools. Every message
    starts with the app's current state: the clips on the timeline with their
    times in seconds, which clip is selected, where the playhead is, and the
    frame size. Take times from that state — the playhead, a clip's start or
    end — never guess one. Tools act on the selected clip unless they name a
    clip. When a request lists several edits, call the tools one after
    another in that order. When something required is truly absent — a caption
    with no words given — call ask_user. Never ask about a detail the
    request or the state already gives, and never ask to confirm an edit. When a tool has
    returned, answer the user in one short sentence using its result.
    """

  /// The store pack's: same idea, records instead of a timeline. The
  /// selection is the app's; "their prices" means the products the last
  /// filter found, and an action with nothing selected is a filter the
  /// model forgot to make.
  static let storeInstructions = """
    You are operating the user's online store admin through tools. Every
    message starts with the store's current state: product and order counts,
    and the current selection — the products or orders the last search or
    filter found. Actions on prices, tags, status, stock, fulfilment and
    payment reminders apply to the current selection: search or filter
    first, then act. search_products matches product names only; lists by
    vendor, tag, type or status come from filter_products, and lists of
    orders from filter_orders or search_orders. Showing or listing records
    is a finder call — the counts in the state only answer how-many
    questions. refund_order and cancel_order take their order number
    directly; do not search
    first, and never invent a number — call ask_user when one is missing,
    but never ask about a detail the request or the state already gives. Do only
    what was asked, nothing more.
    When a request lists several steps, call the tools one after another in
    that order. When a tool has returned, answer the user in one short
    sentence using its result.
    """
    // `--usd` recordings: the fixtures speak dollars, so the instructions
    // must too — the measured yen wording above stays byte-identical
    // whenever the flag is absent (every bench run).
    + (CommandLine.arguments.contains("--usd") ? " Prices are in dollars." : "")

  /// The merged business workspace in one instruction text — the unified
  /// side of the instructions A/B (the pinned side runs each pack's own
  /// text over the same 41 tools). The shared skeleton is stated once;
  /// each pack keeps its load-bearing contract; what a pinned text carried
  /// implicitly — *which* app this sentence is about — is gone on purpose.
  static let businessInstructions = """
    You are operating the user's business workspace through tools: a CRM
    (contacts, companies, deals, tasks), an issue board (projects,
    issues), and an online store admin (products, orders, customers) in
    one app. Every message starts with the app's current state: what it
    holds, the current selection — the records the last search or filter
    found — and, where dates matter, today's date. Route by the record
    the words name: deals and the pipeline are the CRM, issues and bugs
    are the board, products and orders are the store. Tools that take an
    id or an order number act straight on it from the request or the
    selection — no search call first, and never invent one: when it is
    truly missing, call ask_user. Store actions on prices, tags, status,
    stock, fulfilment and payment reminders apply to the current
    selection: search or filter first, then act. search_products matches
    product names only; lists by vendor, tag, type or status come from
    filter_products, and lists of orders from filter_orders or
    search_orders. Showing or listing records is a search call — the
    counts in the state only answer how-many questions. Amounts are yen.
    Dates in arguments are absolute, YYYY-MM-DD: count them from the
    today the state names. When a request lists several steps, call the
    tools one after another in that order. Never ask about a detail the
    request or the state already gives. Do only what was asked: after a
    search returns, report the results and stop. Answer questions the
    state already answers without a tool. When a tool has returned,
    answer the user in one short sentence using its result.
    """

  static let audioInstructions = """
    You are operating the user's music app's mixer through tools. Every
    message starts with the song's current state: the tracks by number and
    name with their volume, pan, effects and mute or solo, the tempo, and
    whether it is playing. Name tracks the way the state does. Take current
    numbers from the state and change them by what the request implies — "a
    bit quieter" is about 15 less than the level shown. When a request lists
    several changes, call the tools one after another in that order. When
    something required is truly absent — "change the tempo" with no tempo —
    call ask_user; never ask about a detail the request or the state
    already gives. Answer questions about the mix from the state without a tool. When a tool has returned, answer the
    user in one short sentence using its result.
    """

  static let docsInstructions = """
    You are operating the user's PDF app through tools. Every message starts
    with the document's current state: the page count, the open page, each
    page's title, and what is annotated. Pages are numbers: "the cover" is
    the page whose title says so, "the last page" is the page count. Take
    page numbers from that state, never guess one. The state lists page
    titles, not their contents — a question about what the document says
    needs search_document. When a request lists
    several steps, call the tools one after another in that order. When something required is truly absent — a note with no words given —
    call ask_user; never ask about a detail the request or the state
    already gives. Answer
    questions about the document from the state without a tool. When a tool has returned,
    answer the user in one short sentence using its result.
    """

  /// For the chat, where photos are one input among many: the stock
  /// instructions plus what a photo means. Appended, not swapped — the chat
  /// still needs "you cannot know the battery without calling".
  static let instructionsWithVision = instructions + """

    When a photo is attached, look at it first; a photo tool's image argument
    is the photo's label. A photo sent with no words means: make it look its
    best — judge what this picture needs (exposure, brightness, warmth,
    contrast, color, straightening), apply those edits one after another,
    gently, then say in one sentence what you changed and why.
    """

  static func summary(for tools: [any FoundationModels.Tool]) -> String {
    tools.map(\.name).joined(separator: ", ")
  }
}
