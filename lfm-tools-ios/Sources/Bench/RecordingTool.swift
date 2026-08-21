// A tool with the real schema and no body.
//
// The benchmark has to show the model exactly what the app shows it — same
// name, same description, same argument schema, so routing and guided decoding
// are exercised unchanged — while calling it must move nothing in the world:
// no location prompt, no speech, no app switch to Maps mid-run. `call` returns
// a canned result and that is all. What the model actually called is read back
// from the session transcript afterwards, the same way the stage replay does.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
struct RecordingTool<Base: Tool>: Tool {
  let base: Base
  /// The result every call returns. Canned on purpose: a deterministic world
  /// is the only one where "the model chained search → maps correctly" is a
  /// checkable claim.
  let canned: String
  /// For tools whose honest answer depends on the arguments (a confirm
  /// gate's two branches, a result that must echo a number): a closure that
  /// renders the canned answer from what was actually passed. The world
  /// stays deterministic; the shape stays honest.
  var respond: (@Sendable (Base.Arguments) -> String)? = nil

  var name: String { base.name }
  var description: String { base.description }
  var parameters: GenerationSchema { base.parameters }

  func call(arguments: Base.Arguments) async throws -> String {
    respond?(arguments) ?? canned
  }
}

/// What the fakes believe is selected — primed per case from the case's
/// state line, replaced by each finder echo, read by the bulk fakes. The
/// always-success canned results lied exactly where the bench needed
/// honesty: "Snooze it." with "Selection: none" in the state got back
/// "snoozed the selected messages", where the real app refuses. The fake
/// cannot change the model's first call (the result arrives after it),
/// but it keeps the bench from scoring a trajectory the real app would
/// refuse, and lets mid-case recovery be measured.
@available(iOS 27.0, *)
final class BenchSelection: @unchecked Sendable {
  static let shared = BenchSelection()
  private let lock = NSLock()
  private var storage: String?

  /// nil = nothing selected; otherwise "products", "orders" or "rows" —
  /// the store's order tools refuse a product selection, like the app.
  var kind: String? {
    get {
      lock.lock()
      defer { lock.unlock() }
      return storage
    }
    set {
      lock.lock()
      defer { lock.unlock() }
      storage = newValue
    }
  }

  func prime(from state: String?) {
    guard let state, let range = state.range(of: "Selection: ") else {
      kind = nil
      return
    }
    // "none…", "6 products from stock below 5: …", "5 orders from payment
    // pending: …", "5 from uncategorized: …" — the head names the kind.
    let head = state[range.upperBound...].prefix { $0 != ":" }
    if head.hasPrefix("none") {
      kind = nil
    } else if head.contains("order") {
      kind = "orders"
    } else if head.contains("product") {
      kind = "products"
    } else {
      kind = "rows"
    }
  }
}

@available(iOS 27.0, *)
enum BenchToolBox {
  /// The demo six, neutralized. Canned results describe one fixed world —
  /// a cafe called CAFE LA nearby, a coffee-menu photo — so multi-tool cases
  /// have something coherent to chain on.
  static let demo: [any FoundationModels.Tool] = [
    RecordingTool(base: LocationTool(), canned: "Chuo, Osaka (fix to ±25 m)"),
    RecordingTool(
      base: SearchPlacesTool(),
      canned: "- CAFE LA (140 m)\n- Blue Bottle Coffee (240 m)\n- Doutor (310 m)"),
    RecordingTool(
      base: ReadPhotoTextTool(), canned: "COFFEE MENU\nLatte ¥520\nAmericano ¥450"),
    RecordingTool(base: TranslateTool(), canned: "(translated text)"),
    RecordingTool(base: SpeakLastAnswerTool(), canned: "spoke the previous answer"),
    RecordingTool(base: OpenMapsTool(), canned: "opened it in Maps"),
  ]

  /// The photo-editing pack, neutralized. Every canned result claims success
  /// in the shape the real tool answers with — a result that reads wrong
  /// makes a model retry the call (Apple FM re-translated twice on a canned
  /// result that didn't look translated).
  static let photo: [any FoundationModels.Tool] = [
    RecordingTool(base: CropPhotoTool(), canned: "done: cropped — the result is on screen"),
    RecordingTool(base: ResizePhotoTool(), canned: "done: resized — the result is on screen"),
    RecordingTool(base: ZoomPhotoTool(), canned: "done: zoomed — the result is on screen"),
    RecordingTool(base: RotatePhotoTool(), canned: "done: rotated — the result is on screen"),
    RecordingTool(base: FlipPhotoTool(), canned: "done: flipped — the result is on screen"),
    RecordingTool(
      base: BrightnessPhotoTool(), canned: "done: brightness adjusted — the result is on screen"),
    RecordingTool(
      base: ExposurePhotoTool(), canned: "done: exposure adjusted — the result is on screen"),
    RecordingTool(
      base: ContrastPhotoTool(), canned: "done: contrast adjusted — the result is on screen"),
    RecordingTool(
      base: SaturationPhotoTool(), canned: "done: saturation adjusted — the result is on screen"),
    RecordingTool(
      base: WarmthPhotoTool(), canned: "done: warmth adjusted — the result is on screen"),
    RecordingTool(base: FilterPhotoTool(), canned: "done: look applied — the result is on screen"),
    RecordingTool(base: BlurPhotoTool(), canned: "done: blurred — the result is on screen"),
    RecordingTool(
      base: AutoEnhancePhotoTool(), canned: "done: auto-enhanced — the result is on screen"),
    RecordingTool(
      base: CutOutSubjectTool(),
      canned: "done: remove the background — the result is on screen"),
    RecordingTool(base: UndoPhotoEditTool(), canned: "undid the last edit"),
    RecordingTool(
      base: ResetPhotoEditsTool(),
      canned: "discarded all edits — back to the original photo"),
    RecordingTool(
      base: SavePhotoTool(), canned: "saved a copy with 2 edits to the photo library"),
  ]

  /// The focus pack, neutralized. One fixed world again: two notifications
  /// pending, one note remembered — so list/read cases have something real-
  /// shaped to answer from and cancel has something to claim it cancelled.
  static let focus: [any FoundationModels.Tool] = [
    RecordingTool(
      base: TimerTool(), canned: "timer set — it will ring in the system, not in this app"),
    RecordingTool(base: BrightnessTool(), canned: "brightness set"),
    RecordingTool(base: ReadBrightnessTool(), canned: "35%"),
    RecordingTool(base: NotificationTool(), canned: "notification scheduled"),
    RecordingTool(base: CancelNotificationsTool(), canned: "cancelled 2"),
    RecordingTool(
      base: PendingNotificationsTool(),
      canned: "- Stretch in 1800s\n- Stand up in 2700s"),
    RecordingTool(base: WriteNoteTool(), canned: "noted"),
    RecordingTool(
      base: ReadNotesTool(),
      canned: "2026-08-18T09:12:00Z  the wifi password is 4471"),
    RecordingTool(base: TorchTool(), canned: "torch on"),
    RecordingTool(base: HapticTool(), canned: "played the success haptic"),
  ]

  /// The field-report pack, neutralized — except the clock. A fixed world
  /// again (a gauge photo, one note, one open reminder), but get_current_time
  /// stays the real tool: `dateResolvesTo` scores against the device's date
  /// at run time, and a canned "today" would silently break every tomorrow
  /// case the day after it was written. Create results claim success without
  /// echoing specifics a mismatch could contradict.
  static let report: [any FoundationModels.Tool] = [
    RecordingTool(
      base: ReadPhotoTextTool(), canned: "TANK 3\nPRESSURE 82 PSI\nCHECKED 08:40"),
    RecordingTool(base: ClassifyPhotoTool(), canned: "gauge 92%, pipe 74%"),
    RecordingTool(
      base: PhotoLibraryTool(),
      canned: "1204 photos, 87 videos, most recent Aug 18, 2026 at 8:40"),
    RecordingTool(base: WriteNoteTool(), canned: "noted"),
    RecordingTool(
      base: ReadNotesTool(), canned: "2026-08-18T08:41:22Z  TANK 3 PRESSURE 82 PSI"),
    RecordingTool(base: CreateReminderTool(), canned: "added the reminder"),
    RecordingTool(base: ListRemindersTool(), canned: "- File the report"),
    RecordingTool(base: CreateEventTool(), canned: "created the event on the calendar"),
    RecordingTool(base: ListEventsTool(), canned: "- Aug 19 14:00 Site inspection"),
    CurrentTimeTool(),
  ]

  /// The video pack, neutralized. Results claim success in the shape the
  /// real tools answer with, against the one fixed timeline every video
  /// case's `state` describes (see the cases file): one 12.4 s landscape
  /// clip, playhead at 5 s. Nothing here renders.
  static let video: [any FoundationModels.Tool] = [
    RecordingTool(base: TrimClipTool(), canned: "trimmed — the timeline is on screen"),
    RecordingTool(base: SplitClipTool(), canned: "split — clip 1 is selected"),
    RecordingTool(base: SelectClipTool(), canned: "selected"),
    RecordingTool(base: DeleteClipTool(), canned: "deleted — 1 clip left"),
    RecordingTool(base: ClipSpeedTool(), canned: "speed set — the timeline is on screen"),
    RecordingTool(base: CropVideoTool(), canned: "cropped — the frame is on screen"),
    RecordingTool(base: AddCaptionTool(), canned: "caption added"),
    RecordingTool(base: AddFadeTool(), canned: "fade added (picture and sound)"),
    RecordingTool(base: StabilizeVideoTool(), canned: "stabilization on"),
    RecordingTool(base: VideoVolumeTool(), canned: "volume set"),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .video), canned: "undid the last change"),
    RecordingTool(
      base: AutoCaptionsTool(),
      canned: "transcribed the speech on the device and added 3 captions where it was said"),
    RecordingTool(base: AddMusicTool(), canned: "calm music added under the whole video"),
    RecordingTool(base: RemoveMusicTool(), canned: "music removed"),
    RecordingTool(
      base: MakeReelTool(),
      canned: "made a Reel — cropped to 9:16; fade out over 1 s; exported to the photo library"),
    RecordingTool(base: RevertVideoTool(), canned: "discarded all edits — back to the original video"),
    RecordingTool(base: ExportVideoTool(), canned: "exported to the photo library"),
  ]

  /// The moment-seek pack: the video fakes plus the canned index. The
  /// searches echo the frozen match dynamically (like the store's finders);
  /// seek and keep_range echo the numbers they were handed — the timestamp
  /// copy from a result into an argument is the measured competence.
  static let moments: [any FoundationModels.Tool] =
    video + [
      RecordingTool(
        base: SearchFramesTool(), canned: "",
        respond: { MomentEcho.search(MomentEcho.frames, in: "the picture", query: $0.query) }),
      RecordingTool(
        base: SearchTranscriptTool(), canned: "",
        respond: { MomentEcho.search(MomentEcho.transcript, in: "the speech", query: $0.query) }),
      RecordingTool(
        base: SearchScreenTextTool(), canned: "",
        respond: {
          MomentEcho.search(MomentEcho.screenText, in: "the on-screen text", query: $0.query)
        }),
      RecordingTool(
        base: CheckMomentTool(), canned: "",
        respond: { MomentEcho.check(seconds: $0.seconds, question: $0.question, options: $0.options) }),
      RecordingTool(
        base: SeekTool(), canned: "",
        respond: { "playhead at \(VideoEditBox.f($0.seconds)) s — the frame is on screen" }),
      RecordingTool(
        base: KeepRangeTool(), canned: "",
        respond: {
          "kept \(VideoEditBox.f($0.start_seconds))–\(VideoEditBox.f($0.end_seconds)) s — the rest is cut"
        }),
      // The ritual sink (r45): the same neutral acknowledgement the real
      // tool returns, since the real one has no world to move either.
      RecordingTool(base: DoneTool(), canned: "acknowledged"),
    ]

  /// The store pack, neutralized. The canned world is the pack's own
  /// canned data, frozen at "6 products under 5 in stock, 5 orders awaiting
  /// payment"; results claim success in the shape the real tools answer.
  static let store: [any FoundationModels.Tool] = [
    // Finder fakes echo DYNAMICALLY: a pure render over the frozen canned
    // data with the real arguments reflected. The earlier neutral fakes
    // ("the matching products are selected and listed on screen") existed
    // because a FIXED echo ("4 products (draft)") contradicted other
    // queries — reflecting the arguments removes the contradiction instead
    // of the content, and gives the model rows to answer from rather than
    // a reason to call something else (the JA spurious second call).
    RecordingTool(base: SearchProductsTool(), canned: "", respond: { StoreEcho.search($0.query) }),
    RecordingTool(base: FilterProductsTool(), canned: "", respond: { StoreEcho.filter(by: $0.by, value: $0.value) }),
    RecordingTool(base: LowStockTool(), canned: "", respond: { StoreEcho.lowStock(below: $0.below) }),
    RecordingTool(base: GetProductTool(), canned: "", respond: { StoreEcho.product($0.name) }),
    // Bulk fakes are honest about the selection: with nothing selected the
    // real app refuses, and so do these (BenchSelection, primed per case).
    RecordingTool(base: AdjustProductPriceTool(), canned: "", respond: { _ in
      StoreFake.onProducts("prices changed on the selected products") }),
    RecordingTool(base: SetPriceTool(), canned: "", respond: { _ in
      StoreFake.onProducts("price set on the selected products") }),
    RecordingTool(base: AddTagTool(), canned: "", respond: { _ in
      StoreFake.onProducts("tagged the selected products") }),
    RecordingTool(base: SetProductStatusTool(), canned: "", respond: { _ in
      StoreFake.onProducts("status changed on the selected products") }),
    RecordingTool(base: AdjustInventoryTool(), canned: "", respond: { _ in
      StoreFake.onProducts("stock adjusted on the selected products") }),
    RecordingTool(base: SearchOrdersTool(), canned: "", respond: { StoreEcho.searchOrders(customer: $0.customer) }),
    RecordingTool(base: ExportProductsTool(), canned: "exported 24 products to products.csv in the app's Documents"),
    RecordingTool(
      base: FilterOrdersTool(), canned: "",
      respond: { StoreEcho.filterOrders(payment: $0.payment_status, fulfillment: $0.fulfillment_status) }),
    RecordingTool(base: FulfillOrdersTool(), canned: "", respond: { _ in
      StoreFake.onOrders("fulfilled 5 orders: #1010, #1013, #1016, #1017, #1019") }),
    RecordingTool(base: SendInvoiceTool(), canned: "", respond: { _ in
      StoreFake.onOrders("sent a payment reminder for 5 orders") }),
    RecordingTool(
      base: RefundOrderTool(), canned: "",
      respond: { args in
        args.confirm
          ? "refunded order #\(args.order_number) in full"
          : "refunding order #\(args.order_number) is permanent — ask the user to confirm, then call refund_order again with confirm true"
      }),
    RecordingTool(
      base: CancelOrderTool(), canned: "",
      respond: { args in
        args.confirm
          ? "cancelled order #\(args.order_number) — it will not ship"
          : "cancelling order #\(args.order_number) is permanent — ask the user to confirm, then call cancel_order again with confirm true"
      }),
    RecordingTool(
      base: SearchCustomersTool(), canned: "",
      respond: { StoreEcho.customers(name: $0.name, email: $0.email, spentAbove: $0.total_spent_above) }),
    RecordingTool(base: CreateDiscountTool(), canned: "discount created — it is in the discounts list"),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .store), canned: "undid the last change"),
    RecordingTool(base: OrderNoteTool(), canned: "", respond: { _ in
      StoreFake.onOrders("note added to the selected orders") }),
    RecordingTool(base: SalesSummaryTool(), canned: "last 7 days: 13 orders, ¥104,500 in sales, up 44% on the 7 days before"),
  ]

  /// The audio pack, neutralized, against the state the cases carry (four
  /// tracks at 110 bpm, stopped).
  static let audio: [any FoundationModels.Tool] = [
    RecordingTool(base: TrackVolumeTool(), canned: "volume set"),
    RecordingTool(base: TrackPanTool(), canned: "panned"),
    RecordingTool(base: MuteTrackTool(), canned: "mute changed"),
    RecordingTool(base: SoloTrackTool(), canned: "solo changed"),
    RecordingTool(base: AddEffectTool(), canned: "effect added"),
    RecordingTool(base: RemoveEffectTool(), canned: "effect removed"),
    RecordingTool(base: DuplicateTrackTool(), canned: "duplicated as track 5"),
    RecordingTool(base: DeleteTrackTool(), canned: "deleted; 3 tracks left"),
    RecordingTool(base: RenameTrackTool(), canned: "renamed"),
    RecordingTool(base: SetTempoTool(), canned: "tempo set"),
    RecordingTool(base: SetBarsTool(), canned: "the song is now 16 bars (34.9 s)"),
    RecordingTool(base: SongFadeTool(), canned: "fade added"),
    RecordingTool(base: PlaySongTool(), canned: "playing from 0 s (17.5 s song, 110 bpm)"),
    RecordingTool(base: StopSongTool(), canned: "stopped"),
    RecordingTool(base: ExportSongTool(), canned: "exported 17.5 s of 4 tracks to mix.m4a"),
    RecordingTool(base: RevertSongTool(), canned: "back to the original mix — 4 tracks at 110 bpm"),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .audio), canned: "undid the last change"),
  ]

  /// The documents pack, neutralized, against the six-page lease the cases
  /// describe.
  static let docs: [any FoundationModels.Tool] = [
    // Neutral on purpose: a canned "on page 3" contradicted a go_to_page(5)
    // and the model retried the call (Mac, 2026-08-19) — the fake-results
    // recipe again.
    RecordingTool(base: GoToPageTool(), canned: "on that page now"),
    RecordingTool(base: DeletePageTool(), canned: "deleted the page; 5 pages left"),
    RecordingTool(base: MovePageTool(), canned: "moved the page"),
    RecordingTool(base: RotatePageTool(), canned: "rotated the page"),
    RecordingTool(base: InsertBlankPageTool(), canned: "inserted a blank page; 7 pages now"),
    RecordingTool(base: HighlightTextTool(), canned: "highlighted 4 occurrences of \"deposit\" on pages 3, 4"),
    RecordingTool(base: RemoveHighlightsTool(), canned: "removed 4 highlights"),
    RecordingTool(base: AddNoteTool(), canned: "note added to the open page"),
    RecordingTool(base: SignPageTool(), canned: "signed the page"),
    RecordingTool(base: WatermarkTool(), canned: "\"DRAFT\" watermarked across all 6 pages"),
    RecordingTool(
      base: ExtractPagesTool(),
      canned: "extracted pages 3–4 to saved-pages-3-4.pdf in the app's Documents (2 pages)"),
    RecordingTool(base: SearchDocumentTool(), canned: "\"deposit\" appears 4 times: page 3 (3×, Rent and Deposit); page 4 (1×, Term)"),
    RecordingTool(base: SavePDFTool(), canned: "saved as saved-lease.pdf in the app's Documents (6 pages)"),
    RecordingTool(base: RevertDocumentTool(), canned: "back to the original document — 6 pages, no annotations"),
    RecordingTool(
      base: FillFieldTool(), canned: "",
      respond: { args in "\"\(args.value)\" entered in the \(args.field) field" }),
    RecordingTool(
      base: RedactTextTool(), canned: "",
      respond: { args in "blacked out every occurrence of \"\(args.text)\" — they can no longer be read" }),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .docs), canned: "undid the last change"),
  ]

  /// The shopping pack, neutralized, against the state the cases carry
  /// (five earbuds results, one cart line).
  static let shopping: [any FoundationModels.Tool] = [
    RecordingTool(base: SearchCatalogTool(), canned: "the results are numbered and listed on screen"),
    RecordingTool(base: SortResultsTool(), canned: "sorted — the renumbered results are on screen"),
    RecordingTool(base: ShowProductTool(), canned: "that product's details are on screen"),
    RecordingTool(base: AddToCartTool(), canned: "added to the cart — cart total ¥9,980"),
    RecordingTool(base: ChangeQuantityTool(), canned: "quantity changed — cart total ¥4,990"),
    RecordingTool(base: RemoveFromCartTool(), canned: "removed from the cart"),
    RecordingTool(base: ApplyCouponTool(), canned: "coupon applied: −10% — total ¥4,490"),
    RecordingTool(
      base: CheckoutTool(), canned: "",
      respond: { args in
        args.confirm
          ? "order #5231 placed — arriving in 2 days"
          : "placing this order charges the cart total — ask the user to confirm, then call checkout again with confirm true"
      }),
    RecordingTool(base: FilterResultsTool(), canned: "the narrowed results are renumbered and listed on screen"),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .shopping), canned: "undid the last change"),
    RecordingTool(base: TrackOrderTool(), canned: "order #5230 is out for delivery — arriving today by 21:00"),
  ]

  /// The money pack against the month of canned spending — finders echo
  /// dynamically (see the store pack's note). The rows carry the total, so
  /// "how much at Maruetsu?" is answered by the finder's own result instead
  /// of by a spurious follow-up report call.
  static let money: [any FoundationModels.Tool] = [
    RecordingTool(base: ListTransactionsTool(), canned: "", respond: { MoneyEcho.list(days: $0.days) }),
    RecordingTool(base: FilterTransactionsTool(), canned: "", respond: { MoneyEcho.filter($0.category) }),
    RecordingTool(base: SearchPayeeTool(), canned: "", respond: { MoneyEcho.search(payee: $0.payee) }),
    RecordingTool(base: CategorizeTool(), canned: "", respond: { _ in
      BenchSelection.shared.kind != nil
        ? "categorized the selected transactions"
        : "nothing is selected — list, search or filter first" }),
    RecordingTool(base: FlagTransactionsTool(), canned: "", respond: { _ in
      BenchSelection.shared.kind != nil
        ? "flagged the selected transactions"
        : "nothing is selected — list, search or filter first" }),
    RecordingTool(base: SetBudgetTool(), canned: "budget set"),
    RecordingTool(base: SpendingReportTool(), canned: "last 7 days: ¥21,340 across 7 transactions — groceries ¥4,820, eating_out ¥1,180"),
    RecordingTool(base: BudgetReportTool(), canned: "this month against budgets: eating_out over by ¥2,460; groceries ¥13,810 left"),
    RecordingTool(base: FindSubscriptionsTool(), canned: "2 recurring payments, about ¥2,470 a month: Netflix ¥1,490; Spotify ¥980"),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .money), canned: "undid the last change"),
  ]

  /// The CRM pack against the frozen quarter. Finders and gets echo
  /// dynamically — the pack started on static neutral fakes and r11
  /// measured the starvation the recipe predicts: every Japanese finder
  /// grew a get_opportunity tail, one English search flailed through five
  /// queries the neutral line never answered, and the final answers
  /// confabulated rows (¥1,500,000 deals that do not exist). Actions stay
  /// neutral; nothing here mutates.
  static let crm: [any FoundationModels.Tool] = [
    RecordingTool(
      base: SearchOpportunitiesTool(), canned: "",
      respond: {
        CrmEcho.opportunities(
          company: $0.company, owner: $0.owner, stage: $0.stage,
          minAmount: $0.min_amount, maxAmount: $0.max_amount,
          closeFrom: $0.close_date_from, closeTo: $0.close_date_to)
      }),
    RecordingTool(base: GetOpportunityTool(), canned: "", respond: { CrmEcho.opportunity($0.id) }),
    RecordingTool(base: UpdateOpportunityStageTool(), canned: "stage changed — the pipeline is on screen"),
    RecordingTool(base: UpdateOpportunityAmountTool(), canned: "amount changed — the pipeline is on screen"),
    RecordingTool(base: AssignOpportunityTool(), canned: "owner changed — the pipeline is on screen"),
    RecordingTool(
      base: CrmSearchContactsTool(), canned: "",
      respond: { CrmEcho.contacts(name: $0.name, company: $0.company, email: $0.email) }),
    RecordingTool(base: CrmGetContactTool(), canned: "", respond: { CrmEcho.contact($0.id) }),
    RecordingTool(
      base: CrmSearchCompaniesTool(), canned: "",
      respond: { CrmEcho.companies(name: $0.name, industry: $0.industry, location: $0.location) }),
    RecordingTool(base: CreateFollowUpTaskTool(), canned: "follow-up task added"),
    RecordingTool(base: CrmAddNoteTool(), canned: "note added"),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .crm), canned: "undid the last change"),
  ]

  /// The PM pack against the frozen sprint. Echoes from day one — the
  /// starvation neutral fakes cause was re-measured on the CRM pack the
  /// same week (get-tails on every Japanese finder, confabulated rows);
  /// that lesson is not bought again. Actions stay neutral; nothing here
  /// mutates.
  static let pm: [any FoundationModels.Tool] = [
    RecordingTool(
      base: SearchIssuesTool(), canned: "",
      respond: {
        PmEcho.issues(
          project: $0.project, status: $0.status, priority: $0.priority,
          assignee: $0.assignee, creator: $0.creator, keyword: $0.keyword,
          dueFrom: $0.due_from, dueTo: $0.due_by)
      }),
    RecordingTool(base: GetIssueTool(), canned: "", respond: { PmEcho.issue($0.id) }),
    RecordingTool(base: CreateIssueTool(), canned: "issue created — it is on the board"),
    RecordingTool(base: AssignIssueTool(), canned: "assignee changed — the board is on screen"),
    RecordingTool(base: ChangeIssueStatusTool(), canned: "status changed — the board is on screen"),
    RecordingTool(base: ChangeIssuePriorityTool(), canned: "priority changed — the board is on screen"),
    RecordingTool(base: ChangeDueDateTool(), canned: "due date changed — the board is on screen"),
    RecordingTool(base: AddCommentTool(), canned: "comment added"),
    RecordingTool(base: CloseIssueTool(), canned: "closed — it left the open board"),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .pm), canned: "undid the last change"),
  ]

  /// The inbox pack against the fifteen canned messages — finders echo
  /// dynamically: a pure render over the same frozen world the real tools
  /// compute over, arguments reflected. The neutral fakes ("the matching
  /// messages are selected and listed on screen") starved the model — it
  /// piled a read_message onto "show me", stalled mid-chain narrating what
  /// it would do next, and had no numbers to act on. Reflecting the real
  /// arguments keeps the fake-well recipe's rule (the echo can never
  /// contradict the call) while giving the model the rows the real app
  /// would show. Actions stay neutral; nothing here mutates.
  static let inbox: [any FoundationModels.Tool] = [
    RecordingTool(base: ListInboxTool(), canned: "", respond: { InboxEcho.list($0.filter) }),
    RecordingTool(base: SearchMailTool(), canned: "", respond: { InboxEcho.search($0.query) }),
    RecordingTool(
      base: ReadMessageTool(), canned: "",
      respond: { args in
        guard let message = InboxEcho.message(args.number) else {
          return "there is no message #\(args.number)"
        }
        return "#\(message.id) from \(message.from), \(MoneyBox.when(message.daysAgo)) — \"\(message.subject)\": \(message.snippet)"
      }),
    RecordingTool(base: ArchiveTool(), canned: "", respond: { _ in
      InboxFake.onSelection("archived the selected messages") }),
    RecordingTool(base: MarkReadTool(), canned: "", respond: { _ in
      InboxFake.onSelection("marked the selected messages read") }),
    RecordingTool(base: FlagMailTool(), canned: "", respond: { _ in
      InboxFake.onSelection("flagged the selected messages") }),
    RecordingTool(base: SnoozeTool(), canned: "", respond: { _ in
      InboxFake.onSelection("snoozed the selected messages") }),
    RecordingTool(
      base: DraftReplyTool(), canned: "",
      respond: { args in
        guard let message = InboxEcho.message(args.number) else {
          return "there is no message #\(args.number)"
        }
        return "draft saved, replying to \(message.from) re \"\(message.subject)\" — nothing sent"
      }),
    RecordingTool(
      base: UnsubscribeTool(), canned: "",
      respond: { args in
        guard let message = InboxEcho.message(args.number) else {
          return "there is no message #\(args.number)"
        }
        return message.newsletter
          ? "unsubscribed from \(message.from) and archived the message"
          : "#\(message.id) from \(message.from) is not a newsletter — archived it instead"
      }),
    RecordingTool(
      base: DeleteMessageTool(), canned: "",
      respond: { args in
        args.confirm
          ? "deleted message #\(args.number)"
          : "not deleted — deleting message #\(args.number) is permanent. Ask the user to confirm and stop; only their yes in a later message allows confirm true"
      }),
    AskUserTool(),
    RecordingTool(base: UndoLastTool(target: .inbox), canned: "undid the last change"),
  ]

  /// The honest-refusal halves of the bulk fakes, in each app's own words.
  enum StoreFake {
    static func onProducts(_ success: String) -> String {
      BenchSelection.shared.kind == "products"
        ? success
        : "nothing is selected — search or filter products first, then act on them"
    }
    static func onOrders(_ success: String) -> String {
      BenchSelection.shared.kind == "orders"
        ? success
        : "no orders are selected — filter orders first, then act on them"
    }
  }

  enum InboxFake {
    static func onSelection(_ success: String) -> String {
      BenchSelection.shared.kind != nil
        ? success
        : "nothing is selected — list, search or filter first"
    }
  }

  /// Pure renders of the money finders over the frozen month — the same
  /// filter, search and row format as MoneyBox (total included), selection
  /// state left out.
  enum MoneyEcho {
    private static var rows: [MoneyBox.Transaction] {
      MoneyData.transactions.sorted { $0.daysAgo < $1.daysAgo }
    }

    private static func render(_ matched: [MoneyBox.Transaction], how: String) -> String {
      BenchSelection.shared.kind = matched.isEmpty ? nil : "rows"
      guard !matched.isEmpty else { return "no transactions match \(how)" }
      let listed = matched.prefix(8).map {
        "\(MoneyBox.when($0.daysAgo)) \($0.payee) — \(StoreBox.yen($0.amount))\($0.category.map { ", \($0)" } ?? ", uncategorized")"
      }
      let total = matched.reduce(0) { $0 + $1.amount }
      return "\(matched.count) transaction\(matched.count == 1 ? "" : "s") (\(how)), \(StoreBox.yen(total)) in all — now the selection:\n"
        + listed.joined(separator: "\n") + (matched.count > 8 ? "\n… and \(matched.count - 8) more" : "")
    }

    static func list(days: Int) -> String {
      let span = max(1, days)
      return render(rows.filter { $0.daysAgo < span }, how: "last \(span) day\(span == 1 ? "" : "s")")
    }

    static func filter(_ category: String) -> String {
      let want = category.lowercased()
      if want == "uncategorized" || want == "none" {
        return render(rows.filter { $0.category == nil }, how: "uncategorized")
      }
      return render(rows.filter { $0.category == want }, how: "category \(want)")
    }

    static func search(payee: String) -> String {
      let needle = MoneyData.romaji(payee)
      return render(rows.filter { $0.payee.lowercased().contains(needle) }, how: "payee \"\(payee)\"")
    }
  }

  /// Pure renders of the store finders over the frozen products and orders —
  /// the same matching and row format as StoreBox, selection state left out.
  enum StoreEcho {
    private static func renderProducts(_ matched: [StoreBox.Product], how: String) -> String {
      BenchSelection.shared.kind = matched.isEmpty ? nil : "products"
      guard !matched.isEmpty else { return "no products match \(how)" }
      let listed = matched.prefix(8).map {
        "\($0.title) — \(StoreBox.yen($0.price)), stock \($0.stock), \($0.status)"
      }
      return "\(matched.count) product\(matched.count == 1 ? "" : "s") selected (\(how)):\n"
        + listed.joined(separator: "\n") + (matched.count > 8 ? "\n… and \(matched.count - 8) more" : "")
    }

    private static func renderOrders(_ matched: [StoreBox.Order], how: String) -> String {
      BenchSelection.shared.kind = matched.isEmpty ? nil : "orders"
      guard !matched.isEmpty else { return "no orders match \(how)" }
      let listed = matched.prefix(8).map {
        "#\($0.number) \($0.customer) — \(StoreBox.yen($0.total)), \($0.payment), \($0.fulfillment), \($0.daysAgo == 0 ? "today" : "\($0.daysAgo) d ago")"
      }
      return "\(matched.count) order\(matched.count == 1 ? "" : "s") selected (\(how)):\n"
        + listed.joined(separator: "\n") + (matched.count > 8 ? "\n… and \(matched.count - 8) more" : "")
    }

    static func search(_ query: String) -> String {
      let needle = StoreData.romaji(query)
      let words = needle.split(separator: " ").map(String.init)
      let matched = StoreData.products.filter { p in
        needle.isEmpty || words.allSatisfy { p.title.lowercased().contains($0) }
      }
      return renderProducts(matched, how: needle.isEmpty ? "all products" : "search \"\(query)\"")
    }

    static func product(_ name: String) -> String {
      let needle = StoreData.romaji(name)
      let words = needle.split(separator: " ").map(String.init)
      let matched = StoreData.products.filter { p in
        !needle.isEmpty && words.allSatisfy { p.title.lowercased().contains($0) }
      }
      guard !matched.isEmpty else { return "no product matches \"\(name)\"" }
      guard matched.count == 1 else {
        return "\(matched.count) products match \"\(name)\": " + matched.map(\.title).joined(separator: "; ")
      }
      let p = matched[0]
      return "\(p.title) — \(StoreBox.yen(p.price)), stock \(p.stock), \(p.status); vendor \(p.vendor), type \(p.type)"
        + (p.tags.isEmpty ? "" : ", tags: \(p.tags.joined(separator: ", "))")
    }

    static func customers(name: String?, email: String?, spentAbove: Int?) -> String {
      struct Customer {
        var total = 0
        var count = 0
      }
      var byName: [String: Customer] = [:]
      for order in StoreData.orders where order.payment != "refunded" {
        byName[order.customer, default: Customer()].total += order.total
        byName[order.customer]!.count += 1
      }
      var rows = byName.map { (name: $0.key, email: $0.key.lowercased().replacingOccurrences(of: " ", with: ".") + "@example.com", total: $0.value.total, count: $0.value.count) }
      var how: [String] = []
      if let name, !name.isEmpty {
        let needle = StoreData.romaji(name)
        rows = rows.filter { $0.name.lowercased().contains(needle) }
        how.append("name \"\(name)\"")
      }
      if let email, !email.isEmpty {
        let needle = email.lowercased()
        rows = rows.filter { $0.email.contains(needle) }
        how.append("email \"\(email)\"")
      }
      if let spentAbove {
        rows = rows.filter { $0.total >= spentAbove }
        how.append("spent \(StoreBox.yen(spentAbove))+")
      }
      let caption = how.isEmpty ? "all customers" : how.joined(separator: ", ")
      let sorted = rows.sorted { $0.total > $1.total }
      guard !sorted.isEmpty else { return "no customers match \(caption)" }
      return "\(sorted.count) customer\(sorted.count == 1 ? "" : "s") (\(caption)):\n"
        + sorted.prefix(8).map { "\($0.name) (\($0.email)) — \(StoreBox.yen($0.total)) across \($0.count) order\($0.count == 1 ? "" : "s")" }
          .joined(separator: "\n")
        + (sorted.count > 8 ? "\n… and \(sorted.count - 8) more" : "")
    }

    static func filter(by field: String, value: String) -> String {
      let want = value.lowercased().trimmingCharacters(in: .whitespaces)
      let matched = StoreData.products.filter { p in
        switch field.lowercased() {
        case "status": return p.status == want
        case "vendor": return p.vendor.lowercased() == want || p.vendor.lowercased().contains(want)
        case "tag": return p.tags.contains { $0.lowercased() == want }
        case "product_type", "type": return p.type.lowercased() == want || p.type.lowercased().contains(want)
        default: return false
        }
      }
      return renderProducts(matched, how: "\(field.lowercased()) = \(value)")
    }

    static func lowStock(below: Int) -> String {
      renderProducts(
        StoreData.products.filter { $0.stock < below && $0.status != "archived" }
          .sorted { $0.stock < $1.stock },
        how: "stock below \(below)")
    }

    static func searchOrders(customer: String) -> String {
      let needle = customer.lowercased().trimmingCharacters(in: .whitespaces)
      return renderOrders(
        StoreData.orders.filter { $0.customer.lowercased().contains(needle) }
          .sorted { $0.daysAgo < $1.daysAgo },
        how: "customer \"\(customer)\"")
    }

    static func filterOrders(payment: String, fulfillment: String) -> String {
      let p = payment.lowercased()
      let f = fulfillment.lowercased()
      let matched = StoreData.orders.filter { o in
        (p == "any" || o.payment == p) && (f == "any" || o.fulfillment == f)
      }.sorted { $0.daysAgo < $1.daysAgo }
      var parts: [String] = []
      if p != "any" { parts.append("payment \(p)") }
      if f != "any" { parts.append(f) }
      return renderOrders(matched, how: parts.isEmpty ? "all orders" : parts.joined(separator: ", "))
    }
  }

  /// Pure renders of the PM finder and get over the frozen sprint — the
  /// same filters, kana normalization and row format as IssueBox
  /// (IssueBox.oneLine is shared), selection state left out.
  enum PmEcho {
    static func issues(
      project: String?, status: String?, priority: String?, assignee: String?,
      creator: String?, keyword: String?, dueFrom: String?, dueTo: String?
    ) -> String {
      var how: [String] = []
      var rows = PmData.issues
      if let project, !project.isEmpty {
        let want = project.lowercased()
        guard let canonical = PmData.projects.first(where: { $0.lowercased() == want }) else {
          return "no project called \(project) — the projects are \(PmData.projects.joined(separator: ", "))"
        }
        rows = rows.filter { $0.project == canonical }
        how.append("project \(canonical)")
      }
      if let status, !status.isEmpty {
        let want = status.lowercased()
        guard PmData.statuses.contains(want) else {
          return "unknown status; the statuses are \(PmData.statuses.joined(separator: ", "))"
        }
        rows = rows.filter { $0.status == want }
        how.append("status \(want)")
      }
      if let priority, !priority.isEmpty {
        let want = priority.uppercased()
        guard PmData.priorities.contains(want) else {
          return "unknown priority; the priorities are \(PmData.priorities.joined(separator: ", "))"
        }
        rows = rows.filter { $0.priority == want }
        how.append("priority \(want)")
      }
      if let assignee, !assignee.isEmpty {
        let want = assignee.lowercased()
        if want == "unassigned" || want == "none" || want == "nobody" {
          rows = rows.filter { $0.assignee == nil }
          how.append("unassigned")
        } else {
          guard let canonical = PmData.person(assignee) else {
            return "nobody called \(assignee) here — the assignees are \(PmData.assignees.joined(separator: ", "))"
          }
          rows = rows.filter { $0.assignee == canonical }
          how.append("assignee \(canonical)")
        }
      }
      if let creator, !creator.isEmpty {
        guard let canonical = PmData.person(creator) else { return "nobody called \(creator) here" }
        rows = rows.filter { $0.creator == canonical }
        how.append("creator \(canonical)")
      }
      if let keyword, !keyword.isEmpty {
        let needle = PmData.romaji(keyword)
        rows = rows.filter { $0.title.lowercased().contains(needle) }
        how.append("keyword \"\(keyword)\"")
      }
      if let dueFrom, !dueFrom.isEmpty {
        rows = rows.filter { $0.due >= dueFrom }
        how.append("due from \(dueFrom)")
      }
      if let dueTo, !dueTo.isEmpty {
        rows = rows.filter { $0.due <= dueTo }
        how.append("due by \(dueTo)")
      }
      let sorted = rows.sorted { $0.due < $1.due }
      let caption = how.isEmpty ? "all issues" : how.joined(separator: ", ")
      BenchSelection.shared.kind = sorted.isEmpty ? nil : "rows"
      guard !sorted.isEmpty else { return "no issues match \(caption)" }
      return "\(sorted.count) issue\(sorted.count == 1 ? "" : "s") (\(caption)) — now the selection:\n"
        + sorted.prefix(8).map(IssueBox.oneLine).joined(separator: "\n")
        + (sorted.count > 8 ? "\n… and \(sorted.count - 8) more" : "")
    }

    static func issue(_ id: String) -> String {
      let want = id.uppercased()
      guard let issue = PmData.issues.first(where: { $0.id == want }) else {
        return "there is no issue \(id) — ids look like APP-3"
      }
      return IssueBox.oneLine(issue) + "\ncreated by \(issue.creator)"
    }
  }

  /// Pure renders of the CRM finders and gets over the frozen quarter —
  /// the same filters, kana normalization and row format as CrmBox
  /// (CrmBox.oneLine is shared so the formats cannot drift), selection
  /// state left out.
  enum CrmEcho {
    private static func render(_ matched: [CrmBox.Opportunity], how: String) -> String {
      BenchSelection.shared.kind = matched.isEmpty ? nil : "rows"
      guard !matched.isEmpty else { return "no opportunities match \(how)" }
      let total = matched.reduce(0) { $0 + $1.amount }
      let word = matched.count == 1 ? "opportunity" : "opportunities"
      return "\(matched.count) \(word) (\(how)), \(StoreBox.yen(total)) in all — now the selection:\n"
        + matched.prefix(8).map(CrmBox.oneLine).joined(separator: "\n")
        + (matched.count > 8 ? "\n… and \(matched.count - 8) more" : "")
    }

    static func opportunities(
      company: String?, owner: String?, stage: String?,
      minAmount: Int?, maxAmount: Int?, closeFrom: String?, closeTo: String?
    ) -> String {
      var how: [String] = []
      var rows = CrmData.opportunities
      if let company, !company.isEmpty {
        let needle = CrmData.romaji(company)
        rows = rows.filter { $0.company.lowercased().contains(needle) }
        how.append("company \"\(company)\"")
      }
      if let owner, !owner.isEmpty {
        guard let canonical = CrmData.owner(owner) else {
          return "no owner called \(owner) — the owners are \(CrmData.owners.joined(separator: ", "))"
        }
        rows = rows.filter { $0.owner == canonical }
        how.append("owner \(canonical)")
      }
      if let stage, !stage.isEmpty {
        let want = stage.lowercased()
        guard CrmBox.stages.contains(want) else {
          return "unknown stage; the stages are \(CrmBox.stages.joined(separator: ", "))"
        }
        rows = rows.filter { $0.stage == want }
        how.append("stage \(want)")
      }
      if let minAmount {
        rows = rows.filter { $0.amount >= minAmount }
        how.append("\(StoreBox.yen(minAmount))+")
      }
      if let maxAmount {
        rows = rows.filter { $0.amount <= maxAmount }
        how.append("up to \(StoreBox.yen(maxAmount))")
      }
      if let closeFrom, !closeFrom.isEmpty {
        rows = rows.filter { $0.closeDate >= closeFrom }
        how.append("closing from \(closeFrom)")
      }
      if let closeTo, !closeTo.isEmpty {
        rows = rows.filter { $0.closeDate <= closeTo }
        how.append("closing by \(closeTo)")
      }
      return render(
        rows.sorted { $0.closeDate < $1.closeDate },
        how: how.isEmpty ? "all opportunities" : how.joined(separator: ", "))
    }

    static func opportunity(_ id: String) -> String {
      let want = id.uppercased()
      guard let o = CrmData.opportunities.first(where: { $0.id == want }) else {
        return "there is no opportunity \(id) — ids look like O3"
      }
      var text = CrmBox.oneLine(o)
      if let c = CrmData.contacts.first(where: { $0.company == o.company }) {
        text += "\ncontact: \(c.name) (\(c.role), \(c.email))"
      }
      let tied = CrmData.tasks.filter { $0.about == want }
      if !tied.isEmpty {
        text += "\ntasks: " + tied.map { "\($0.id) \"\($0.title)\" due \($0.due)" }.joined(separator: "; ")
      }
      return text
    }

    static func contacts(name: String?, company: String?, email: String?) -> String {
      var how: [String] = []
      var rows = CrmData.contacts
      if let name, !name.isEmpty {
        let needle = CrmData.romaji(name)
        rows = rows.filter { $0.name.lowercased().contains(needle) }
        how.append("name \"\(name)\"")
      }
      if let company, !company.isEmpty {
        let needle = CrmData.romaji(company)
        rows = rows.filter { $0.company.lowercased().contains(needle) }
        how.append("company \"\(company)\"")
      }
      if let email, !email.isEmpty {
        let needle = email.lowercased()
        rows = rows.filter { $0.email.lowercased().contains(needle) }
        how.append("email \"\(email)\"")
      }
      let caption = how.isEmpty ? "all contacts" : how.joined(separator: ", ")
      BenchSelection.shared.kind = rows.isEmpty ? nil : "rows"
      guard !rows.isEmpty else { return "no contacts match \(caption)" }
      return "\(rows.count) contact\(rows.count == 1 ? "" : "s") (\(caption)) — now the selection:\n"
        + rows.prefix(8).map { "\($0.id) \($0.name) — \($0.company), \($0.role), \($0.email)" }.joined(separator: "\n")
    }

    static func contact(_ id: String) -> String {
      let want = id.uppercased()
      guard let c = CrmData.contacts.first(where: { $0.id == want }) else {
        return "there is no contact \(id) — ids look like C2"
      }
      var text = "\(c.id) \(c.name) — \(c.company), \(c.role), \(c.email)"
      let deals = CrmData.opportunities.filter { $0.company == c.company && $0.stage != "won" && $0.stage != "lost" }
      if !deals.isEmpty { text += "\nopen deals: " + deals.map(CrmBox.oneLine).joined(separator: "; ") }
      return text
    }

    static func companies(name: String?, industry: String?, location: String?) -> String {
      var how: [String] = []
      var rows = CrmData.companies
      if let name, !name.isEmpty {
        let needle = CrmData.romaji(name)
        rows = rows.filter { $0.name.lowercased().contains(needle) }
        how.append("name \"\(name)\"")
      }
      if let industry, !industry.isEmpty {
        let needle = CrmData.romaji(industry)
        rows = rows.filter { $0.industry.lowercased().contains(needle) }
        how.append("industry \"\(industry)\"")
      }
      if let location, !location.isEmpty {
        let needle = CrmData.romaji(location)
        rows = rows.filter { $0.location.lowercased().contains(needle) }
        how.append("location \"\(location)\"")
      }
      let caption = how.isEmpty ? "all companies" : how.joined(separator: ", ")
      BenchSelection.shared.kind = rows.isEmpty ? nil : "rows"
      guard !rows.isEmpty else { return "no companies match \(caption)" }
      return "\(rows.count) compan\(rows.count == 1 ? "y" : "ies") (\(caption)) — now the selection:\n"
        + rows.map { "\($0.name) — \($0.industry), \($0.location)" }.joined(separator: "\n")
    }
  }

  /// Pure renders of the inbox finders over the frozen fifteen messages —
  /// the same filter, search and row format as InboxBox, with the selection
  /// and the undo history left out. Deterministic across cases by
  /// construction: the data never moves.
  enum InboxEcho {
    private static var live: [InboxBox.Message] {
      InboxData.messages.sorted { $0.daysAgo < $1.daysAgo }
    }

    static func message(_ id: Int) -> InboxBox.Message? {
      InboxData.messages.first { $0.id == id }
    }

    private static func render(_ matched: [InboxBox.Message], how: String) -> String {
      BenchSelection.shared.kind = matched.isEmpty ? nil : "rows"
      guard !matched.isEmpty else { return "no messages match \(how)" }
      let listed = matched.prefix(8).map {
        "#\($0.id) \($0.from) — \"\($0.subject)\" (\(MoneyBox.when($0.daysAgo))\($0.unread ? ", unread" : ""))"
      }
      return "\(matched.count) message\(matched.count == 1 ? "" : "s") (\(how)) — now the selection:\n"
        + listed.joined(separator: "\n")
        + (matched.count > 8 ? "\n… and \(matched.count - 8) more" : "")
    }

    static func list(_ filter: String) -> String {
      switch filter.lowercased() {
      case "unread": return render(live.filter(\.unread), how: "unread")
      case "flagged": return render(live.filter(\.flagged), how: "flagged")
      case "newsletters": return render(live.filter(\.newsletter), how: "newsletters")
      case "read": return render(live.filter { !$0.unread }, how: "already read")
      default: return render(live, how: "all")
      }
    }

    static func search(_ query: String) -> String {
      let needle = query.lowercased()
      return render(
        live.filter {
          $0.from.lowercased().contains(needle) || $0.subject.lowercased().contains(needle)
            || $0.snippet.lowercased().contains(needle)
        },
        how: "search \"\(query)\"")
    }
  }

  /// The vision pack, and a deliberate break with the all-canned rule
  /// above: the goal-driven loop re-attaches the edited photo and asks the
  /// model to judge its own work, so the edits must actually land on
  /// pixels — a canned "done" would loop the model over an unchanged
  /// picture and measure nothing. Only the tools that leave the app are
  /// faked: save (a photo-library consent dialog would hang a shell-driven
  /// run) and the note.
  static let vision: [any FoundationModels.Tool] = [
    SeenBrightnessTool(), SeenExposureTool(), SeenContrastTool(), SeenSaturationTool(),
    SeenWarmthTool(), SeenRotateTool(), SeenCropTool(), SeenFilterTool(),
    SeenAutoEnhanceTool(), SeenRemoveBackgroundTool(), SeenReadTextTool(), SeenRedactTool(),
    SeenRevertTool(),
    RecordingTool(base: SeenSaveTool(), canned: "saved to the library"),
    RecordingTool(base: WriteNoteTool(), canned: "noted"),
  ]

  /// The business wing merged: CRM + PM + Commerce in one list, the rails
  /// deduplicated by name (three packs each carry ask_user and undo_last;
  /// a session cannot hold two tools with one name). 41 tools — the
  /// cross-domain instrument: 「昨日の未発送注文」 must find the store's
  /// order filter with the CRM and the board in the room.
  static let business: [any FoundationModels.Tool] = {
    var seen = Set<String>()
    var merged: [any FoundationModels.Tool] = []
    for tool in crm + pm + store where !seen.contains(tool.name) {
      seen.insert(tool.name)
      merged.append(tool)
    }
    return merged
  }()

  /// The moment index's canned world: one 600 s soccer-match recording,
  /// frozen. Three indexes over the same match — what is seen, what is
  /// said, what is written on screen — so a clause's right index is a
  /// scorable choice. Keywords carry the JA the cases can utter (recipes:
  /// canned data must be findable in every language the pack tests). No
  /// bare "card"/"カード" key on the yellow-card row on purpose: a red-card
  /// query must come back empty, not find the wrong booking.
  enum MomentEcho {
    struct Row {
      let start: Double
      let end: Double
      let text: String
      let keys: [String]
    }

    static let frames: [Row] = [
      Row(start: 3, end: 10, text: "kickoff", keys: ["kickoff", "kick-off", "キックオフ", "開始"]),
      Row(
        start: 130, end: 136, text: "the keeper's diving save",
        keys: ["save", "dive", "keeper", "セーブ", "キーパー"]),
      Row(
        start: 158, end: 163, text: "a yellow card is shown",
        keys: ["yellow", "booking", "イエロー", "黄色"]),
      Row(
        start: 214, end: 226, text: "a goal — a header — and the celebration",
        keys: ["goal", "header", "celebrat", "ゴール", "ヘディング", "得点"]),
      Row(start: 320, end: 330, text: "heavy rain starts falling", keys: ["rain", "雨"]),
      Row(
        start: 380, end: 390, text: "a substitution at the touchline",
        keys: ["substitut", "交代"]),
      Row(
        start: 440, end: 462, text: "a penalty — awarded, then converted: a goal",
        keys: ["penalty", "spot", "ペナルティ", "PK", "goal", "ゴール", "得点"]),
      Row(
        start: 592, end: 600, text: "the final whistle and the celebrations",
        keys: ["whistle", "final", "終了", "ホイッスル", "笛"]),
    ]

    static let transcript: [Row] = [
      Row(
        start: 131.5, end: 134, text: "\"What an absolutely incredible save!\"",
        keys: ["save", "incredible", "セーブ"]),
      Row(
        start: 213, end: 216, text: "\"He rises highest — and it's in! What a header!\"",
        keys: ["header", "it's in", "ヘディング"]),
      Row(
        start: 299, end: 301, text: "\"And that's half time.\"",
        keys: ["half time", "half-time", "ハーフタイム"]),
      Row(
        start: 440.5, end: 443, text: "\"The referee points to the spot — it's a penalty!\"",
        keys: ["penalty", "referee", "spot", "ペナルティ", "PK"]),
      Row(
        start: 457, end: 459, text: "\"He sends the keeper the wrong way — two nil!\"",
        keys: ["two nil", "two-nil", "2-0", "2対0"]),
      Row(start: 575, end: 577, text: "\"Surely that's the win now.\"", keys: ["win", "勝"]),
    ]

    static let screenText: [Row] = [
      Row(start: 0, end: 6, text: "scoreboard \"BLU 0-0 RED\"", keys: ["0-0", "0対0"]),
      Row(
        start: 218, end: 224, text: "the scoreboard changes to \"BLU 1-0 RED\"",
        keys: ["1-0", "scoreboard", "スコア", "1対0"]),
      Row(
        start: 458, end: 464, text: "the scoreboard changes to \"BLU 2-0 RED\"",
        keys: ["2-0", "2対0"]),
      Row(
        start: 520, end: 526, text: "banner \"ATTENDANCE 48,113\"",
        keys: ["attendance", "観客"]),
      Row(
        start: 597, end: 600, text: "\"FULL TIME  BLU 2-0 RED\"",
        keys: ["full time", "full-time", "フルタイム"]),
    ]

    /// A row matches when any of its keys appears in the query — multiword
    /// queries carry the keyword; a bare keyword query is its own key.
    static func search(_ index: [Row], in what: String, query: String) -> String {
      let q = query.lowercased()
      let hits = index.filter { row in row.keys.contains { q.contains($0.lowercased()) } }
      guard !hits.isEmpty else { return "no moments found for \"\(query)\" in \(what)" }
      let lines = hits.map {
        "\(VideoEditBox.f($0.start))–\(VideoEditBox.f($0.end)) s — \($0.text)"
      }
      // "found" leads, exactly as the real index answers: the model's final
      // answer follows the strongest verdict word among recent results, and
      // a bare count loses to a later "no moments found" (demo-playbook:
      // answers follow the verdict word). The bench must ask the same
      // question the app does.
      return "found \(hits.count) moment\(hits.count == 1 ? "" : "s"):\n"
        + lines.joined(separator: "\n")
    }

    /// The forced-choice check, canned: the ground truth at one moment,
    /// answered by picking from the options given — the stand-in for the
    /// per-candidate VLM look. The truth is the score at that time plus
    /// whatever rows cover it.
    ///
    /// The matcher is the real check's, transplanted onto the canned truths
    /// (VideoEditTools.MomentIndexBox.check — keep the two readable side by
    /// side): negation partition first, then a direct match on the positive
    /// options, then presence decided by the content words of the question
    /// and the positive options, and the verdict word always leads with the
    /// evidence tail behind it for humans and logs. The bench must not be
    /// kinder than the app — r33–36 were scored against a stricter canned
    /// check that refused anything it could not match by name.
    static func check(seconds: Double, question: String, options: [String]) -> String {
      let score = seconds < 218 ? "0-0" : (seconds < 458 ? "1-0" : "2-0")
      var truths = [score, score.replacingOccurrences(of: "-", with: "対")]
      for row in frames where seconds >= row.start && seconds <= row.end {
        truths.append(row.text)
        truths.append(contentsOf: row.keys)
      }
      for row in screenText where seconds >= row.start && seconds <= row.end {
        truths.append(row.text)
        truths.append(contentsOf: row.keys)
      }
      // The real check collects its truths already lowercased; the canned
      // rows keep their casing for the evidence tail, so match on a
      // lowercased mirror and print the rows as written.
      let lower = truths.map { $0.lowercased() }
      let shows = " — around \(VideoEditBox.f(seconds)) s the frame shows: "

      func negated(_ option: String) -> Bool {
        // JA carries the negation inside the word and writes no spaces, so
        // these match as bare substrings. Without them a JA option pair is
        // all-positive: a miss falls through to "none of those" and the
        // verdict word vetoes a correct retrieval — the findability rule
        // (playbook spec D) applies to the verdict, not only to the index.
        // The second row is spec D2's: 「いいえ」 is the no half of the
        // plainest pair a model can offer, and without it はい/いいえ reads
        // all-positive (measured, r38 m-ja-check-2). 「ません」 is the polite
        // negation ありません/いません are two instances of, so it catches
        // 写っていません too; 無い/無し are the kanji spellings.
        for marker in [
          "なし", "ない", "いない", "ありません", "いません",
          "いいえ", "ません", "無い", "無し",
        ] where option.contains(marker) { return true }
        let o = " " + option.lowercased() + " "
        return o.contains(" no ") || o.contains(" not ") || o.contains("n't ")
          || o.contains(" none ") || o.contains(" without ") || o.contains(" nothing ")
      }
      let positives = options.filter { !negated($0) }
      let negatives = options.filter { negated($0) }
      if let hit = positives.first(where: { option in
        let o = option.lowercased()
        return lower.contains { $0.contains(o) || o.contains($0) }
      }) {
        return hit + shows + truths.prefix(6).joined(separator: ", ")
      }
      let stop: Set<String> = [
        "the", "there", "this", "that", "does", "did", "is", "are", "was", "were", "and",
        "not", "moment", "frame", "video", "clip", "show", "shows", "shown", "appear",
        "appears", "visible", "have", "has", "any", "still", "you", "can", "see", "around",
        "second", "seconds", "present", "yes", "true", "what", "which", "contain", "contains",
      ]
      // The two tokenizers must agree on short tokens: this one and the
      // real index's search tokenizer both decide what of the model's
      // wording ever reaches the truths, and search already keeps a
      // 2-character token that carries a digit ("1-0"). A bare 3-character
      // floor here does not. Measured (r39, m-ja-check-2): the model asked
      // "Is the PK scene at 460 seconds?" of a frame the penalty row covers
      // — and the floor dropped "PK", the one word that row holds by name,
      // leaving the wrapper noun "scene" to test with, so the check
      // reported a real absence of the wrong word. Keep a token that is
      // long enough, carries a digit, or is written as an acronym (2+
      // uppercase letters as the model typed it); compare lowercased as
      // before.
      //
      // The boundary has to agree too, and did not: search splits on spaces
      // and punctuation and so keeps "1-0" whole, while this one split on
      // every non-alphanumeric and handed back "1" and "0" — a one-digit
      // content word, which is a looser presence test than the search side
      // would ever run. Same separator set as search, so a question naming
      // "1-0" meets the truths' own "1-0" directly.
      func contentWords(_ text: String) -> [String] {
        text.split(whereSeparator: { " ,.!?'\"「」『』、。".contains($0) })
          .map { (typed: String($0), word: String($0).lowercased()) }
          .filter { token in
            guard !stop.contains(token.word) else { return false }
            return token.word.count >= 3 || token.word.contains(where: \.isNumber)
              || (token.typed.count >= 2 && token.typed.allSatisfy { $0.isUppercase })
          }
          .map(\.word)
      }
      var words = contentWords(question) + positives.flatMap(contentWords)
      // JA writes no spaces, so contentWords hands back whole clauses that no
      // English label can ever contain. The same JA→EN detector-noun aliases
      // search_frames carries (the real index's search) carry the presence
      // test across — the findability rule, applied to the check (playbook
      // spec D). The canned rows keep JA keys of their own, so this only
      // matters for the detector nouns a real take utters.
      let asked = ([question] + positives).joined(separator: " ")
      for (ja, en) in [("犬", "dog"), ("いぬ", "dog"), ("猫", "cat"), ("ねこ", "cat")]
      where asked.contains(ja) {
        words.append(en)
      }
      // A check that cannot read the question must not veto (playbook spec
      // D2). contentWords splits on non-letters and JA writes no spaces, so
      // a Japanese question arrives as one whole-clause token — and the
      // presence test asks whether a truth *contains* that token, which no
      // English label can and no short JA key can either (the direct option
      // match above is where JA truths get their chance). What follows is
      // then not an absence but blindness, and blind, the code below used to
      // fall through to the negative option, where the model reads the
      // verdict word as "not there": a confident no from a tool that never
      // read the question is the failure this lane keeps rediscovering
      // (verification vetoes retrieval). So test what the code can see — the
      // truths are written in Latin letters and digits (Vision labels, OCR
      // lines, scores), and a content word holding no ASCII letter or digit
      // could not appear in them whatever the frame held. No such word, the
      // empty list included, means the check could not evaluate: it says so
      // and leaves the model the search hit it already has. A word that does
      // hold one is testable, and an absence found with it is a real one.
      //
      // The same ruling has a second edge, and there the check *can*
      // evaluate — wrongly. A wrapper noun builds the frame of a question
      // without naming what it asks about, so a question left holding only
      // wrapper nouns is being tested for a word it never asked about: the
      // absence is real and the verdict is not. Measured (r39,
      // m-ja-check-2): "Is the PK scene at 460 seconds?" lost "PK" to the
      // old floor, tested "scene" against a frame the penalty row covers,
      // found it absent honestly, and told the model the PK was not there.
      // The floor is fixed above; this is the case that survives it,
      // because a question can also simply be worded that way. Wrappers are
      // their own list, not stopwords — a stopword is dropped and the rest
      // of the question still carries it, while a question that is *all*
      // wrapper has nothing left to carry.
      let wrapper: Set<String> = [
        "scene", "moment", "part", "section", "place", "spot", "thing", "area",
        "場面", "瞬間", "部分", "箇所", "ところ",
      ]
      let named = words.filter { !wrapper.contains($0) }
      let testable = named.contains { word in
        word.contains { $0.isASCII && ($0.isLetter || $0.isNumber) }
      }
      guard testable else {
        return "cannot tell from this frame" + shows + truths.prefix(8).joined(separator: ", ")
      }
      let present = words.contains { word in lower.contains { $0.contains(word) } }
      if present {
        let verdict = positives.first { !["yes", "true"].contains($0.lowercased()) } ?? "yes"
        return verdict + shows + truths.prefix(6).joined(separator: ", ")
      }
      if let no = negatives.first
        ?? options.first(where: { ["no", "false"].contains($0.lowercased()) })
      {
        return no + shows + truths.prefix(6).joined(separator: ", ")
      }
      return "none of those" + shows + truths.prefix(8).joined(separator: ", ")
    }
  }

  /// `--toolset <name>` picks the pack a bench run offers the model.
  static func named(_ name: String) -> [any FoundationModels.Tool]? {
    switch name {
    case "demo": return demo
    case "photo": return photo
    case "focus": return focus
    case "report": return report
    case "video": return video
    case "moments": return moments
    case "store": return store
    case "audio": return audio
    case "docs": return docs
    case "shopping": return shopping
    case "money": return money
    case "inbox": return inbox
    case "crm": return crm
    case "pm": return pm
    case "business": return business
    case "vision": return vision
    // Sight alone, like the stage's look scenario: no tools in the room —
    // the perception controls' clean condition.
    case "look": return []
    default: return nil
    }
  }

  /// The instructions that travel with a pack — the stage's, so a bench
  /// case is the same message the demo sends.
  static func instructions(for name: String) -> String {
    switch name {
    case "video": return ToolBox.videoInstructions
    case "moments": return ToolBox.momentsInstructions
    case "store": return ToolBox.storeInstructions
    case "audio": return ToolBox.audioInstructions
    case "docs": return ToolBox.docsInstructions
    case "shopping": return ToolBox.shoppingInstructions
    case "money": return ToolBox.moneyInstructions
    case "inbox": return ToolBox.inboxInstructions
    case "crm": return ToolBox.crmInstructions
    case "pm": return ToolBox.pmInstructions
    case "business": return ToolBox.businessInstructions
    case "vision", "look": return ToolBox.visionInstructions
    // The loop's own contract: one edit per round, judged on the result.
    // Pinned by run-mac.sh's polish entry (--instructions loop).
    case "loop": return ToolBox.visionLoopInstructions
    default: return ToolBox.instructions
    }
  }
}
