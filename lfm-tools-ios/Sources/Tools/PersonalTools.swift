// Calendar, reminders, contacts and photos. The interesting half of a phone
// agent, and the half where a wrong answer is expensive — every write here is
// narrow and reports exactly what it created.
import Contacts
import EventKit
import Foundation
import FoundationModels
import Photos

@available(iOS 27.0, *)
enum EventStoreBox {
  /// One store for the whole app: `EKEventStore` re-prompts and drops its
  /// authorization if a new instance is made per call. `EKEventStore` is not
  /// `Sendable`, and its own calls are thread-safe, so the global is marked
  /// unsafe rather than wrapped in an actor that would buy nothing.
  nonisolated(unsafe) static let shared = EKEventStore()
}

@available(iOS 27.0, *)
struct ListEventsTool: Tool {
  let name = "list_calendar_events"
  let description = "Upcoming calendar events."

  @Generable struct Arguments {
    @Guide(description: "How many days ahead to look, 1 to 30.")
    var days: Int
  }

  func call(arguments: Arguments) async throws -> String {
    guard try await EventStoreBox.shared.requestFullAccessToEvents() else {
      return "calendar permission was refused"
    }
    let days = min(30, max(1, arguments.days))
    let end = Calendar.current.date(byAdding: .day, value: days, to: Date()) ?? Date()
    let predicate = EventStoreBox.shared.predicateForEvents(
      withStart: Date(), end: end, calendars: nil)
    let events = EventStoreBox.shared.events(matching: predicate).prefix(20)
    guard !events.isEmpty else { return "nothing in the next \(days) days" }
    let formatter = DateFormatter()
    formatter.dateFormat = "MMM d HH:mm"
    return events.map { "- \(formatter.string(from: $0.startDate)) \($0.title ?? "untitled")" }
      .joined(separator: "\n")
  }
}

@available(iOS 27.0, *)
struct CreateEventTool: Tool {
  let name = "create_calendar_event"
  let description = "Add an event to the default calendar."

  @Generable struct Arguments {
    @Guide(description: "Event title.")
    var title: String
    @Guide(description: "Start time as ISO 8601, e.g. 2026-08-20T14:00:00+09:00")
    var start: String
    @Guide(description: "Length in minutes.")
    var minutes: Int
  }

  func call(arguments: Arguments) async throws -> String {
    guard try await EventStoreBox.shared.requestFullAccessToEvents() else {
      return "calendar permission was refused"
    }
    guard let start = ISO8601DateFormatter().date(from: arguments.start) else {
      return "could not read the start time \(arguments.start)"
    }
    let event = EKEvent(eventStore: EventStoreBox.shared)
    event.title = arguments.title
    event.startDate = start
    event.endDate = start.addingTimeInterval(TimeInterval(max(5, arguments.minutes) * 60))
    event.calendar = EventStoreBox.shared.defaultCalendarForNewEvents
    try EventStoreBox.shared.save(event, span: .thisEvent)
    return "created \"\(arguments.title)\" at \(arguments.start)"
  }
}

@available(iOS 27.0, *)
struct ListRemindersTool: Tool {
  let name = "list_reminders"
  let description = "Reminders that are not done yet."

  func call(arguments: NoArguments) async throws -> String {
    guard try await EventStoreBox.shared.requestFullAccessToReminders() else {
      return "reminders permission was refused"
    }
    let predicate = EventStoreBox.shared.predicateForIncompleteReminders(
      withDueDateStarting: nil, ending: nil, calendars: nil)
    // The titles cross the continuation, not the reminders: `EKReminder` is not
    // `Sendable`, so reading it has to finish inside the callback.
    let titles: [String] = await withCheckedContinuation { continuation in
      EventStoreBox.shared.fetchReminders(matching: predicate) { found in
        continuation.resume(returning: (found ?? []).prefix(20).map { $0.title ?? "untitled" })
      }
    }
    guard !titles.isEmpty else { return "no open reminders" }
    return titles.map { "- \($0)" }.joined(separator: "\n")
  }
}

@available(iOS 27.0, *)
struct CreateReminderTool: Tool {
  let name = "create_reminder"
  let description = "Add a reminder."

  @Generable struct Arguments {
    @Guide(description: "What to be reminded of.")
    var title: String
    @Guide(description: "Optional due time as ISO 8601.")
    var due: String?
  }

  func call(arguments: Arguments) async throws -> String {
    guard try await EventStoreBox.shared.requestFullAccessToReminders() else {
      return "reminders permission was refused"
    }
    let reminder = EKReminder(eventStore: EventStoreBox.shared)
    reminder.title = arguments.title
    reminder.calendar = EventStoreBox.shared.defaultCalendarForNewReminders()
    if let due = arguments.due, let date = ISO8601DateFormatter().date(from: due) {
      reminder.dueDateComponents = Calendar.current.dateComponents(
        [.year, .month, .day, .hour, .minute], from: date)
    }
    try EventStoreBox.shared.save(reminder, commit: true)
    return "added the reminder \"\(arguments.title)\""
  }
}

@available(iOS 27.0, *)
struct SearchContactsTool: Tool {
  let name = "search_contacts"
  let description = "Look someone up in the address book."

  @Generable struct Arguments {
    @Guide(description: "Name or part of a name.")
    var name: String
  }

  func call(arguments: Arguments) async throws -> String {
    let store = CNContactStore()
    guard try await store.requestAccess(for: .contacts) else {
      return "contacts permission was refused"
    }
    let keys =
      [
        CNContactGivenNameKey, CNContactFamilyNameKey, CNContactPhoneNumbersKey,
        CNContactEmailAddressesKey,
      ] as [CNKeyDescriptor]
    let found = try store.unifiedContacts(
      matching: CNContact.predicateForContacts(matchingName: arguments.name), keysToFetch: keys)
    guard !found.isEmpty else { return "no contact matches \(arguments.name)" }
    return found.prefix(5).map { contact in
      let phone = contact.phoneNumbers.first?.value.stringValue
      let mail = contact.emailAddresses.first?.value as String?
      let detail = [phone, mail].compactMap { $0 }.joined(separator: ", ")
      return "- \(contact.givenName) \(contact.familyName)\(detail.isEmpty ? "" : ": \(detail)")"
    }.joined(separator: "\n")
  }
}

@available(iOS 27.0, *)
struct PhotoLibraryTool: Tool {
  let name = "photo_library_summary"
  let description = "How many photos and videos."

  func call(arguments: NoArguments) async throws -> String {
    let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    guard status == .authorized || status == .limited else {
      return "photo library permission was refused"
    }
    let photos = PHAsset.fetchAssets(with: .image, options: nil).count
    let videos = PHAsset.fetchAssets(with: .video, options: nil).count
    let options = PHFetchOptions()
    options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
    options.fetchLimit = 1
    let newest = PHAsset.fetchAssets(with: options).firstObject?.creationDate
    let when = newest.map { DateFormatter.localizedString(from: $0, dateStyle: .medium, timeStyle: .short) }
    return "\(photos) photos, \(videos) videos"
      + (when.map { ", most recent \($0)" } ?? "")
      + (status == .limited ? " (limited selection)" : "")
  }
}
