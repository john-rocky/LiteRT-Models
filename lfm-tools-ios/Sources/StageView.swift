// One beat at a time, full screen.
//
// The chat transcript was the wrong frame: everything arrived at the same size
// in the same grey, so nothing was ever the thing to look at. Here the screen
// shows exactly one moment — the question, then the tokens, then the tool
// firing, then the result — and clears before the next.
import FoundationModels
import LiteRTLM
import LiteRTLMFoundationModels
import SwiftUI

@available(iOS 27.0, *)
struct StageView: View {
  @State private var stage = StageModel()
  @State private var caretOn = false

  var body: some View {
    ZStack {
      Color.black.ignoresSafeArea()
      VStack(spacing: 0) {
        hud
        // The photo scenario's star stays on screen the whole run — every
        // edit lands on it in place. layoutPriority hands it the free space
        // the beat would otherwise claim.
        if let image = stage.stageImage {
          Image(uiImage: image)
            .resizable()
            .scaledToFit()
            .frame(maxWidth: .infinity)
            // The video frame shares the stage with its timeline; a 9:16 crop
            // would otherwise take the whole screen and leave the beat nowhere.
            .frame(maxHeight: stage.stageTimeline == nil && stage.stagePages == nil ? .infinity : 360)
            .clipShape(.rect(cornerRadius: 18))
            .layoutPriority(1)
            .padding(.top, 10)
            .animation(.easeInOut(duration: 0.35), value: stage.stageImageID)
        }
        if let timeline = stage.stageTimeline {
          TimelineStrip(snapshot: timeline)
            .padding(.top, 8)
            .animation(.easeInOut(duration: 0.35), value: stage.stageImageID)
        }
        if let pages = stage.stagePages {
          PageStrip(snapshot: pages)
            .padding(.top, 8)
            .animation(.easeInOut(duration: 0.35), value: stage.stageImageID)
        }
        Spacer(minLength: 8)
        // The beat gets the screen. It was sharing it with two flexible spacers
        // and losing, which is why the map and the photo came out postcard-sized.
        beat
          .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        if case .typing = stage.phase {
          composer
        }
        timeline
      }
      .padding(.horizontal, 16)
    }
    .preferredColorScheme(.dark)
    .statusBarHidden()
    .task { await stage.start() }
  }

  // MARK: The claim, always on screen

  private var hud: some View {
    HStack(spacing: 10) {
      Text(stage.backendName)
        .font(.system(size: 13, weight: .bold, design: .rounded))
      pill("OFFLINE", .green)
      pill("\(stage.toolCount) TOOLS", .secondary)
      Spacer()
      if stage.rate > 0 {
        Text("\(Int(stage.rate)) tok/s")
          .font(.system(size: 13, weight: .semibold, design: .monospaced))
          .foregroundStyle(.green)
          .contentTransition(.numericText())
      }
    }
    .padding(.top, 8)
    .foregroundStyle(.white)
  }

  private func pill(_ text: String, _ tint: Color) -> some View {
    Text(text)
      .font(.system(size: 10, weight: .heavy, design: .rounded))
      .padding(.horizontal, 7)
      .padding(.vertical, 3)
      .background(tint.opacity(0.22), in: Capsule())
      .foregroundStyle(tint)
  }

  // MARK: The beat

  @ViewBuilder private var beat: some View {
    VStack(alignment: .leading, spacing: 20) {
      if !stage.question.isEmpty {
        // A sent message, not a headline: right-aligned, in a bubble, with the
        // person who sent it named.
        HStack(alignment: .top, spacing: 8) {
          Spacer(minLength: 40)
          VStack(alignment: .trailing, spacing: 5) {
            Text("YOU")
              .font(.system(size: 10, weight: .heavy, design: .rounded))
              .foregroundStyle(.white.opacity(0.45))
            Text(stage.question)
              .font(.system(size: 25, weight: .semibold, design: .rounded))
              .foregroundStyle(.white)
              .multilineTextAlignment(.trailing)
              .fixedSize(horizontal: false, vertical: true)
              .padding(.horizontal, 16)
              .padding(.vertical, 12)
              .background(Color.accentColor, in: .rect(cornerRadius: 20))
          }
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
      }

      switch stage.phase {
      case .idle, .typing:
        EmptyView()
      case .thinking:
        ThinkingView(text: stage.live, compact: stage.stageImage != nil)
      case .calling(let name, let arguments, let returned):
        ToolBadge(name: name, arguments: arguments, returned: returned)
      case .result(let text, let artifact):
        // With the photo already filling the stage, a photo artifact card
        // would show the same pixels twice at half the size.
        ResultView(text: text, artifact: stage.stageImage == nil ? artifact : nil)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .animation(.spring(response: 0.42, dampingFraction: 0.78), value: stage.phaseID)
  }

  /// The instruction being typed, in something that looks like a text field,
  /// so the sending is visible rather than assumed.
  private var composer: some View {
    HStack(spacing: 10) {
      HStack(spacing: 2) {
        if stage.listening && stage.typed.isEmpty {
          Text("Listening…")
            .font(.system(size: 19, weight: .medium, design: .rounded))
            .foregroundStyle(.white.opacity(0.4))
        }
        Text(stage.typed)
          .font(.system(size: 19, weight: .medium, design: .rounded))
          .foregroundStyle(.white)
        Rectangle()
          .fill(Color.accentColor)
          .frame(width: 2, height: 22)
          .opacity(caretOn ? 1 : 0.15)
          .animation(.easeInOut(duration: 0.5).repeatForever(), value: caretOn)
        Spacer(minLength: 0)
      }
      .padding(.horizontal, 14)
      .padding(.vertical, 11)
      .background(Color.white.opacity(0.10), in: Capsule())
      // A live microphone while listening, the send arrow otherwise: the
      // recording has to show that the words came in through the air.
      Image(systemName: stage.listening ? "mic.fill" : "arrow.up.circle.fill")
        .font(.system(size: 32))
        .foregroundStyle(stage.listening ? Color.red : Color.accentColor)
        .symbolEffect(.pulse, isActive: stage.listening)
    }
    .padding(.bottom, 10)
    .transition(.move(edge: .bottom).combined(with: .opacity))
    .onAppear { caretOn = true }
  }

  private var timeline: some View {
    HStack(spacing: 7) {
      ForEach(0..<stage.beatCount, id: \.self) { i in
        Capsule()
          .fill(i < stage.beatIndex ? Color.green : Color.white.opacity(0.22))
          .frame(height: 4)
          .frame(maxWidth: .infinity)
      }
    }
    .padding(.bottom, 14)
  }
}

// MARK: - Pieces

/// The documents pack's pages: thumbnails in a row, the open one outlined,
/// a dot on the ones that carry annotations.
@available(iOS 27.0, *)
private struct PageStrip: View {
  let snapshot: DocBox.Snapshot

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack {
        Text(snapshot.title)
        Spacer()
        Text("page \(snapshot.current) of \(snapshot.pageCount)")
      }
      .font(.system(size: 12, weight: .semibold, design: .rounded))
      .foregroundStyle(.white.opacity(0.7))
      ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 8) {
          ForEach(Array(snapshot.thumbnails.enumerated()), id: \.offset) { index, thumb in
            let number = index + 1
            VStack(spacing: 3) {
              Image(uiImage: thumb)
                .resizable()
                .scaledToFit()
                .frame(height: 64)
                .clipShape(.rect(cornerRadius: 4))
                .overlay(
                  RoundedRectangle(cornerRadius: 4)
                    .strokeBorder(number == snapshot.current ? Color.yellow : Color.white.opacity(0.25),
                      lineWidth: number == snapshot.current ? 2.5 : 1))
              HStack(spacing: 3) {
                Text("\(number)")
                  .font(.system(size: 10, weight: .bold, design: .rounded))
                  .foregroundStyle(.white.opacity(0.7))
                if snapshot.annotated[number] != nil {
                  Circle().fill(Color.cyan).frame(width: 5, height: 5)
                }
              }
            }
          }
        }
      }
    }
  }
}

/// The video pack's timeline: clips as blocks (thumbnails inside when they
/// are ready), the selected one outlined, captions and fades marked, and a
/// line at the moment the frame above was taken from. What an editor's
/// timeline looks like, at the size a phone recording can read.
@available(iOS 27.0, *)
private struct TimelineStrip: View {
  let snapshot: VideoEditBox.Snapshot

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 8) {
        Text(snapshot.frame)
        Text("\(VideoEditBox.f(snapshot.duration)) s")
        if snapshot.fadeIn > 0 { Text("fade in") }
        if snapshot.fadeOut > 0 { Text("fade out") }
        if snapshot.volume != 100 {
          Label(snapshot.volume == 0 ? "muted" : "\(snapshot.volume)%",
            systemImage: snapshot.volume == 0 ? "speaker.slash.fill" : "speaker.wave.2.fill")
        }
        Spacer()
        Text("\(snapshot.blocks.count) clip\(snapshot.blocks.count == 1 ? "" : "s")")
      }
      .font(.system(size: 12, weight: .semibold, design: .rounded))
      .foregroundStyle(.white.opacity(0.7))

      GeometryReader { geo in
        let width = geo.size.width
        let scale = snapshot.duration > 0 ? width / snapshot.duration : 0
        ZStack(alignment: .topLeading) {
          // Thumbnails run across the whole timeline; the blocks sit over them.
          HStack(spacing: 0) {
            ForEach(Array(snapshot.thumbnails.enumerated()), id: \.offset) { _, thumb in
              Image(uiImage: thumb)
                .resizable()
                .scaledToFill()
                .frame(width: width / CGFloat(max(1, snapshot.thumbnails.count)), height: 56)
                .clipped()
            }
          }
          .frame(width: width, height: 56)
          .clipShape(.rect(cornerRadius: 8))
          .opacity(0.9)

          ForEach(snapshot.blocks) { block in
            RoundedRectangle(cornerRadius: 8)
              .strokeBorder(
                block.selected ? Color.yellow : Color.white.opacity(0.35),
                lineWidth: block.selected ? 3 : 1)
              .background(
                RoundedRectangle(cornerRadius: 8)
                  .fill(block.selected ? Color.yellow.opacity(0.12) : Color.clear))
              .overlay(alignment: .bottomLeading) {
                if block.speed != 1 {
                  Text("\(VideoEditBox.f(block.speed))×")
                    .font(.system(size: 11, weight: .heavy, design: .rounded))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(Color.black.opacity(0.6), in: Capsule())
                    .foregroundStyle(.white)
                    .padding(4)
                }
              }
              .frame(width: max(2, (block.end - block.start) * scale), height: 56)
              .offset(x: block.start * scale)
          }

          // Captions as a bar along the top edge.
          ForEach(Array(snapshot.captions.enumerated()), id: \.offset) { _, span in
            Capsule()
              .fill(Color.cyan)
              .frame(width: max(3, (span.end - span.start) * scale), height: 4)
              .offset(x: span.start * scale, y: -7)
          }

          // The moment shown above.
          Rectangle()
            .fill(Color.white)
            .frame(width: 2, height: 64)
            .offset(x: min(max(0, snapshot.preview * scale - 1), width - 2), y: -4)
        }
      }
      .frame(height: 64)
    }
  }
}

/// The model generating, made the centre of the screen rather than a footnote.
@available(iOS 27.0, *)
private struct ThinkingView: View {
  let text: String
  var compact = false
  @State private var pulse = false

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack(spacing: 9) {
        Circle()
          .fill(Color.green)
          .frame(width: 9, height: 9)
          .opacity(pulse ? 0.25 : 1)
          .animation(.easeInOut(duration: 0.6).repeatForever(), value: pulse)
        Text("thinking on the phone")
          .font(.system(size: 14, weight: .medium, design: .rounded))
          .foregroundStyle(.secondary)
      }
      // Compact when a photo owns the stage: the stream is a heartbeat there,
      // not the show.
      Text(String(text.suffix(compact ? 150 : 400)))
        .font(.system(size: compact ? 14 : 18, design: .monospaced))
        .foregroundStyle(.green)
        .frame(maxWidth: .infinity, minHeight: compact ? 40 : 220, alignment: .topLeading)
        .animation(.none, value: text)
    }
    .onAppear { pulse = true }
  }
}

/// The moment the model reaches out of itself. Deliberately loud.
@available(iOS 27.0, *)
private struct ToolBadge: View {
  let name: String
  let arguments: String
  let returned: String?
  @State private var landed = false

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack(spacing: 12) {
        Image(systemName: "bolt.fill")
          .font(.system(size: 26, weight: .black))
        Text(name)
          .font(.system(size: 30, weight: .heavy, design: .monospaced))
          .lineLimit(1)
          .minimumScaleFactor(0.5)
      }
      .foregroundStyle(.black)
      .padding(.horizontal, 18)
      .padding(.vertical, 12)
      .background(Color.green, in: .rect(cornerRadius: 16))
      .scaleEffect(landed ? 1 : 1.35)
      .opacity(landed ? 1 : 0)

      if !arguments.isEmpty {
        Text(arguments)
          .font(.system(size: 16, design: .monospaced))
          .foregroundStyle(.white.opacity(0.75))
      }
      if let returned, !returned.isEmpty {
        HStack(alignment: .top, spacing: 8) {
          Image(systemName: "arrow.turn.down.right").foregroundStyle(.green)
          Text(returned)
            .font(.system(size: 19, weight: .medium, design: .rounded))
            .foregroundStyle(.white)
            .fixedSize(horizontal: false, vertical: true)
        }
        .transition(.opacity)
      }
    }
    .onAppear {
      withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) { landed = true }
    }
  }
}

@available(iOS 27.0, *)
private struct ResultView: View {
  let text: String
  let artifact: Artifact?

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      if let artifact {
        ArtifactView(artifact: artifact)
      }
      Text(text)
        .font(.system(size: 24, weight: .medium, design: .rounded))
        .foregroundStyle(.white)
        .fixedSize(horizontal: false, vertical: true)
    }
  }
}
