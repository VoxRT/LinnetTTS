import SwiftUI

/// One screen: a text field, Speak, Stop, and a status line with the timings of the last utterance.
struct ContentView: View {
    @StateObject private var speaker = Speaker()
    @State private var text = "Hello there. This is Linnet speaking from your device, one step at a time."

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("LinnetTTS").font(.title.bold())
            TextEditor(text: $text)
                .frame(minHeight: 120)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.4)))
            HStack {
                Button("Speak") { speaker.speak(text) }
                    .buttonStyle(.borderedProminent)
                    .disabled(speaker.busy || !speaker.ready)
                Button("Stop") { speaker.stop() }
                    .buttonStyle(.bordered)
                    .disabled(!speaker.busy)
                Spacer()
                Picker("Threads", selection: $speaker.threads) {
                    Text("auto").tag(0)
                    Text("1").tag(1)
                    Text("2").tag(2)
                }
                .pickerStyle(.segmented)
                .frame(width: 160)
            }
            Text(speaker.status)
                .font(.footnote.monospaced())
                .textSelection(.enabled)
            Spacer()
        }
        .padding()
        .onAppear { speaker.start() }
    }
}
