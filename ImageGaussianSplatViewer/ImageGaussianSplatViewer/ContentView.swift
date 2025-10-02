import SwiftUI
import ImageGaussianSplatCore

struct ContentView: View {
    @StateObject private var controller: RendererController

    init() {
        if let rendererController = RendererController() {
            _controller = StateObject(wrappedValue: rendererController)
        } else {
            fatalError("Unable to create Gaussian renderer on this device")
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SplatView(renderer: controller.renderer)
                .frame(minWidth: 640, minHeight: 480)
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .shadow(radius: 4)
            controlBar
        }
        .padding()
        .navigationTitle("Image Gaussian Splat Viewer")
    }

    @ViewBuilder
    private var controlBar: some View {
        HStack(spacing: 12) {
            #if os(macOS)
            Button("Open Gaussian File…") {
                controller.openDocumentPanel()
            }
            #else
            Button("Reset View") {
                controller.resetView()
            }
            #endif

            #if os(macOS)
            Button("Reset View") {
                controller.resetView()
            }
            #endif

            Spacer()
            Text(controller.status)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    ContentView()
}
