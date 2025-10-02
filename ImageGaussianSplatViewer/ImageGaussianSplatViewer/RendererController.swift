import Foundation
import Combine
import SwiftUI
import UniformTypeIdentifiers
import ImageGaussianSplatCore

#if os(macOS)
import AppKit
#endif

@MainActor
final class RendererController: ObservableObject {
    let renderer: GaussianSplatRenderer
    #if os(macOS)
    @Published var status: String = "Load an .igs2 file to render."
    #else
    @Published var status: String = "Provide .igs2 data programmatically to render on this platform."
    #endif

    init?() {
        guard let renderer = GaussianSplatRenderer() else {
            return nil
        }
        self.renderer = renderer
    }

    #if os(macOS)
    func openDocumentPanel() {
        let panel = NSOpenPanel()
        let allowedTypes: [UTType] = [UTType(filenameExtension: "igs2")].compactMap { $0 }
        panel.allowedContentTypes = allowedTypes
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.title = "Select Gaussian Splat File"
        if panel.runModal() == .OK, let url = panel.url {
            loadFile(url: url)
        }
    }
    #endif

    func loadFile(url: URL) {
        do {
            try renderer.loadScene(from: url)
            status = "Loaded \(url.lastPathComponent)"
        } catch {
            status = "Failed to load: \(error.localizedDescription)"
        }
    }

    func resetView() {
        renderer.resetView()
        #if os(macOS)
        status = "View reset"
        #endif
    }
}
