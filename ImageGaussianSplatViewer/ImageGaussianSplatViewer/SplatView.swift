import SwiftUI
import MetalKit
import ImageGaussianSplatCore

#if os(macOS)
import AppKit

struct SplatView: NSViewRepresentable {
    let renderer: GaussianSplatRenderer

    func makeNSView(context: Context) -> SplatMTKView {
        SplatMTKView(renderer: renderer)
    }

    func updateNSView(_ nsView: SplatMTKView, context: Context) {}
}

final class SplatMTKView: MTKView {
    private let gaussianRenderer: GaussianSplatRenderer
    private var lastDragLocation: NSPoint?

    init(renderer: GaussianSplatRenderer) {
        self.gaussianRenderer = renderer
        super.init(frame: .zero, device: renderer.device)
        clearColor = MTLClearColorMake(0, 0, 0, 1)
        renderer.configure(view: self)
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func magnify(with event: NSEvent) {
        gaussianRenderer.adjustZoom(by: Float(event.magnification))
    }

    override func scrollWheel(with event: NSEvent) {
        let delta = SIMD2<Float>(Float(event.scrollingDeltaX), -Float(event.scrollingDeltaY))
        gaussianRenderer.pan(by: delta)
    }

    override func mouseDown(with event: NSEvent) {
        if event.clickCount == 2 {
            gaussianRenderer.resetView()
            lastDragLocation = nil
        } else {
            lastDragLocation = convert(event.locationInWindow, from: nil)
        }
    }

    override func mouseDragged(with event: NSEvent) {
        guard let last = lastDragLocation else { return }
        let location = convert(event.locationInWindow, from: nil)
        let delta = SIMD2<Float>(Float(location.x - last.x), Float(last.y - location.y))
        gaussianRenderer.pan(by: delta)
        lastDragLocation = location
    }

    override func mouseUp(with event: NSEvent) {
        lastDragLocation = nil
    }
}

#else

import UIKit

struct SplatView: UIViewRepresentable {
    let renderer: GaussianSplatRenderer

    func makeUIView(context: Context) -> SplatMTKUIView {
        SplatMTKUIView(renderer: renderer)
    }

    func updateUIView(_ uiView: SplatMTKUIView, context: Context) {}
}

final class SplatMTKUIView: MTKView {
    private let gaussianRenderer: GaussianSplatRenderer

    init(renderer: GaussianSplatRenderer) {
        self.gaussianRenderer = renderer
        super.init(frame: .zero, device: renderer.device)
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = false
        isPaused = false
        preferredFramesPerSecond = 60
        renderer.configure(view: self)
        backgroundColor = .black
        registerGestures()
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func registerGestures() {
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        addGestureRecognizer(pinch)
        let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        addGestureRecognizer(pan)
    }

    @objc private func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
        let delta = Float(recognizer.scale - 1.0)
        gaussianRenderer.adjustZoom(by: delta)
        recognizer.scale = 1.0
    }

    @objc private func handlePan(_ recognizer: UIPanGestureRecognizer) {
        let translation = recognizer.translation(in: self)
        let delta = SIMD2<Float>(Float(translation.x), Float(-translation.y))
        gaussianRenderer.pan(by: delta)
        recognizer.setTranslation(.zero, in: self)
    }
}

#endif
