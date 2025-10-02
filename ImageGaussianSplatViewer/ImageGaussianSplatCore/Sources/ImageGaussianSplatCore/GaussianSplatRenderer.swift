import Foundation
import MetalKit
import simd

public struct MetalGaussian {
    public var center: SIMD2<Float>
    public var conic: SIMD3<Float>
    public var pad: Float = 0

    public init(center: SIMD2<Float>, conic: SIMD3<Float>) {
        self.center = center
        self.conic = conic
        self.pad = 0
    }
}

struct TileRangeGPU {
    var start: UInt32
    var end: UInt32
}

struct Vertex {
    var position: SIMD2<Float>
    var texCoord: SIMD2<Float>
}

struct ComputeUniforms {
    var imageWidth: UInt32
    var imageHeight: UInt32
    var tileWidth: UInt32
    var tileHeight: UInt32
    var tileCountX: UInt32
    var tileCountY: UInt32
    var channels: UInt32
    var padding: UInt32 = 0
}

public final class GaussianSplatRenderer: NSObject, MTKViewDelegate {
    public let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let computePipeline: MTLComputePipelineState
    private let renderPipeline: MTLRenderPipelineState
    private let samplerState: MTLSamplerState

    private var scene: GaussianScene?
    private var gaussianBuffer: MTLBuffer?
    private var colorBuffer: MTLBuffer?
    private var gaussianIdBuffer: MTLBuffer?
    private var tileRangeBuffer: MTLBuffer?
    private var uniformBuffer: MTLBuffer?
    private var vertexBuffer: MTLBuffer?
    private var accumTexture: MTLTexture?

    private var needsCompute = false
    private var transformDirty = true
    private var currentDrawableSize: CGSize = .zero

    private var zoomValue: Float = 1.0
    private var panValue: SIMD2<Float> = .zero

    public init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device = device,
              let queue = device.makeCommandQueue()
        else { return nil }
        self.device = device
        self.commandQueue = queue

        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let computeFunction = library.makeFunction(name: "splatGaussians"),
                  let vertexFunction = library.makeFunction(name: "quadVertex"),
                  let fragmentFunction = library.makeFunction(name: "texturedFragment")
            else {
                return nil
            }

            self.computePipeline = try device.makeComputePipelineState(function: computeFunction)

            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            self.renderPipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

            let samplerDescriptor = MTLSamplerDescriptor()
            samplerDescriptor.minFilter = .linear
            samplerDescriptor.magFilter = .linear
            samplerDescriptor.mipFilter = .notMipmapped
            samplerDescriptor.sAddressMode = .clampToEdge
            samplerDescriptor.tAddressMode = .clampToEdge
            guard let samplerState = device.makeSamplerState(descriptor: samplerDescriptor) else {
                return nil
            }
            self.samplerState = samplerState
        } catch {
            print("Failed to initialize renderer: \(error)")
            return nil
        }
        super.init()
    }

    public func configure(view: MTKView) {
        view.device = device
        view.delegate = self
        view.framebufferOnly = false
        view.colorPixelFormat = .bgra8Unorm
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.preferredFramesPerSecond = 60
    }

    public func setZoom(_ value: Float) {
        let clamped = min(max(value, 0.1), 20.0)
        if abs(clamped - zoomValue) > 1e-4 {
            zoomValue = clamped
            transformDirty = true
        }
    }

    public func adjustZoom(by magnification: Float) {
        setZoom(zoomValue * (1.0 + magnification))
    }

    public func pan(by delta: SIMD2<Float>) {
        panValue += delta
        transformDirty = true
    }

    public func resetView() {
        zoomValue = 1.0
        panValue = .zero
        transformDirty = true
    }

    public func loadScene(from url: URL) throws {
        let newScene = try GaussianFileLoader.load(url: url)
        buildBuffers(for: newScene)
        scene = newScene
        needsCompute = true
        transformDirty = true
    }

    public func loadScene(data: Data) throws {
        let newScene = try GaussianFileLoader.load(data: data)
        buildBuffers(for: newScene)
        scene = newScene
        needsCompute = true
        transformDirty = true
    }

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        currentDrawableSize = size
        transformDirty = true
    }

    public func draw(in view: MTKView) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        currentDrawableSize = view.drawableSize

        if needsCompute {
            encodeComputePass(commandBuffer: commandBuffer)
        }

        guard let renderDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable
        else {
            commandBuffer.commit()
            return
        }

        updateVertexBufferIfNeeded()

        guard let vertexBuffer = vertexBuffer,
              let accumTexture = accumTexture
        else {
            commandBuffer.commit()
            return
        }

        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderDescriptor) {
            renderEncoder.label = "GaussianRenderPass"
            renderEncoder.setRenderPipelineState(renderPipeline)
            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            renderEncoder.setFragmentTexture(accumTexture, index: 0)
            renderEncoder.setFragmentSamplerState(samplerState, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func encodeComputePass(commandBuffer: MTLCommandBuffer) {
        guard needsCompute else { return }
        guard let scene = scene else {
            needsCompute = false
            return
        }
        guard let gaussianBuffer,
              let colorBuffer,
              let gaussianIdBuffer,
              let tileRangeBuffer,
              let uniformBuffer,
              let accumTexture,
              let computeEncoder = commandBuffer.makeComputeCommandEncoder()
        else {
            needsCompute = false
            return
        }

        var uniforms = ComputeUniforms(
            imageWidth: UInt32(scene.header.imageWidth),
            imageHeight: UInt32(scene.header.imageHeight),
            tileWidth: UInt32(scene.header.tileWidth),
            tileHeight: UInt32(scene.header.tileHeight),
            tileCountX: UInt32(scene.header.tileCountX),
            tileCountY: UInt32(scene.header.tileCountY),
            channels: UInt32(scene.header.channels)
        )
        memcpy(uniformBuffer.contents(), &uniforms, MemoryLayout<ComputeUniforms>.stride)

        computeEncoder.label = "GaussianComputePass"
        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setBuffer(gaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(colorBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(gaussianIdBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(tileRangeBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(uniformBuffer, offset: 0, index: 4)
        computeEncoder.setTexture(accumTexture, index: 0)

        let threadsPerGroup = MTLSize(width: scene.header.tileWidth, height: scene.header.tileHeight, depth: 1)
        let groupsX = (scene.header.imageWidth + scene.header.tileWidth - 1) / scene.header.tileWidth
        let groupsY = (scene.header.imageHeight + scene.header.tileHeight - 1) / scene.header.tileHeight
        let threadgroups = MTLSize(width: groupsX, height: groupsY, depth: 1)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        needsCompute = false
    }

    private func buildBuffers(for scene: GaussianScene) {
        let count = scene.header.gaussianCount
        var gaussians: [MetalGaussian] = []
        gaussians.reserveCapacity(count)
        for idx in 0..<count {
            gaussians.append(MetalGaussian(center: scene.centers[idx], conic: scene.conics[idx]))
        }
        gaussianBuffer = makeBuffer(from: gaussians, label: "Gaussians")
        colorBuffer = makeBuffer(from: scene.colors, label: "Colors")
        gaussianIdBuffer = makeBuffer(from: scene.gaussianIds, label: "GaussianIndices")

        var ranges: [TileRangeGPU] = []
        ranges.reserveCapacity(scene.tileBins.count)
        for tile in scene.tileBins {
            ranges.append(TileRangeGPU(start: tile.x, end: tile.y))
        }
        tileRangeBuffer = makeBuffer(from: ranges, label: "TileRanges")
        uniformBuffer = device.makeBuffer(length: MemoryLayout<ComputeUniforms>.stride, options: .storageModeShared)

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: scene.header.imageWidth,
            height: scene.header.imageHeight,
            mipmapped: false
        )
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .private
        accumTexture = device.makeTexture(descriptor: descriptor)
        accumTexture?.label = "GaussianAccumulation"
    }

    private func makeBuffer<T>(from array: [T], label: String) -> MTLBuffer? {
        if array.isEmpty {
            let length = max(MemoryLayout<T>.stride, 16)
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                return nil
            }
            memset(buffer.contents(), 0, length)
            buffer.label = label
            return buffer
        } else {
            let length = array.count * MemoryLayout<T>.stride
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                return nil
            }
            _ = array.withUnsafeBytes { raw in
                memcpy(buffer.contents(), raw.baseAddress!, raw.count)
            }
            buffer.label = label
            return buffer
        }
    }

    private func updateVertexBufferIfNeeded() {
        guard transformDirty,
              let scene = scene
        else { return }

        let drawableWidth = max(Float(currentDrawableSize.width), 1.0)
        let drawableHeight = max(Float(currentDrawableSize.height), 1.0)
        let imageWidth = Float(scene.header.imageWidth)
        let imageHeight = Float(scene.header.imageHeight)

        let clipHalfWidth = (imageWidth * zoomValue) / drawableWidth
        let clipHalfHeight = (imageHeight * zoomValue) / drawableHeight
        let offsetX = (panValue.x / drawableWidth) * 2.0
        let offsetY = (panValue.y / drawableHeight) * 2.0

        let left = -clipHalfWidth + offsetX
        let right = clipHalfWidth + offsetX
        let top = clipHalfHeight + offsetY
        let bottom = -clipHalfHeight + offsetY

        let vertices: [Vertex] = [
            Vertex(position: SIMD2(left, bottom), texCoord: SIMD2(0, 1)),
            Vertex(position: SIMD2(left, top), texCoord: SIMD2(0, 0)),
            Vertex(position: SIMD2(right, bottom), texCoord: SIMD2(1, 1)),
            Vertex(position: SIMD2(right, top), texCoord: SIMD2(1, 0)),
        ]

        if vertexBuffer == nil {
            vertexBuffer = device.makeBuffer(length: MemoryLayout<Vertex>.stride * vertices.count, options: .storageModeShared)
            vertexBuffer?.label = "QuadVertices"
        }
        if let contents = vertexBuffer?.contents() {
            _ = vertices.withUnsafeBytes { buffer in
                memcpy(contents, buffer.baseAddress!, buffer.count)
            }
        }
        transformDirty = false
    }
}
