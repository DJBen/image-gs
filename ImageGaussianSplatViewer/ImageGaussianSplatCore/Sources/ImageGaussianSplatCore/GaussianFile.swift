import Foundation
import simd

public struct GaussianFileHeader {
    public let imageWidth: Int
    public let imageHeight: Int
    public let tileWidth: Int
    public let tileHeight: Int
    public let channels: Int
    public let gaussianCount: Int
    public let intersectionCount: Int
    public let topK: Int

    public var tileCountX: Int {
        (imageWidth + tileWidth - 1) / tileWidth
    }

    public var tileCountY: Int {
        (imageHeight + tileHeight - 1) / tileHeight
    }
}

public struct GaussianScene {
    public let header: GaussianFileHeader
    public let centers: [SIMD2<Float>]
    public let conics: [SIMD3<Float>]
    public let colors: [Float]
    public let gaussianIds: [UInt32]
    public let tileBins: [SIMD2<UInt32>]

    public init(
        header: GaussianFileHeader,
        centers: [SIMD2<Float>],
        conics: [SIMD3<Float>],
        colors: [Float],
        gaussianIds: [UInt32],
        tileBins: [SIMD2<UInt32>]
    ) {
        self.header = header
        self.centers = centers
        self.conics = conics
        self.colors = colors
        self.gaussianIds = gaussianIds
        self.tileBins = tileBins
    }
}

public enum GaussianFileError: Error {
    case corrupted(String)
    case invalidChunk(tag: String)
    case unsupported(String)
}

private enum DType: UInt16 {
    case float16 = 0
    case float32 = 1
    case float64 = 2
    case uint16  = 3
    case uint32  = 4
    case int16   = 5
    case int32   = 6
}

private struct ChunkHeader {
    let tag: String
    let dtype: DType
    let rank: Int
    let byteLength: Int
    let shape: [Int]
    let payloadRange: Range<Int>
}

public enum GaussianFileLoader {
    public static func load(url: URL) throws -> GaussianScene {
        let data = try Data(contentsOf: url)
        return try load(data: data)
    }

    public static func load(data: Data) throws -> GaussianScene {
        guard data.count >= 64 else {
            throw GaussianFileError.corrupted("File too small")
        }

        let header = try parseHeader(data: data.subdata(in: 0..<64))
        var offset = 64
        var chunks: [String: ChunkHeader] = [:]

        while offset < data.count {
            let chunk = try parseChunk(data: data, offset: offset)
            chunks[chunk.tag] = chunk
            offset = chunk.payloadRange.upperBound
        }

        guard let centersHeader = chunks["CNTR"], centersHeader.dtype == .float32 else {
            throw GaussianFileError.invalidChunk(tag: "CNTR")
        }
        guard let conicsHeader = chunks["CNIC"], conicsHeader.dtype == .float32 else {
            throw GaussianFileError.invalidChunk(tag: "CNIC")
        }
        guard let colorsHeader = chunks["COLR"], colorsHeader.dtype == .float32 else {
            throw GaussianFileError.invalidChunk(tag: "COLR")
        }
        guard let gidxHeader = chunks["GIDX"], gidxHeader.dtype == .uint32 else {
            throw GaussianFileError.invalidChunk(tag: "GIDX")
        }
        guard let tbinHeader = chunks["TBIN"], tbinHeader.dtype == .uint32 else {
            throw GaussianFileError.invalidChunk(tag: "TBIN")
        }

        let centers = try decodeSIMD2(data.subdata(in: centersHeader.payloadRange))
        let conics = try decodeSIMD3(data.subdata(in: conicsHeader.payloadRange))
        let colors = try decodeFloatArray(data.subdata(in: colorsHeader.payloadRange))
        let gaussianIds = try decodeUInt32Array(data.subdata(in: gidxHeader.payloadRange))
        let tileBins = try decodeSIMD2UInt(data.subdata(in: tbinHeader.payloadRange))

        if centers.count != header.gaussianCount {
            throw GaussianFileError.corrupted("Center count mismatch")
        }
        if conics.count != header.gaussianCount {
            throw GaussianFileError.corrupted("Conic count mismatch")
        }
        if colors.count != header.gaussianCount * header.channels {
            throw GaussianFileError.corrupted("Color count mismatch")
        }
        if gaussianIds.count != header.intersectionCount {
            throw GaussianFileError.corrupted("Intersection count mismatch")
        }

        return GaussianScene(
            header: header,
            centers: centers,
            conics: conics,
            colors: colors,
            gaussianIds: gaussianIds,
            tileBins: tileBins
        )
    }

    private static func parseHeader(data: Data) throws -> GaussianFileHeader {
        var cursor = data
        guard let magic = String(data: cursor.prefix(4), encoding: .ascii), magic == "IGS2" else {
            throw GaussianFileError.corrupted("Missing IGS2 magic")
        }
        cursor.removeFirst(4)
        let versionMajor = cursor.removeFirst()
        let versionMinor = cursor.removeFirst()
        if versionMajor != 0 {
            throw GaussianFileError.unsupported("Unsupported version \(versionMajor).\(versionMinor)")
        }
        cursor.removeFirst(1) // flags
        cursor.removeFirst(1) // padding
        let tileWidth = Int(cursor.readUInt16())
        let tileHeight = Int(cursor.readUInt16())
        let channels = Int(cursor.readUInt16())
        let topK = Int(cursor.readUInt16())
        let imageWidth = Int(cursor.readUInt32())
        let imageHeight = Int(cursor.readUInt32())
        let gaussianCount = Int(cursor.readUInt32())
        let intersectionCount = Int(cursor.readUInt32())
        return GaussianFileHeader(
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            channels: channels,
            gaussianCount: gaussianCount,
            intersectionCount: intersectionCount,
            topK: topK
        )
    }

    private static func parseChunk(data: Data, offset: Int) throws -> ChunkHeader {
        guard offset + 12 <= data.count else {
            throw GaussianFileError.corrupted("Truncated chunk header")
        }
        let headerSlice = data.subdata(in: offset..<(offset + 12))
        var cursor = headerSlice
        guard let tag = String(data: cursor.prefix(4), encoding: .ascii) else {
            throw GaussianFileError.corrupted("Invalid chunk tag")
        }
        cursor.removeFirst(4)
        let dtypeRaw = cursor.readUInt16()
        guard let dtype = DType(rawValue: dtypeRaw) else {
            throw GaussianFileError.unsupported("Unknown dtype code \(dtypeRaw)")
        }
        let rank = Int(cursor.readUInt16())
        let byteLength = Int(cursor.readInt32())
        if byteLength < 0 {
            throw GaussianFileError.corrupted("Negative payload length for chunk \(tag)")
        }
        let shapeLength = 4 * rank
        let shapeStart = offset + 12
        let shapeEnd = shapeStart + shapeLength
        guard shapeEnd <= data.count else {
            throw GaussianFileError.corrupted("Truncated shape for chunk \(tag)")
        }
        let shapeSlice = data.subdata(in: shapeStart..<shapeEnd)
        var shapeCursor = shapeSlice
        var shape: [Int] = []
        for _ in 0..<rank {
            shape.append(Int(shapeCursor.readUInt32()))
        }
        let payloadStart = shapeEnd
        let payloadEnd = payloadStart + byteLength
        guard payloadEnd <= data.count else {
            throw GaussianFileError.corrupted("Truncated payload for chunk \(tag)")
        }
        return ChunkHeader(
            tag: tag,
            dtype: dtype,
            rank: rank,
            byteLength: byteLength,
            shape: shape,
            payloadRange: payloadStart..<payloadEnd
        )
    }

    private static func decodeSIMD2(_ data: Data) throws -> [SIMD2<Float>] {
        let floats = try decodeFloatArray(data)
        guard floats.count % 2 == 0 else {
            throw GaussianFileError.corrupted("CNTR payload size invalid")
        }
        return strideArray(floats, stride: 2) { SIMD2($0[0], $0[1]) }
    }

    private static func decodeSIMD3(_ data: Data) throws -> [SIMD3<Float>] {
        let floats = try decodeFloatArray(data)
        guard floats.count % 3 == 0 else {
            throw GaussianFileError.corrupted("CNIC payload size invalid")
        }
        return strideArray(floats, stride: 3) { SIMD3($0[0], $0[1], $0[2]) }
    }

    private static func decodeSIMD2UInt(_ data: Data) throws -> [SIMD2<UInt32>] {
        let ints = try decodeUInt32Array(data)
        guard ints.count % 2 == 0 else {
            throw GaussianFileError.corrupted("TBIN payload size invalid")
        }
        return strideArray(ints, stride: 2) { SIMD2($0[0], $0[1]) }
    }

    private static func decodeFloatArray(_ data: Data) throws -> [Float] {
        return data.withUnsafeBytes { buffer -> [Float] in
            let ptr = buffer.bindMemory(to: Float.self)
            return Array(ptr)
        }
    }

    private static func decodeUInt32Array(_ data: Data) throws -> [UInt32] {
        return data.withUnsafeBytes { buffer -> [UInt32] in
            let ptr = buffer.bindMemory(to: UInt32.self)
            return Array(ptr)
        }
    }

    private static func strideArray<T, R>(_ array: [T], stride: Int, builder: ([T]) -> R) -> [R] {
        var result: [R] = []
        result.reserveCapacity(array.count / stride)
        var index = 0
        while index + stride <= array.count {
            let slice = Array(array[index..<(index + stride)])
            result.append(builder(slice))
            index += stride
        }
        return result
    }
}

private extension Data {
    mutating func readUInt16() -> UInt16 {
        let value = withUnsafeBytes { $0.load(as: UInt16.self) }
        removeFirst(MemoryLayout<UInt16>.size)
        return UInt16(littleEndian: value)
    }

    mutating func readUInt32() -> UInt32 {
        let value = withUnsafeBytes { $0.load(as: UInt32.self) }
        removeFirst(MemoryLayout<UInt32>.size)
        return UInt32(littleEndian: value)
    }

    mutating func readInt32() -> Int32 {
        let value = withUnsafeBytes { $0.load(as: Int32.self) }
        removeFirst(MemoryLayout<Int32>.size)
        return Int32(littleEndian: value)
    }
}
