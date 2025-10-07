#include <metal_stdlib>
using namespace metal;

// Match CUDA config.h constants
constant uint MAX_TOPK = 10;
constant float EPS = 1e-4;

struct Gaussian {
    float2 center;
    float3 conic;
    float pad;
};

struct TileRange {
    uint start;
    uint end;
};

struct ComputeUniforms {
    uint imageWidth;
    uint imageHeight;
    uint tileWidth;
    uint tileHeight;
    uint tileCountX;
    uint tileCountY;
    uint channels;
    uint topK;  // 0 = no topk normalization, >0 = use topk normalization
};

// Basic rendering kernel (no topk normalization)
kernel void splatGaussians(
    device const Gaussian *gaussians [[buffer(0)]],
    device const float *colors [[buffer(1)]],
    device const uint *gaussianIds [[buffer(2)]],
    device const TileRange *tileRanges [[buffer(3)]],
    constant ComputeUniforms & uniforms [[buffer(4)]],
    texture2d<float, access::write> outTexture [[texture(0)]],
    uint2 pixel [[thread_position_in_grid]],
    uint2 tileIndex [[threadgroup_position_in_grid]]
) {
    if (pixel.x >= uniforms.imageWidth || pixel.y >= uniforms.imageHeight) {
        return;
    }

    const uint tileLinear = tileIndex.y * uniforms.tileCountX + tileIndex.x;
    const TileRange range = tileRanges[tileLinear];

    const uint channels = uniforms.channels;
    float4 accum = float4(0.0f);

    for (uint idx = range.start; idx < range.end; ++idx) {
        const uint gIdx = gaussianIds[idx];
        const Gaussian g = gaussians[gIdx];
        const float2 delta = float2(g.center.x - float(pixel.x), g.center.y - float(pixel.y));
        const float sigma = 0.5f * (g.conic.x * delta.x * delta.x +
                                    g.conic.z * delta.y * delta.y) +
                                    g.conic.y * delta.x * delta.y;
        if (!(sigma >= 0.0f)) {
            continue;
        }
        const float weight = fast::exp(-sigma);
        const uint base = gIdx * channels;
        if (channels > 0) accum.x += colors[base + 0] * weight;
        if (channels > 1) accum.y += colors[base + 1] * weight;
        if (channels > 2) accum.z += colors[base + 2] * weight;
        if (channels > 3) accum.w += colors[base + 3] * weight;
    }

    if (channels == 1) {
        accum = float4(accum.x, accum.x, accum.x, 1.0);
    } else if (channels == 2) {
        accum = float4(accum.x, accum.y, 0.0, 1.0);
    } else if (channels == 3) {
        accum.w = 1.0;
    }

    outTexture.write(accum, pixel);
}

// TopK normalization rendering kernel
kernel void splatGaussiansTopK(
    device const Gaussian *gaussians [[buffer(0)]],
    device const float *colors [[buffer(1)]],
    device const uint *gaussianIds [[buffer(2)]],
    device const TileRange *tileRanges [[buffer(3)]],
    constant ComputeUniforms & uniforms [[buffer(4)]],
    texture2d<float, access::write> outTexture [[texture(0)]],
    uint2 pixel [[thread_position_in_grid]],
    uint2 tileIndex [[threadgroup_position_in_grid]]
) {
    if (pixel.x >= uniforms.imageWidth || pixel.y >= uniforms.imageHeight) {
        return;
    }

    const uint tileLinear = tileIndex.y * uniforms.tileCountX + tileIndex.x;
    const TileRange range = tileRanges[tileLinear];

    const uint channels = uniforms.channels;
    const uint topK = min(uniforms.topK, MAX_TOPK);
    
    // Arrays to track top-K gaussians
    int topkIds[MAX_TOPK];
    float topkVals[MAX_TOPK];
    
    // Initialize topk arrays
    for (uint k = 0; k < MAX_TOPK; ++k) {
        topkIds[k] = -1;
        topkVals[k] = 0.0f;
    }

    // First pass: collect top-K gaussians based on their alpha values
    for (uint idx = range.start; idx < range.end; ++idx) {
        const uint gIdx = gaussianIds[idx];
        const Gaussian g = gaussians[gIdx];
        const float2 delta = float2(g.center.x - float(pixel.x), g.center.y - float(pixel.y));
        const float sigma = 0.5f * (g.conic.x * delta.x * delta.x +
                                    g.conic.z * delta.y * delta.y) +
                                    g.conic.y * delta.x * delta.y;
        if (!(sigma >= 0.0f)) {
            continue;
        }
        
        const float alpha = fast::exp(-sigma);
        
        // Find the minimum value in topk array
        int minTopkIdx = -1;
        float minTopkVal = 1e30f;
        for (uint k = 0; k < topK; ++k) {
            if (topkIds[k] < 0) {
                minTopkIdx = k;
                minTopkVal = -1.0f;
                break;
            } else if (topkVals[k] < minTopkVal) {
                minTopkIdx = k;
                minTopkVal = topkVals[k];
            }
        }
        
        // Replace if this alpha is larger than the minimum
        if (alpha > minTopkVal) {
            topkIds[minTopkIdx] = int(gIdx);
            topkVals[minTopkIdx] = alpha;
        }
    }

    // Second pass: accumulate colors with normalization
    float4 accum = float4(0.0f);
    
    // Compute sum of topk values for normalization
    float sumVal = 0.0f;
    for (uint k = 0; k < topK; ++k) {
        if (topkIds[k] >= 0) {
            sumVal += topkVals[k];
        }
    }
    
    // Accumulate normalized contributions
    for (uint k = 0; k < topK; ++k) {
        const int gIdx = topkIds[k];
        if (gIdx < 0) continue;
        
        // Normalize by sum of topk values
        const float vis = topkVals[k] / (sumVal + EPS);
        const uint base = uint(gIdx) * channels;
        
        if (channels > 0) accum.x += colors[base + 0] * vis;
        if (channels > 1) accum.y += colors[base + 1] * vis;
        if (channels > 2) accum.z += colors[base + 2] * vis;
        if (channels > 3) accum.w += colors[base + 3] * vis;
    }

    if (channels == 1) {
        accum = float4(accum.x, accum.x, accum.x, 1.0);
    } else if (channels == 2) {
        accum = float4(accum.x, accum.y, 0.0, 1.0);
    } else if (channels == 3) {
        accum.w = 1.0;
    }

    outTexture.write(accum, pixel);
}

struct VertexIn {
    float2 position;
    float2 texCoord;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut quadVertex(
    const device VertexIn *vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    const VertexIn v = vertices[vid];
    out.position = float4(v.position, 0.0, 1.0);
    out.texCoord = v.texCoord;
    return out;
}

fragment float4 texturedFragment(
    VertexOut in [[stage_in]],
    texture2d<float> textureIn [[texture(0)]],
    sampler texSampler [[sampler(0)]]
) {
    return textureIn.sample(texSampler, in.texCoord);
}
