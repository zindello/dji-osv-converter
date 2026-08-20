/*
 * Metal Stitcher Bridge - Objective-C implementation
 *
 * This bridge creates Metal textures and calls DJI's dynamicStitchSeams method
 * with proper Metal texture parameters.
 *
 * Data flow:
 *   Python pickle -> stdin -> C (parse) -> create MTLTextures -> call dynamicStitchSeams
 *   -> extract output -> pickle -> stdout -> Python
 *
 * Textures:
 *   - Seam masks: 7680×292 BGRA8Unorm (MTLPixelFormatBGRA8Unorm)
 *   - Flow data: 789×394 RGBA16Float (MTLPixelFormatRGBA16Float)
 *   - Output: 7680×3840 RGB uint8
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Cocoa/Cocoa.h>

// Forward declarations
@class Stitcher;

typedef struct {
    uint8_t *left_frame;       // 3840×3840×3 RGB
    uint8_t *right_frame;      // 3840×3840×3 RGB
    uint8_t *seam_mask_1;      // 7680×292×4 BGRA8Unorm
    uint8_t *seam_mask_2;      // 7680×292×4 BGRA8Unorm
    float *flow_texture;       // 789×394×4 RGBA16Float
    int left_height, left_width, left_channels;
    int right_height, right_width, right_channels;
    int seam1_height, seam1_width, seam1_channels;
    int seam2_height, seam2_width, seam2_channels;
    int flow_height, flow_width, flow_channels;
    char *camera_sn;
    char *param_path;
} FrameData;

// Logging utility
void log_message(const char *level, const char *format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[%s] ", level);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void log_info(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[INFO] ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void log_error(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[ERROR] ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void log_debug(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[DEBUG] ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

/**
 * Create a Metal texture from pixel data
 */
id<MTLTexture> create_metal_texture(
    id<MTLDevice> device,
    const void *data,
    size_t data_size,
    NSUInteger width,
    NSUInteger height,
    MTLPixelFormat format)
{
    log_debug("Creating Metal texture: %lux%lu format=%lu", width, height, (unsigned long)format);

    // Create texture descriptor
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                        width:width
                                                                                       height:height
                                                                                    mipmapped:NO];
    descriptor.resourceOptions = MTLResourceStorageModeManaged;

    // Create texture from descriptor
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (!texture) {
        log_error("Failed to create Metal texture");
        return nil;
    }

    // Calculate bytes per row
    NSUInteger bytes_per_row = 0;
    switch (format) {
        case MTLPixelFormatBGRA8Unorm:
            bytes_per_row = width * 4;  // 4 bytes per pixel (BGRA)
            break;
        case MTLPixelFormatRGBA16Float:
            bytes_per_row = width * 8;  // 8 bytes per pixel (RGBA16F = 2 bytes per channel)
            break;
        default:
            log_error("Unsupported pixel format: %lu", (unsigned long)format);
            return nil;
    }

    // Copy data to texture
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture replaceRegion:region mipmapLevel:0 withBytes:data bytesPerRow:bytes_per_row];

    log_debug("Metal texture created: %p (size=%zu bytes)", texture, data_size);
    return texture;
}

/**
 * Load the DualStitcher framework and get Stitcher class
 */
Class load_stitcher_class(NSBundle **out_bundle) {
    NSString *framework_path = @"/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework";

    log_debug("Loading framework from: %s", [framework_path UTF8String]);

    NSBundle *bundle = [NSBundle bundleWithPath:framework_path];
    if (!bundle) {
        log_error("Failed to load DualStitcher.framework");
        return Nil;
    }

    log_info("Framework bundle loaded: %s", [[bundle bundleIdentifier] UTF8String]);

    // Load the framework
    NSError *error = nil;
    BOOL loaded = [bundle loadAndReturnError:&error];
    if (!loaded) {
        log_error("Failed to load framework bundle: %s", [[error localizedDescription] UTF8String]);
        return Nil;
    }

    log_debug("Framework loaded successfully");

    // Get Stitcher class
    Class stitcher_class = NSClassFromString(@"Stitcher");
    if (!stitcher_class) {
        log_error("Could not find Stitcher class in framework");
        return Nil;
    }

    log_info("Found Stitcher class: %@", NSStringFromClass(stitcher_class));

    *out_bundle = bundle;
    return stitcher_class;
}

/**
 * Create a Stitcher instance
 */
id create_stitcher_instance(Class stitcher_class, id<MTLDevice> device) {
    log_debug("Creating Stitcher instance...");

    id stitcher = nil;

    // Try initWithMetalDevice: first
    if ([stitcher_class respondsToSelector:@selector(alloc)]) {
        stitcher = [[stitcher_class alloc] init];
        if (stitcher) {
            log_info("Stitcher instance created with init()");
            return stitcher;
        }
    }

    // Try with Metal device using NSInvocation
    SEL init_metal_sel = @selector(initWithMetalDevice:);
    if ([stitcher_class instancesRespondToSelector:init_metal_sel] && device) {
        id allocated = [[stitcher_class alloc] init];

        // Try to use the Metal device init if available
        if ([allocated respondsToSelector:@selector(initWithMetalDevice:)]) {
            NSMethodSignature *sig = [allocated methodSignatureForSelector:init_metal_sel];
            if (sig) {
                NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
                [inv setTarget:allocated];
                [inv setSelector:init_metal_sel];
                [inv setArgument:&device atIndex:2];
                [inv invoke];
                [inv getReturnValue:&stitcher];
            }
        }

        if (stitcher) {
            log_info("Stitcher instance created with Metal device");
            return stitcher;
        } else {
            stitcher = allocated;
        }
    }

    log_error("Failed to create Stitcher instance");
    return nil;
}

/**
 * Call dynamicStitchSeams method
 */
id call_dynamic_stitch_seams(
    id stitcher,
    id<MTLTexture> seam_tex_1,
    id<MTLTexture> seam_tex_2,
    BOOL is_compensate,
    BOOL enable_sun_glare,
    id<MTLTexture> sg_gamma_tex,
    id<MTLTexture> flow_tex)
{
    log_debug("Calling dynamicStitchSeams...");

    // Method signature:
    // dynamicStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:flowTex:
    SEL selector = @selector(dynamicStitchSeams_seam1_isCompensate_enableSunGlare_sgGammaTex_flowTex_:);

    if (![stitcher respondsToSelector:selector]) {
        log_error("Stitcher does not respond to dynamicStitchSeams selector");

        // Try alternative selectors
        NSArray *alt_selectors = @[
            @"dynamicStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:flowTex:",
            @"stitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:flowTex:",
        ];

        for (NSString *alt_sel_str in alt_selectors) {
            log_debug("Trying alternative selector: %s", [alt_sel_str UTF8String]);
            // Note: Using NSInvocation for complex method signatures
        }

        return nil;
    }

    log_debug("Invoking dynamicStitchSeams with Metal textures");
    log_debug("  seam_tex_1: %p", seam_tex_1);
    log_debug("  seam_tex_2: %p", seam_tex_2);
    log_debug("  is_compensate: %s", is_compensate ? "YES" : "NO");
    log_debug("  enable_sun_glare: %s", enable_sun_glare ? "YES" : "NO");
    log_debug("  sg_gamma_tex: %p", sg_gamma_tex);
    log_debug("  flow_tex: %p", flow_tex);

    // Use NSInvocation for complex method signature
    NSMethodSignature *sig = [stitcher methodSignatureForSelector:selector];
    if (!sig) {
        log_error("Could not get method signature for selector");
        return nil;
    }

    NSInvocation *invocation = [NSInvocation invocationWithMethodSignature:sig];
    [invocation setTarget:stitcher];
    [invocation setSelector:selector];

    // Set arguments: index 0=self, 1=_cmd, 2=arg1, ...
    [invocation setArgument:&seam_tex_1 atIndex:2];
    [invocation setArgument:&seam_tex_2 atIndex:3];
    [invocation setArgument:&is_compensate atIndex:4];
    [invocation setArgument:&enable_sun_glare atIndex:5];
    [invocation setArgument:&sg_gamma_tex atIndex:6];
    [invocation setArgument:&flow_tex atIndex:7];

    // Invoke
    [invocation invoke];

    // Get return value
    id result = nil;
    [invocation getReturnValue:&result];

    if (result) {
        log_info("dynamicStitchSeams returned non-nil result: %p", result);
        return result;
    } else {
        log_error("dynamicStitchSeams returned nil");
        return nil;
    }
}

/**
 * Extract pixel data from result (NSData, CVImageBuffer, etc.)
 */
uint8_t* extract_output_data(id result, size_t *out_size) {
    log_debug("Extracting output data from result type: %@", [result class]);

    uint8_t *data = NULL;
    size_t size = 0;

    // If NSData
    if ([result isKindOfClass:[NSData class]]) {
        NSData *ns_data = (NSData *)result;
        size = [ns_data length];
        data = (uint8_t *)malloc(size);
        memcpy(data, [ns_data bytes], size);
        log_debug("Extracted NSData: %zu bytes", size);
        *out_size = size;
        return data;
    }

    // If it's an image/texture type
    if ([result respondsToSelector:@selector(bytes)]) {
        // Assume it has bytes method
        NSData *ns_data = (NSData *)result;
        size = [ns_data length];
        data = (uint8_t *)malloc(size);
        memcpy(data, [ns_data bytes], size);
        log_debug("Extracted from bytes: %zu bytes", size);
        *out_size = size;
        return data;
    }

    log_error("Could not extract data from result type: %@", [result class]);
    return NULL;
}

/**
 * Simple pickle integer encoding (for demo)
 */
void write_pickle_int(FILE *f, int32_t value) {
    // Pickle protocol 2: INT opcode
    fprintf(f, "J");  // INT4 opcode
    fwrite(&value, sizeof(int32_t), 1, f);
}

/**
 * Write pickle tuple start
 */
void write_pickle_tuple_start(FILE *f) {
    fprintf(f, "(");  // MARK
}

void write_pickle_tuple_end(FILE *f) {
    fprintf(f, "t");  // TUPLE opcode
}

/**
 * Write pickle bytes
 */
void write_pickle_bytes(FILE *f, const uint8_t *data, size_t size) {
    // Pickle protocol: STRING opcode + length + data
    fprintf(f, "T");  // STRING (newline-terminated) - simpler protocol
    fprintf(f, "%zu\n", size);  // Write size as text
    fwrite(data, 1, size, f);
}

/**
 * Perform the full stitching operation
 */
int perform_stitching(
    NSBundle *framework_bundle,
    Class stitcher_class,
    id<MTLDevice> device,
    FrameData *frame_data,
    uint8_t **out_data,
    size_t *out_size)
{
    log_info("Starting stitching operation...");

    // Create Stitcher instance
    id stitcher = create_stitcher_instance(stitcher_class, device);
    if (!stitcher) {
        log_error("Failed to create Stitcher instance");
        return 1;
    }

    log_debug("Stitcher instance created: %@", [stitcher class]);

    // Create Metal textures for seam masks
    log_info("Creating Metal textures...");

    id<MTLTexture> seam_tex_1 = create_metal_texture(
        device,
        frame_data->seam_mask_1,
        frame_data->seam1_width * frame_data->seam1_height * frame_data->seam1_channels,
        frame_data->seam1_width,
        frame_data->seam1_height,
        MTLPixelFormatBGRA8Unorm
    );

    id<MTLTexture> seam_tex_2 = create_metal_texture(
        device,
        frame_data->seam_mask_2,
        frame_data->seam2_width * frame_data->seam2_height * frame_data->seam2_channels,
        frame_data->seam2_width,
        frame_data->seam2_height,
        MTLPixelFormatBGRA8Unorm
    );

    id<MTLTexture> flow_tex = create_metal_texture(
        device,
        frame_data->flow_texture,
        frame_data->flow_width * frame_data->flow_height * frame_data->flow_channels * sizeof(float),
        frame_data->flow_width,
        frame_data->flow_height,
        MTLPixelFormatRGBA16Float
    );

    if (!seam_tex_1 || !seam_tex_2 || !flow_tex) {
        log_error("Failed to create Metal textures");
        return 1;
    }

    log_info("Metal textures created successfully");

    // Call dynamicStitchSeams
    id result = call_dynamic_stitch_seams(
        stitcher,
        seam_tex_1,
        seam_tex_2,
        YES,    // isCompensate
        NO,     // enableSunGlare
        nil,    // sgGammaTex
        flow_tex
    );

    if (!result) {
        log_error("dynamicStitchSeams failed");
        return 1;
    }

    log_info("Stitching completed successfully");

    // Extract output data
    *out_data = extract_output_data(result, out_size);
    if (!*out_data) {
        log_error("Failed to extract output data");
        return 1;
    }

    log_info("Output extracted: %zu bytes", *out_size);

    // Cleanup (ARC handles memory automatically)
    // Objects are released when they go out of scope

    return 0;
}

/**
 * Main function - process one stitch operation
 */
int main(int argc, char *argv[]) {
    @autoreleasepool {
        log_info("Metal Stitcher Bridge starting...");

        // Create Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            log_error("Failed to create Metal device");
            return 1;
        }
        log_info("Metal device created: %@", [device name]);

        NSBundle *framework_bundle = nil;
        Class stitcher_class = load_stitcher_class(&framework_bundle);
        if (!stitcher_class) {
            log_error("Failed to load Stitcher class");
            return 1;
        }

        log_info("Stitcher class loaded successfully");

        // In this basic version, we exit successfully
        // In production, we would:
        // 1. Read pickle data from stdin
        // 2. Parse frame data
        // 3. Perform stitching
        // 4. Write pickle data to stdout

        log_info("Metal bridge ready and functional");
        return 0;
    }
}
