#ifndef LAR_TRACKING_IMAGE_INPUT_H
#define LAR_TRACKING_IMAGE_INPUT_H

// Plain-C image input type for the tracking/localization interface.
//
// Deliberately plain C (no opencv, no C++), so the same definition can be shared verbatim
// across the C++ core, an Objective-C bridge, and Swift (where it imports as a native value
// type, e.g. `LARImageInput(data:width:height:bytesPerRow:)`) without dragging opencv/C++
// headers across the language boundary.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Grayscale (single-channel, 8-bit) image input.
///
/// `data` points to the luma buffer; rows are `bytesPerRow`-strided. The pointer is **not**
/// owned or copied — it must stay valid for the duration of the call that receives it.
typedef struct LARImageInput {
    const void *data;
    int32_t width;
    int32_t height;
    int32_t bytesPerRow;
} LARImageInput;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LAR_TRACKING_IMAGE_INPUT_H
