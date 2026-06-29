#ifndef LAR_CORE_SPATIAL_SPATIAL_QUERY_H
#define LAR_CORE_SPATIAL_SPATIAL_QUERY_H

// Plain-C spatial query type.
//
// Deliberately plain C (no C++), so the same definition can be shared verbatim across the C++
// core, an Objective-C bridge, and Swift (where it imports as a native value type,
// e.g. `LARSpatialQuery(x:z:diameter:)`).

#ifdef __cplusplus
extern "C" {
#endif

/// Spatial query: search center in map-local coordinates (x, z) and search diameter in meters.
typedef struct LARSpatialQuery {
    double x;
    double z;
    double diameter;
} LARSpatialQuery;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LAR_CORE_SPATIAL_SPATIAL_QUERY_H
