# Odometry Confidence Tiers

How we tag each between-snap odometry edge with a coarse, **tuning-free** confidence
signal at capture time, and let the **offline** refiner (and future loop closure) turn
that into weights. This is the on-device input to the loop-closure work in
[`RECONSTRUCTION_CLEANUP_PLAN.md`](RECONSTRUCTION_CLEANUP_PLAN.md).

## Design contract

- **Capture side makes no numeric decisions.** It records only categorical labels the
  AR platform already computed (`ARCamera.trackingState` on iOS, `TrackingState` +
  `TrackingFailureReason` on Android). No thresholds, no per-meter tuning on device.
- **Offline side owns all tuning.** The refiner maps the categorical tier → an
  information-matrix scale. Every number lives in `bundle_adjustment.cpp`, where tuning
  is fine and platform-independent.
- **Platform-neutral on disk.** `frames.json` stores one integer per frame. iOS and a
  future Android app each supply a tiny native→neutral mapping; everything downstream
  is identical regardless of capture platform.
- **The signal is monitored continuously between snaps, not sampled at the snap.** An
  odometry edge is a property of the whole interval between two keyframes, so we
  accumulate the *worst* state seen across that interval and attach it to the edge.

## The platform-neutral enum

Stored as `odom_state` (int) on each `Frame`. Severity increases with the value, so a
plain `max()` over the interval is the correct "worst-of" reduction. Values group into
three offline tiers:

| int | neutral state | tier | meaning for the odometry edge |
| --- | --- | --- | --- |
| 0 | `NORMAL` | **HIGH** | tracking nominal the entire interval — trust the relative pose |
| 1 | `LIMITED_INITIALIZING` | **MEDIUM** | still initializing; noisy but continuous motion |
| 2 | `LIMITED_EXCESSIVE_MOTION` | **MEDIUM** | motion blur / fast motion; downweight |
| 3 | `LIMITED_INSUFFICIENT_VISUAL` | **MEDIUM** | too few features / too little light; downweight |
| 4 | `RELOCALIZING` | **LOW** | recovered from tracking loss — edge **spans a discontinuity** |
| 5 | `UNAVAILABLE` | **LOW** | no tracking during the interval — relative pose meaningless |

The MEDIUM/LOW split is the whole point: MEDIUM = *soften* (real motion, just noisy),
LOW = *break* (the relative pose crosses an ARKit/ARCore correction and isn't smooth
motion). Loop closure distributes drift correction into LOW then MEDIUM edges while
HIGH edges stay rigid.

> Start-simple option: collapse to 2 tiers (`0` = HIGH, everything else = LOW) — a strict
> subset. You can promote to 3 tiers later without recapturing, since the raw int is
> stored, not the tier.

### iOS — `ARCamera.TrackingState` → neutral int

```
.normal                          -> 0  NORMAL
.limited(.initializing)          -> 1  LIMITED_INITIALIZING
.limited(.excessiveMotion)       -> 2  LIMITED_EXCESSIVE_MOTION
.limited(.insufficientFeatures)  -> 3  LIMITED_INSUFFICIENT_VISUAL
.limited(.relocalizing)          -> 4  RELOCALIZING
.notAvailable                    -> 5  UNAVAILABLE
```

### Android — `TrackingState` + `TrackingFailureReason` → neutral int (for the future ARCore app)

```
TRACKING                                 -> 0  NORMAL
PAUSED + NONE                            -> 1  LIMITED_INITIALIZING   (startup pause)
PAUSED + EXCESSIVE_MOTION                -> 2  LIMITED_EXCESSIVE_MOTION
PAUSED + INSUFFICIENT_FEATURES           -> 3  LIMITED_INSUFFICIENT_VISUAL
PAUSED + INSUFFICIENT_LIGHT              -> 3  LIMITED_INSUFFICIENT_VISUAL
PAUSED + BAD_STATE                       -> 4  RELOCALIZING           (internal recovery)
PAUSED + CAMERA_UNAVAILABLE              -> 5  UNAVAILABLE
STOPPED                                  -> 5  UNAVAILABLE
```

Only this mapping and the accumulator below are platform-specific. Nothing else changes.

## Capture accumulator (platform-agnostic logic)

One integer of state, updated every frame in the existing per-frame callback, flushed at
each snap. No allocation, no per-frame persistence.

```
// reset to the current endpoint state so the next interval starts fresh
on each AR frame:      worst = max(worst, neutralState(frame))     // continuous monitor
on snap(frame):        edgeState = worst
                       worst     = neutralState(frame)             // reset for next interval
                       addFrame(..., odomState: edgeState)
```

`odom_state` on a frame describes the edge **into** that frame (the interval from the
previous snap up to and including this one). The refiner reads it on the destination
vertex — consistent with how `addOdometry` builds the edge `(frame_id-1 -> frame_id)`.

### iOS wiring (`Examples/LARScan/.../ViewController.swift`, in the lar-swift repo)

- Add `private var odomWorstState: Int32 = 0` (MainActor-isolated, like the rest of the
  controller state).
- Add a pure mapping helper:

  ```swift
  private func neutralOdomState(_ s: ARCamera.TrackingState) -> Int32 {
      switch s {
      case .normal:                         return 0
      case .limited(.initializing):         return 1
      case .limited(.excessiveMotion):      return 2
      case .limited(.insufficientFeatures): return 3
      case .limited(.relocalizing):         return 4
      case .notAvailable:                   return 5
      @unknown default:                     return 0
      }
  }
  ```

- In `session(_:didUpdate frame:)` (ViewController.swift:533, already hops to MainActor):
  `odomWorstState = max(odomWorstState, neutralOdomState(frame.camera.trackingState))`.
- In `snap()` (ViewController.swift:251): capture `let state = odomWorstState`, then
  `odomWorstState = neutralOdomState(frame.camera.trackingState)`, and pass `state` into
  the widened `addFrame(...)` below.

## Bridge change (`Sources/LocalizeAR-ObjC`, in the lar-swift repo)

Widen the `CVPixelBuffer` variant that `snap()` calls (and, for parity, the `ARFrame`
variant) with an `odomState` parameter. `LARMapper.h`:

```objc
- (void)addFramePixelBuffer:(CVPixelBufferRef)pixelBuffer
                 intrinsics:(simd_float3x3)intrinsics
                  timestamp:(NSTimeInterval)timestamp
                  transform:(simd_float4x4)transform
                  odomState:(int)odomState
    NS_SWIFT_NAME( addFrame(pixelBuffer:intrinsics:timestamp:transform:odomState:) );
```

`LARMapper.mm` (`addFramePixelBuffer`, currently at line 79) sets one field before
`_internal->addFrame(...)`:

```objc
aFrame.odom_state = odomState;
```

The `ARFrame` convenience `addFrame:transform:` can compute the state itself
(`frame.camera.trackingState`) for callers that don't accumulate — but note that's the
*instantaneous* state, not the interval worst-of. The snap path must use the accumulator.

## C++ `Frame` + `frames.json` (this repo)

`include/lar/mapping/frame.h`:

```cpp
class Frame {
  public:
    size_t id;
    long long timestamp;
    Eigen::Matrix3d intrinsics;
    Eigen::Matrix4d extrinsics;
    int odom_state{0};          // platform-neutral worst OdometryState over the interval
    bool processed{false};
    Frame();
};
// _WITH_DEFAULT so old frames.json (no odom_state) parse, defaulting to 0 = NORMAL/HIGH,
// i.e. old captures keep today's uniform-trust behavior (no regression).
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Frame, id, timestamp, intrinsics, extrinsics, odom_state)
```

(nlohmann_json 3.11.3+ provides `..._WITH_DEFAULT`; the project already requires it.)

Optional readability enum in the same header:

```cpp
enum class OdometryConfidence : int {
  Normal = 0, LimitedInitializing = 1, LimitedExcessiveMotion = 2,
  LimitedInsufficientVisual = 3, Relocalizing = 4, Unavailable = 5,
};
```

## Refiner: tier → information weight (this repo)

The only offline-tunable numbers. In `src/processing/bundle_adjustment.cpp`, add a small
table and apply it in `addOdometry` (line 339), scaling the information matrix already
built at lines 364–369:

```cpp
// Offline-tunable. HIGH = full trust; MEDIUM = downweight; LOW = near-free
// (edge spans a tracking discontinuity, let loop closure absorb it here).
static double odomConfidenceWeight(int odom_state) {
  switch (odom_state) {
    case 0:                 return 1.0;   // HIGH  (NORMAL)
    case 1: case 2: case 3: return 0.25;  // MEDIUM (limited)
    case 4: case 5:         return 0.02;  // LOW   (relocalizing / unavailable)
    default:                return 1.0;
  }
}
```

Then, right before `e->setInformation(info);` (line 369), scale by the destination
frame's tier (the edge runs `frame_id-1 -> frame_id`, and `odom_state` describes the
interval into `frame2`):

```cpp
info *= odomConfidenceWeight(frame2.odom_state);
e->setInformation(info);
```

Two policy variants (offline choice, no recapture):

- **Downweight (default):** scale as above. Keeps the graph fully connected; the robust
  chi² outlier pass (`markOutliers`, line 481) still applies.
- **Break on LOW:** for `odom_state >= 4`, skip `addEdge`/`push_back` entirely so the
  discontinuity contributes no odometry constraint. Use once a vision loop closure or
  other constraint keeps that stretch connected — otherwise it can float.

### Future loop closure

Loop closure reads the **same** `odom_state` tiers to decide where drift correction
lands: it should preferentially deform LOW (then MEDIUM) edges and leave HIGH rigid.
Reuse `odomConfidenceWeight` (or a loop-closure-specific table) so capture stays the
single categorical source of truth.

## Backward compatibility

- Old `frames.json` without `odom_state` → default `0` (HIGH) → identical to today's
  behavior. No migration needed.
- `odomConfidenceWeight(default)` returns `1.0`, so an unrecognized future value also
  degrades to full trust rather than dropping the edge.

## Tunable knobs (all offline, none on device)

- The MEDIUM / LOW weights (`0.25` / `0.02`) and whether LOW breaks vs. downweights.
- Whether to collapse to 2 or 3 tiers.
- Whether loop closure uses a separate weight table from BA.

Device-side capture has **zero** knobs — it emits ARKit/ARCore's own categorical label.
