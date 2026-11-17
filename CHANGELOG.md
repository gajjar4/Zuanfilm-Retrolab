## 0.2.34-alpha — Correct ComfyUI DOM-widget height for subtle region separators

- Corrected the three standard 1 px separators between Bloom/Halation, Anti-AI/Sabattier, Light Leak and Soft Focus/Vignette regions. The divider elements were already styled at 1 px, but ComfyUI assigned their custom DOM widgets its default 50 graph-unit minimum height, which rendered as thick horizontal bands at normal canvas zoom.
- Added explicit `getMinHeight: () => 1` and `getMaxHeight: () => 1` options to the three separator DOM widgets. Also constrained each divider element to a 1 px height with no padding or border, retaining the existing transparent vertical spacing around the rule.
- Extended `addDOMWidgetAt` to accept optional widget options without changing defaults for any other DOM widget.
- Preserved the intentional 2 px white divider above Film Grain / Procedural Texture, the subtle 1 px rule below Emulsion Response, all other layout spacing, node geometry behavior, widget names/order, localization, photographic processing, CUDA execution, Emulsion Response and diagnostic report output.
- Validation: frontend regression test asserts 1 px element bounds and explicit 1-unit ComfyUI DOM-widget min/max sizing for all three separators; full Python suite, frontend interaction smoke test, JavaScript syntax and Python compilation validated.

## 0.2.33-alpha — Emulsion Response control

- Added **Emulsion Response**, a creative multiplier for the selected Color Film Emulsion profile, with a range of **0.50×–3.00×**, 0.01 steps and a default of **1.00×**. The control is positioned directly beneath the emulsion selector.
- The response scales the selected profile's complete tonal and color deviation from the image entering the emulsion stage in linear RGB: `L_out = L_in + response × (L_profile − L_in)`. At 1.00×, the additional blend branch is skipped so the established profile output is preserved exactly. At 0.50×, the profile deviation is softened; above 1.00×, it is amplified up to 3× before the existing output conversion and gamut bounds.
- Neutral remains an exact identity bypass at every Emulsion Response value. The control does not generate grain or alter Film Grain, Procedural Texture, White Balance, Anti-AI, Exposure, Bloom, Halation, Light Leak, Sabattier, Soft Focus or Vignette.
- Added `emulsion_response` as an append-only backend widget to preserve existing workflow widget indices. Existing workflows default to 1.00× and retain their previous emulsion appearance. The setting is included in the execution report and applied to the full-resolution pipeline and After composition preview. The Grain & Texture Detail preview remains independent of the emulsion stage.
- Added localized label and hover guidance for all ten interface languages. Preview settings clamp finite values to 0.50–3.00 and safely normalize invalid values to 1.00.
- Validation: **120 / 120 Python tests passed**; frontend interaction smoke test passed, including 1.00× exact baseline, range and default, Spanish localization, callback / report setting synchronization, Reset to Defaults and Grain & Texture Detail independence. JavaScript syntax and Python compilation passed. Actual ComfyUI visual confirmation and physical NVIDIA host validation of this new control remain pending.

## 0.2.32-alpha — Consistent region separators and distinct Grain/Texture preview boundary

- Removed redundant horizontal borders from the Bloom/Halation, Anti-AI/Sabattier, Light Leak and Soft Focus/Vignette containers. Each main region boundary now uses one consistent 1 px separator instead of stacking a section border with a separate divider.
- Standard separators use a single 1 px line with uniform spacing across the main photographic regions.
- Emphasized the boundary above the Film Grain / Procedural Texture region with one 2 px white line. This deliberately distinguishes the lower Grain & Texture Detail region, which uses its own native-resolution preview, from the main composition-preview controls above it.
- Preserved the internal divider above the native Grain & Texture Detail crop, since it separates the detail image from its own controls rather than duplicating a main-region boundary.
- No changes to node geometry, control positions, widget names/order, parameter values, localization, preview generation, photographic processing, CUDA execution or diagnostic report output.
- Validation: **115 / 115 Python tests passed**; frontend interaction smoke test passed, including assertions for exactly one standard separator at each primary boundary, no redundant section borders and the 2 px white Grain/Texture boundary. JavaScript syntax and Python compilation passed. Visual confirmation in the user's ComfyUI host remains pending.

## 0.2.31-alpha — Compact preview identity header and localized color-picker hover help

- Moved the existing `@zuanfilm / RETROLAB / ANALOG FILM INSPIRED` lockup into the preview toolbar, centered between the Live Preview controls and locale selector. Shortened the decorative side rules to compact fixed-width guides, removing the separate full-width brand strip and reducing vertical space above the composition preview.
- Preserved the Live Preview / Before-After control group on the left and the language selector / Reset to Defaults group on the right. The wordmark remains accessible with the same identity label and visual typography.
- Fixed native Halation Color and Leak Color picker hover descriptions so the picker, row and label refresh from the node's currently selected language. The previous implementation captured English tooltip text during node creation and reused it after locale changes.
- Added frontend regression checks for the three-column toolbar arrangement, shortened decorative rules and localized native color-picker hover descriptions across all ten interface languages.
- Preserved node geometry behavior, all creative controls, backend widget names/order, photo processing, Color Film Emulsion popup, Light Leak layout, Grain/Texture blend modes, CUDA execution and the diagnostic report socket.
- Validation: **115 / 115 Python tests passed**; frontend interaction smoke test passed, including all ten locales and color-picker tooltip updates. JavaScript syntax and Python compilation passed. Actual ComfyUI visual confirmation remains pending.

## 0.2.30-alpha — Emulsion popup, procedural texture headroom and Film Grain blend modes

- Replaced the native Color Film Emulsion popup with a localized custom listbox portal anchored below the selector. The 51-entry Neutral + 50-profile atlas opens downward in a viewport-level scrollable menu so it does not expand over the live preview. Selection still updates the canonical `color_emulsion` widget; Escape/outside interaction closes the menu, list scrolling remains available, and page/canvas scrolling dismisses the anchored popup.
- Increased Procedural Texture `noise_amount` maximum from 1.0 to **2.0**. Existing values from 0–1 retain the previous strength mapping; the extended range provides up to twice the former maximum contribution. Default remains zero.
- Added the append-only `grain_blend_mode` backend input and a visible **Grain Blend Mode** selector. It offers the same twelve modes as Texture Blend Mode and defaults to **Additive**, preserving the established grain result. The selector is placed beneath Color Grain in a shared row aligned with Texture Blend Mode.
- Extracted shared signed-layer compositing into `core/blending.py` and used it for both Procedural Texture and Film Grain. The Film Grain and Procedural Texture selections remain independent; all calculations remain PyTorch-native and float32.
- Preserved existing grain/noise seeds, texture kernels, grain generation, image-processing order, CUDA execution manager, CUDA compatibility preflight, execution report, output contract, all pre-existing widget indices and node geometry behavior. New `grain_blend_mode` is appended to the backend widget contract.
- Validation: **115 / 115 Python tests passed**; frontend interaction smoke test passed, including popup opening below the trigger, scroll dismissal/retention, 51 emulsion choices, twelve Grain Blend Mode options, aligned blend selectors and all ten locales. JavaScript syntax and Python compilation passed. Real ComfyUI visual confirmation remains pending.

## 0.2.29-alpha — Compact two-column Light Leak controls

- Reorganized the six existing Light Leak controls into a single two-column, three-row DOM layout: Mode / Position, Exposure / Seed, and Radius / Color.
- Replaced the three visible native Light Leak numeric widgets with compact translated DOM range controls while keeping the canonical backend widgets hidden, serialized, and in their original order.
- Preserved all existing widget names, defaults, min/max/step values, callback behavior, deterministic seed behavior, color normalization, workflow serialization, processing mathematics, and report output.
- Reset to Defaults and workflow restoration now synchronize all six controls through the combined Light Leak control group. All ten interface languages continue to translate the visible labels and selectors.
- No change to the initial node sizing behavior; the previously accepted first-load narrow width behavior remains untouched.

## 0.2.28-alpha — One-time zoom-aware initial node width

- Replaced the undersized fixed 640 graph-unit first-load width with a one-time initial sizing step in ComfyUI's extension-level `nodeCreated` lifecycle hook, after node construction and Retrolab DOM-widget creation.
- Targets a comfortable approximately 480 CSS-pixel starting presentation using the active canvas zoom, with graph-space bounds of 640–1600 units to avoid excessive world-space dimensions at very low zoom. The canvas zoom is sampled once only.
- No persistent minimum-width behavior was introduced: no `setSize`, `computeSize`, `onResize`, canvas draw or animation-frame interception. Manual resizing remains unrestricted; saved workflow geometry is restored by the normal LiteGraph configure path.
- CUDA execution, compatibility preflight, diagnostic `report` output, photographic algorithms, creative controls and serialized widget order are unchanged.
- Frontend regression coverage verifies initial width at multiple zoom levels, the graph-space upper bound, no repeated resizing after user adjustment, and normal workflow-size restoration.

## 0.2.27-alpha — Stable initial node geometry and unrestricted resizing

- Replaced the zoom-derived visible-width enforcement with a one-time **640 LiteGraph graph-unit initial width** applied when a new Retrolab node is created. The initial size is fixed in graph coordinates and does not change with canvas zoom.
- Removed the persistent `computeSize`, `setSize`, `onResize`, lifecycle, canvas `drawNode()` and animation-frame width guards. They could force a very large graph-space width at certain zoom levels and override user resizing.
- Manual node resizing is now unrestricted after creation. Saved workflow dimensions are preserved when LiteGraph restores them after construction; existing workflows are not automatically enlarged or rewritten.
- No creative controls, widget order, preview behavior, output contract, CUDA execution, diagnostic report or photographic processing code changed.
- Frontend regression coverage confirms the one-time initial width, zoom-independent geometry, normal `setSize` / `onResize` behavior, workflow-size restoration, and no geometry mutation during legacy or Vue-node drawing.
- Validation: **108 / 108 Python tests passed**; frontend interaction smoke test, JavaScript syntax and Python compilation passed. Physical ComfyUI host confirmation of the corrected node geometry remains pending.

## 0.2.26-alpha — CUDA compatibility preflight and first-load node width correction

- Added cached CUDA operation compatibility preflight before CPU-origin image transfer. The probe exercises representative float32 math, int64 deterministic indexing, convolution, max pooling, replicate padding, FFT, interpolation and `erf`; a failed probe selects CPU before image transfer, and failures are retried on later executions rather than cached.
- Added compatibility-preflight status to the structured `report` output, including successful cache hits and failure details.
- Restored the node's 800 CSS-pixel visible width floor. Added a per-canvas `drawNode()` guard that enforces the zoom-adjusted graph-space width immediately before LiteGraph renders each Retrolab node, including Vue-node rendering paths that bypass per-node foreground callbacks. Existing geometry/lifecycle guards remain as secondary protection.
- No user-facing hardware controls, additional dependencies, photographic algorithm changes or backend input/widget order changes.
- Validation: 108 / 108 Python tests passed; frontend interaction smoke test and JavaScript syntax check passed. Physical CUDA preflight and ComfyUI visual first-load confirmation remain pending on the user's host.

# Changelog

## 0.2.25-alpha — Structured execution report output

- Appended a ComfyUI `STRING` output named **report** after the existing `IMAGE` output. The IMAGE output remains at index 0 and existing creative widgets, serialized widget order, preview UI and image-processing behavior are unchanged.
- Added a structured JSON execution report (`zuanfilm.retrolab.execution-report.v1`) so host validation can be performed by connecting one text socket and sharing its contents. It records source/output dimensions, batch/channels/dtypes/devices, active effects, planner decision and reason, planned versus actual processing device, CUDA availability and diagnostic errors, GPU name, CUDA runtime, total/free VRAM and reservation estimate when available, PyTorch allocator/global-memory snapshots, full-resolution pipeline time and megapixel throughput, all creative settings used, Python/PyTorch/platform versions, CPU thread configuration and any full-batch CPU fallback after CUDA OOM. The report explicitly excludes preview-cache creation and other graph nodes from its processing timer.
- Report telemetry is best-effort and cannot interrupt a successful image execution. The estimated working set is labelled as an estimate; allocator and free-VRAM values are labelled as before/after snapshots rather than a measured per-node peak. Identity bypass reports explicitly state that hardware probing was skipped.
- Added `process_batch_with_report` for report-aware execution while preserving the existing tensor-only `process_batch` API without report telemetry overhead.
- No new dependencies, user-facing hardware controls, frontend panels, automatic log files or changes to photographic engines were introduced.
- Automated validation: **105 / 105 Python tests passed** in both the working tree and a clean extracted archive; frontend interaction smoke test passed; JavaScript syntax and Python compilation passed. ZIP integrity passed with 35 members, no bytecode/cache payloads and SHA-256 verification. Actual CUDA hardware and ComfyUI host validation remain pending.

## 0.2.24-alpha — Invisible automatic execution and first-load width enforcement

- Added `core/execution.py` with an internal `ExecutionPlan`, effect-aware workload thresholds, conservative per-frame float32 working-set estimates, free-VRAM admission checks and CUDA allocation-failure detection. The node exposes no CPU/CUDA selector, device option, performance mode or VRAM control.
- CPU-origin images with sufficiently large workloads can be processed one RGB frame at a time on CUDA when a compatible CUDA runtime is available and free memory exceeds the estimated working set plus a reservation of at least 1 GiB or 10% of total VRAM. Simple workloads use an 8 MP threshold; workloads with spatial effects use a 2.5 MP threshold. Smaller workloads, unavailable CUDA and failed preflight remain on CPU.
- Preserved the existing photographic processing engines, fixed finishing order and float32 internal calculations. Additional channels are copied directly; CPU-origin results return to CPU. A recognized CUDA allocation failure discards the incomplete batch and retries the whole batch on CPU. For CUDA-origin input, this exceptional fallback returns CPU output.
- Replaced per-frame list accumulation followed by `torch.stack()` with a preallocated output tensor populated sequentially, reducing avoidable batch-output peak memory.
- Strengthened first-load node width enforcement by wrapping LiteGraph `setSize` requests, retaining the 512 CSS-pixel minimum across construction, restoration and resize, and reapplying the floor across multiple render frames after ComfyUI's delayed geometry writes. No visible node controls or serialized widget order changed.
- Added automated tests for CPU-only selection, small-workload CPU routing, CUDA eligibility, VRAM pressure, area-based memory estimates, whole-batch OOM retry, non-OOM error propagation, identity bypass and LiteGraph `setSize` regression.
- Validation: 101 / 101 Python tests passed; frontend interaction smoke test passed; JavaScript syntax and Python compilation passed. CUDA hardware execution, real ComfyUI first-load geometry and performance benchmarks remain pending host validation.

## 0.2.23-alpha — Pre-release layout and Exposure refinement

- Replaced the unreliable static graph-space width hint with a **512 CSS-pixel minimum visible width**, converted into LiteGraph units using the current canvas zoom. Enforcement now runs through measured sizing, interactive resize, node prototype callbacks, and ComfyUI's extension-level `nodeCreated` and `loadedGraphNode` hooks after construction and workflow restoration.
- Tightened the White Balance instrument grouping by removing the oversized minimum width from the value readouts and reducing the wheel/readout gap to 1 px. The grouped wheel/readouts remain right-aligned within the White Balance column.
- Preserved the approved negative Exposure response exactly. Positive Exposure now uses a 2.5-power ease-in in linear light, slowing the low and middle positive range while retaining the +100 endpoint at 95% of the remaining distance to white.
- Restored the requested public-facing README introduction and updated Exposure documentation to distinguish the negative and positive response curves.
- Added regression tests for zoom-aware node width across creation, graph insertion, resize and workflow restoration, plus the more gradual positive Exposure midpoint.

## 0.2.22-alpha — First-load layout, Exposure and README refinement

- Raised the node's minimum width to **800 px** and enforce that width after both node creation and workflow configuration, preventing restored narrow dimensions from collapsing the two-column controls.
- Anchored the White Balance wheel/readout group to the right edge of its column, with a compact 3 px gap between the wheel and its live values. The assembly uses intrinsic readout width rather than stretching the values across the remaining column.
- Added **Exposure** directly below Anti-AI **Strength**. The slider ranges from -100 to +100 (default 0), with an exact bypass at zero. In linear light, -100 retains 5% of source intensity and +100 moves 95% of the remaining distance toward white. Intermediate values follow a continuous linear response. Exposure is processed after Anti-AI and before Bloom so the downstream light effects respond to its tonal adjustment.
- Added translated Exposure labels and hover descriptions in all ten interface languages. The Exposure backend widget is appended after the existing controls to preserve serialized workflow positions.
- Reworked the root README as evergreen v1.0 documentation: removed the product-identity paragraph, decorative maker-mark description, Anti-AI UI-layout note, obsolete Gabor migration sentence, White Balance calibration aside and detailed frontend inventory from Installation. Iteration history remains in this Changelog.
- Automated validation: **89 / 89 Python tests passed**; the frontend interaction smoke test verifies 800 px width enforcement on creation and workflow restore, compact right-aligned White Balance layout, Exposure control placement and localization, and widget synchronization. JavaScript syntax and Python compilation passed. The clean extracted archive passes all tests, ZIP integrity and SHA-256 verification. ComfyUI host validation remains pending; 0.2.20-alpha is the latest user-host-verified baseline.

## 0.2.21-alpha — Localized hover guidance and texture compositing

- Localized descriptive hover guidance for every parameter across all ten interface languages. Native ComfyUI widget metadata, custom DOM controls, labels, selectors and color-picker controls resolve their hover descriptions against the active locale.
- Added a **Texture Blend Mode** selector with twelve choices: Additive, Multiply, Screen, Overlay, Soft Light, Hard Light, Color Dodge, Color Burn, Lighten, Darken, Difference and Exclusion. **Additive** remains the default to preserve the established 0.2.20 procedural texture response. The new backend widget is appended to preserve previously saved workflow widget positions.
- Changed **Texture Scale** minimum from 0.25 to **0**. Zero is valid and uses the finest supported internal sampling interval without division by zero; the default remains 1.0 and the maximum remains 4.0.
- Renamed the inspection panel to **Grain & Texture Detail · Center Crop 240 px**, localized in all ten interface languages. It displays Procedural Texture followed by Film Grain on a native-resolution centered crop.
- Enforced a **512 px minimum initial node width**, including measured width and workflow restore, so the first-load interface opens at a usable width.
- Automated validation: **84 / 84 Python tests passed**; frontend interaction smoke test passed for all 12 blend modes, blend/preview synchronization, zero-scale input, localized hover text and 512 px node width. JavaScript syntax and Python compilation passed. ComfyUI host validation is pending; 0.2.20-alpha remains the latest user-confirmed host baseline.

## 0.2.20-alpha — Control clarity and photographic imperfection atlas

- Added descriptive hover guidance to all visible parameters, including native ComfyUI widgets. The Anti-AI group heading remains **ANTI-AI**, while its individual slider label is now **Strength**. Renamed **Neutral WB** to **Base WB** to communicate that the button returns the source white-balance wheel to zero without auto-analysis; renamed **Leak Mode** to **Light Leak Mode**.
- Moved **Reset to Defaults** directly below the language selector. The reset button remains available in all ten interface languages.
- Changed both Light Leak Seed and Noise Seed to integer sliders spanning 0–999; defaults remain 27 and 73 respectively. Their deterministic roles remain independent.
- Removed **Gabor Texture** and expanded Procedural Texture to twelve selectable families. Added **Silver Halide Clumps**, **Emulsion Mottle**, **Film Scratches**, **Dust & Fibers** and **Paper Foxing**. Paper Foxing includes clustered stains, pores and a restrained warm ochre discolouration. Old saved Gabor selections migrate to Emulsion Mottle.
- Added **Texture Scale** (0.25–4.0, default 1.0), which adjusts procedural feature size relative to the source dimensions. Larger values create broader marks; smaller values create finer texture while preserving source-coordinate crop alignment.
- Updated all ten language dictionaries for the expanded texture atlas and revised labels. Existing control/widget order remains append-only and the earlier 0.2.19 processing defaults are preserved.
- Automated validation: **79 / 79 Python tests passed**; frontend interaction smoke test passed, including all twelve texture choices, Noise Seed range callbacks, Reset placement, tooltips, revised labels and ten-language coverage. JavaScript syntax and Python compilation passed. ComfyUI host validation is pending.

## 0.2.19-alpha — Anti-AI normalization and deterministic procedural texture

- Added **Anti-AI** correction as a 0–100% slider (default 0%) immediately ahead of the creative light/film sequence. The deterministic global transform gently reduces red/orange chroma, softens contrast, lifts the lower toe and broad midtones, and rolls highlights more softly. It uses smooth tonal and hue weighting, has no spatial/temporal seed, and bypasses exactly at zero.
- Added a separate **Procedural Texture** stage and selector with eight 2D image-space families: Gaussian, Value fBm, Gradient fBm, Simplex fBm, Ridged fBm, Worley Microcells, Domain Warp and Gabor Texture. Noise Amount ranges 0–1 (default 0); Noise Seed ranges 0–2,147,483,647 (default 73).
- Noise Seed is independent of Leak Seed and the fixed internal Film Grain seed. Noise Texture and Film Grain remain separate processing stages; Procedural Texture is applied after Vignette and immediately before Film Grain.
- Reorganized the visible controls: Bloom occupies the left column and Halation the right column so Halation Color stays on the right; Anti-AI occupies the left column opposite Sabattier; Film Grain and Procedural Texture appear as paired controls. Existing Sabattier and color controls remain independent.
- Composition Preview bypasses Procedural Texture and Film Grain to avoid exaggerating texture at reduced resolution. The native-resolution Grain Detail crop applies Procedural Texture first and Film Grain second, aligned with absolute source-pixel coordinates. Full-resolution IMAGE output applies both texture stages.
- Added all new controls to the ten-language frontend, appended backend widgets after the existing controls, and updated README, architecture and validation documentation.
- Automated validation: **77 / 77 Python tests passed**; frontend interaction smoke test passed, including all eight texture options, Anti-AI slider updates, separate seed behavior, two-column control placement, preview bypass/detail behavior and localization. JavaScript syntax and Python compilation passed. The clean extracted ZIP also passed the full 77-test Python suite, frontend interaction test, syntax checks and checksum verification. ComfyUI host validation is pending.

## 0.2.18-alpha — Emulsion / White Balance column separation and alignment

- Added a vertical divider between the Color Film Emulsion and White Balance regions, using the same 1 px border treatment and half-column geometry as the Halation / Bloom separator below.
- Reworked both parameter columns to share a common 82 px control-row centerline. The Emulsion dropdown is vertically centered within its left-side row; the Neutral WB button and White Balance wheel are centered on the same line in the right-side row.
- Right-aligned the wheel/readout group within the White Balance column while retaining the 3 px wheel-to-readout gap and right-aligned signed values.
- Kept the section headings top-aligned and White Balance right-aligned. The new vertical separator spans the heading and control area to communicate that Emulsion selection and White Balance correction are distinct parameter groups.
- Frontend-only layout refinement. No processing, White Balance math, wheel behavior, preview APIs, backend inputs or serialized widget order changed.
- Automated validation: 70 / 70 Python tests pass; frontend interaction tests verify the top divider matches the Halation / Bloom divider, both columns share the same grid and control-row centerline, and the wheel/readout group is right-aligned. JavaScript syntax and Python compilation pass. ComfyUI host visual validation remains pending.

## 0.2.17-alpha — White Balance / Emulsion alignment polish

- Renamed the White Balance reset action to **Neutral WB**, with localized labels in all ten supported languages.
- Right-aligned the WHITE BALANCE heading to match the right-aligned temperature/tint axis labels and readouts.
- Set the Emulsion / White Balance parent grid to top alignment so the Emulsion label and White Balance label share a baseline, and the Emulsion dropdown aligns with the Neutral WB button.
- Reduced the wheel-to-readout gap from 7 px to 3 px so the disc and its live values read as one compact instrument.
- Preserved White Balance math, interaction, pointer geometry, Neutral recentering, emulsion selection, preview behavior, backend inputs and workflow widget order.
- Automated validation: 70 / 70 Python tests pass; frontend interaction tests verify translated Neutral WB labels, heading alignment, dropdown/button baseline, and tightened wheel/readout spacing. JavaScript syntax and Python compilation pass. ComfyUI host validation remains pending.

## 0.2.16-alpha — White Balance control alignment and exact neutral centering

- Reorganized the Emulsion / White Balance row. The Neutral reset button now occupies the leading position in the White Balance control row, aligned vertically with the Emulsion dropdown; the wheel and live axis/value readouts are grouped to its right and right-aligned.
- Moved the WHITE BALANCE heading to the top of its right-hand column, aligned with the COLOR FILM EMULSION heading.
- Corrected the movable puck anchor to use `translate(-50%, -50%)` with border-box sizing. The zero-temperature/zero-tint state now places its center exactly on the wheel's geometric center instead of leaving a one-pixel offset from negative puck margins.
- Reduced the White Balance wheel/rim diameter to 82 px / 74 px for the more compact three-part layout. The center reference marker and smaller graphite-grey movable puck remain visually distinct.
- Preserved the temperature/tint ranges, pointer mapping, wheel response, keyboard controls, Neutral reset, ten translations, White Balance image transform, emulsion selector, preview behavior and backend widget contracts.
- Automated validation: 70 / 70 Python tests pass; frontend interaction tests verify the right-aligned readouts, Neutral button alignment, exact centered puck at zero, cursor/reference size distinction, pointer updates, keyboard behavior and translations. JavaScript syntax and Python compilation pass. ComfyUI host validation remains pending.

## 0.2.15-alpha — White Balance wheel readout layout and maker tagline refinement

- Updated the centered maker-mark descriptor from **ANALOG IMAGE WORKSHOP** to **ANALOG FILM INSPIRED**, including its accessible identity label.
- Reorganized the White Balance wheel into a compact horizontal composition: the color disc stays on the left, while the translated Cool/Warm and Green/Magenta labels and their live signed temperature/tint values sit in a right-hand readout column.
- Kept the **WHITE BALANCE** heading and **Neutral** reset button together in the footer row beneath the wheel/readouts.
- Reduced the movable wheel cursor to a smaller graphite-grey puck with a soft silver rim, separating it visually from the fixed center reference marker.
- Preserved all temperature/tint values, pointer geometry, wheel mapping, keyboard controls, neutral reset, image processing and workflow widget contracts. Frontend-only refinement; no new backend inputs.
- Automated validation: 70 / 70 Python tests pass; frontend interaction test verifies readout placement, live values, smaller graphite cursor, tagline and the ten-language translations. JavaScript syntax and Python compilation pass. ComfyUI host validation remains pending.

## 0.2.14-alpha — White Balance wheel visual refinement

- Moved the translated **WHITE BALANCE** heading from above the wheel into a compact footer row beside the **Neutral** reset button, making the label and its primary action read as one control group.
- Added a double graduated metallic rim around the color field: a bright white/silver outer chamfer, a graphite/silver inner bevel, subtle inner highlights and a more dimensional center reference puck. The color field, puck movement, temperature/tint math, pointer mapping, keyboard behavior and backend values are unchanged.
- Kept the implementation frontend-only inside the existing non-serialized Emulsion / White Balance row. No new node inputs, workflow widget values, preview API behavior or image-processing changes.
- Extended frontend tests to verify the heading/reset alignment, both gradient rim layers, wheel dimensions, ten-language heading translation and unchanged pointer/keyboard/reset behavior.
- Automated validation: **70 / 70 Python tests passed**; frontend interaction smoke test passed; JavaScript syntax and Python compilation passed. Clean extracted archive validation is required before distribution. ComfyUI visual confirmation remains pending.

## 0.2.13-alpha — Half-width Emulsion Atlas, Resolve-inspired White Balance wheel and creator mark

- Reworked the Emulsion Atlas row into two equal-width columns directly below Live Preview. The 50-look Color Film Emulsion selector occupies the left half; a two-axis White Balance gain wheel occupies the right half, keeping the preview visible while the emulsion list is open.
- Added a circular, draggable White Balance puck. Vertical travel adjusts Cool ↔ Warm; horizontal travel adjusts Green ↔ Magenta. Arrow keys adjust the values, Shift increases keyboard step size, and the Neutral button returns both axes to zero. All wheel labels and accessibility text are translated in the ten supported languages.
- Added `white_balance_temperature` and `white_balance_tint` as two FLOAT backend inputs appended after `color_emulsion`, both defaulting to 0.0 and bounded to -1…+1. Existing widget positions before the new fields remain unchanged.
- Added a deterministic linear-RGB primary correction before Bloom and the creative finishing pipeline. Channel gains are normalized with Rec.709 luminance to avoid introducing a global exposure offset. The model is a creative temperature/tint wheel inspired by color-grading interfaces, not Kelvin calibration, camera RAW white balance or a DaVinci Resolve color-science reproduction.
- Enlarged the centered creator maker mark from `ZUANFILM` to `@zuanfilm`, preserving its position and warm microtype treatment.
- White Balance affects full-resolution output and the processed After composition preview. It does not affect the untouched Before view or the dedicated Film Grain Detail crop.
- Automated validation: **70 / 70 Python tests pass**; the frontend interaction smoke test verifies the half-width two-column row, wheel pointer and keyboard interaction, Neutral reset, all ten translations, creator mark and serialized widget ordering. Preview tests confirm White Balance changes Composition but not Grain Detail. JavaScript syntax and Python compilation pass. ComfyUI host validation remains pending.

## 0.2.12-alpha — 50-look Color Film Emulsions Atlas

- Added a public Color Film Emulsions Atlas selector as the first parameter directly below Live Preview. The dropdown is grouped into 10 Cinema / Motion Picture, 10 Instant, 10 Slide / Reversal and 20 35mm Color Negative looks, plus a separate Neutral identity entry.
- Added 50 independent creative profiles with exposure, smooth tonal-curve, saturation, channel-balance, shadow-tint and highlight-tint parameters. Neutral bypasses the engine exactly.
- The profile is applied to the full-resolution IMAGE output and the processed Composition After preview. The untouched Before preview remains the original source; Grain Detail remains a Film-Grain-only inspection.
- Emulsion processing does not generate or modify Film Grain, Halation, Bloom, Light Leak, Soft Focus or Vignette settings. Film Grain remains separately controlled.
- Appended the backend `color_emulsion` combo after all existing inputs so the prior 0.2.11 backend widget positions remain stable. The visible grouped dropdown is a non-serialized DOM control immediately beneath the preview and is localized in all ten interface languages.
- Documented profile families, entries, processing order and calibration limitations. The stock names are used as reference vocabulary; profiles are original creative interpretations, not spectrally/sensitometrically calibrated or exact chemical/scanner reproductions.
- Automated validation: **66 / 66 Python tests passed**; frontend interaction tests verify 50 visible profiles, category counts, translated category headings, Neutral reset, selection-to-backend synchronization and widget-order preservation. Preview tests verify emulsions change Composition but not Grain Detail. JavaScript syntax and Python compilation passed. The clean extracted ZIP passed the full Python and frontend suites, compilation, ZIP integrity and SHA-256 verification. ComfyUI host validation remains pending.

## 0.2.11-alpha — Before / After preview and two-column Halation / Bloom controls

- Added a compact Before / After switch beside the Live Preview heading. Before displays the untouched reduced input proxy; After displays the current processed composition.
- The original comparison proxy is returned only on the first preview request for each preview token and retained as a browser-side data URL. Subsequent slider updates omit it from the request/response, refreshing only the processed composition and Grain Detail images. The full-resolution source is never retained by the preview cache or sent to the browser.
- Grouped Halation and Bloom into a compact two-column DOM control panel. Halation Amount, Ring Radius and Halation Color share the left column; Bloom Amount, Threshold and Radius occupy the right column.
- Replaced the separate visible Halation Color row and stacked native Halation/Bloom sliders with non-serialized DOM controls. The existing backend widgets remain in their original positions; no new ComfyUI inputs or workflow widget-order changes were introduced.
- Preserved the dedicated native-resolution Film Grain Detail crop and the grain-free reduced Composition preview. The actual full-resolution IMAGE output still processes Film Grain normally.
- Updated all ten languages for the Before / After state labels and Halation / Bloom group headings; comparison state is frontend-only and resets to After when a new source token is received.
- Automated validation: **59 / 59 Python tests passed**; frontend interaction smoke test passed, including original-source caching, Before / After toggling, translated state labels, two-column Halation / Bloom controls and backend widget-order preservation. JavaScript syntax and Python compilation passed. ComfyUI host validation remains pending.

## 0.2.10-alpha — Grain-focused preview and Light Leak control refinements

- The reduced Live Preview composition now bypasses Film Grain. Its downscaled image remains clean and representative of Bloom, Halation, Light Leak, Sabattier, Soft Focus and Vignette without magnifying grain texture; the dedicated native-resolution Grain Detail crop remains the only preview of Film Grain. The actual full-resolution ComfyUI IMAGE output continues to apply Film Grain normally.
- Changed Film Grain factory defaults to Size `0.3` and Color Grain `0.09`.
- Renamed the public selector and backend field from `leak_character` to `leak_mode`, with the visible label **Leak Mode**. Renamed the 135 perforation mode to **Holga 135**; the former `135 Sprocket` string remains accepted as an internal compatibility alias.
- Renamed `pattern_seed` to `leak_seed` and changed its default to `27`. Leak Seed now affects only procedural Light Leak geometry. Film Grain uses a separate fixed internal seed, so changing Leak Seed cannot alter grain texture.
- Changed Highlight Confinement default to `0.09` and Leak Position default to `Bottom`.
- Updated all ten localized selector labels and Leak Seed labels.
- This is a development-stage interface/schema update; saved workflows using the former `leak_character` or `pattern_seed` input names need to be rebuilt or have those fields migrated.
- Automated validation: **58 / 58 Python tests passed**; frontend interaction smoke test, JavaScript syntax check and Python compilation passed. The expanded preview regression verifies that Grain Amount, Grain Size and Color Grain affect only the native Grain Detail PNG, never the reduced Composition PNG. Clean extracted archive validation is required before distribution; real ComfyUI host validation remains pending.

## 0.2.9-alpha — RetroLab centered vintage wordmark

- Added a centered decorative ZUANFILM RETROLAB wordmark inside the preview panel, following the two-layer identity treatment accepted in RenderQn: the native ComfyUI title remains functional, while the body carries a quiet centered brand lockup with horizontal guide rules and a short vertical registration mark.
- Styled RETROLAB with a retro slab-serif font stack (`Rockwell Extra Bold`, `Rockwell`, `Bookman Old Style`, `Georgia`) and restrained warm photographic-paper tones. ZUANFILM and the workshop descriptor use tracked monospaced microtype.
- The treatment is entirely frontend-only and remains inside the existing non-serialized composition-preview DOM widget. It adds no ComfyUI inputs, does not alter node registration, does not modify saved widget order, and does not change processing or preview rendering.
- No font files, external font downloads or new dependencies are introduced; rendering uses fonts available on the host with local fallbacks.


## 0.2.8-alpha — Zuanfilm Retrolab identity and public release preparation

- Rebranded the ComfyUI node as **Zuanfilm Retrolab** and moved it to the **Zuanfilm / RetroLab** category. The node class and registration key are now `ZuanfilmRetroLab`.
- Renamed the Python node module and frontend extension to `zuanfilm_retrolab.py` and `zuanfilm_retrolab.js`; renamed the processing settings model to `RetroLabSettings`.
- Moved the preview API to `POST /retrolab/preview` and `DELETE /retrolab/preview/{token}` and renamed the node UI preview token to `retrolab_preview_id`.
- Replaced the public README with a product-focused description, engine/control reference, Mermaid pipeline and preview diagrams, installation instructions, image contract and development test guide.
- Rewrote the architecture and technical-reference documents for the new public identity. Removed unrelated product comparisons and all references to separate confidential image-processing projects.
- This is a development-stage breaking rename. Workflows saved before this rebrand use the previous node type and may need the node re-added and its connections restored. Browser-local language preferences also start fresh under the Retrolab storage key.
- Processing algorithms, effect defaults, image pipeline order, preview behavior and UI layout are unchanged from the 0.2.7-alpha host-verified baseline.

## 0.2.7-alpha — Fine Grain, compact optical controls and layout refinement

- Lowered the public Film Grain Size minimum from 0.7 to 0.1 reference pixels in the node and Live Preview validation. At high-resolution source sizes, this reaches uncorrelated pixel-scale grain instead of forcing the smallest available radius through a blurred field. The Grain Size default remains 1.2; grain amplitude and tonal response are unchanged.
- Raised Vignette Amount maximum from 1.0 to 2.0. The previous 0–1 response is retained, while Amount=2 reaches a three-stop corner exposure reduction. The processing engine and preview-side clamp now share the same 0–2 bounds.
- Removed the visible Mackie Lines slider to reduce node height. The serialized `mackie_lines` widget remains hidden in its original backend position: legacy workflows with a saved nonzero value continue to render, while new workflows and Reset to Defaults use zero.
- Combined Soft Focus and Vignette into a two-column optical-finishing panel with translated group headings and compact sliders. Their four backend widgets remain in their original serialized positions; the visible DOM panel is non-serialized.
- Relocated the Leak Color picker to appear immediately below Leak Radius, with no backend widget reordering.
- Preserved the ten-language selector, globe icon, composition preview, bottom Film Grain controls and dedicated Grain Detail crop.
- Automated validation: 55 / 55 Python tests pass; frontend interaction tests verify the two-column layout, slider updates, hidden legacy Mackie field, Leak Color placement, all ten translations and unchanged serialized widget order. JavaScript syntax and compilation checks pass. ComfyUI host validation remains pending.

## 0.2.6-alpha — Vignette and corner-weighted Soft Focus

- Added an independent Vignette engine with Amount (0–1) and Feather (0–1). It applies aspect-correct radial exposure falloff in linear RGB, reaching a maximum of 1.5 stops of corner attenuation while preserving the exact image center.
- Added an independent Soft Focus engine with Amount (0–1) and Radius (1–80 reference pixels at a 1024 px long edge). It computes a resolution-aware Gaussian blur in linear RGB and blends it progressively toward the corners, leaving the center and protected inner region sharp.
- Appended four FLOAT widgets after the existing `halation_color` widget: `soft_focus_amount`, `soft_focus_radius`, `vignette_amount`, and `vignette_feather`. Existing serialized workflow widget positions remain unchanged.
- Added localized labels for all ten supported languages and discreet dividers before the new Soft Focus and Vignette groups. The controls remain ordinary ComfyUI sliders.
- Extended the live composition preview to render both effects; the native Grain Detail preview remains Film Grain-only. Preview-side numeric limits mirror the node input ranges.
- Fixed processing order: Bloom → Halation → Light Leak → Sabattier → Soft Focus → Vignette → Film Grain, keeping Grain as the final crisp texture layer. Both new effects default to Amount=0 and preserve exact identity bypass when disabled.
- Added tests for center preservation, corner blur, vignette exposure, feather response, amount/radius clamping, combined batch processing, alpha preservation, preview response and saved-widget ordering.
- Automated validation: 53 / 53 Python tests and the mocked frontend interaction test pass; JavaScript syntax check passes. CUDA and ComfyUI host validation remain pending.

## 0.2.5-alpha — Color picker alignment, globe language selector and Russian localization

- Right-aligned both Halation Color and Leak Color rows as compact label / hex value / color-picker groups, placing each label directly beside its picker cluster at the right edge of the node.
- Replaced the visible “Language” text with a 🌐 globe icon. The control retains translated accessible names and a translated tooltip for assistive technology and hover discovery.
- Replaced Indonesian with Russian and set the exact language order: English, Simplified Chinese, Japanese, Spanish, Portuguese, French, Russian, German, Hindi and Arabic.
- Added Russian translations for the preview state, node controls, color pickers, Reset to Defaults, Film Grain controls and both Light Leak selectors.
- Kept the language choice browser-local and non-serialized. No ComfyUI backend inputs, values, widget names or serialized widget positions changed.
- Validation: automated Python tests, JavaScript syntax check and mocked frontend tests cover the color-row alignment, globe icon accessibility, exact locale order, Russian UI strings, canonical selector values and original backend widget serialization order. ComfyUI host confirmation remains pending.

## 0.2.4-alpha — Grain headroom, revised defaults and multilingual UI

- Raised Film Grain Amount maximum from 1.0 to 3.0 in the node input, preview settings clamp and bottom DOM slider. The existing per-unit grain response is retained, allowing three times the previous maximum noise amplitude; Amount=3 reaches 0.066 display-referred standard deviation before tonal modulation.
- Changed the default Bloom Threshold to 0.1 and the default Light Leak Color to `#C2AF8E`, including reset behavior and preview normalization.
- Removed the internal “CPU preview” wording and “MAX 512 PX” label from the visible Live Preview UI. Source dimensions remain visible; the 512 px preview proxy remains an internal performance detail.
- Added a browser-local language selector for English, Simplified Chinese, Hindi, Spanish, Arabic, French, Portuguese, Indonesian, Japanese and German. Language choice is persisted in local storage and does not add, reorder or alter serialized ComfyUI widget values.
- Localized visible node labels, color-picker labels, reset action, preview text and Film Grain controls. Light Leak Character and Position use translated DOM option labels while preserving canonical English backend values.
- Preserved the 0.2.3-alpha node socket contract, color picker defaults for Halation, image-processing order, DOM preview architecture, preview endpoints and source/output behavior.
- Automated validation: 47 / 47 Python tests pass; frontend interaction tests exercise all ten locales, canonical selector values, browser-local language persistence, preview PNG rendering, controls and widget-order preservation. Exact ComfyUI host validation is pending.

## 0.2.3-alpha — Halation color, expanded Bloom and preview hierarchy

- Expanded Bloom Amount from 0–2 to 0–3 while retaining the per-unit exposure coefficient; the engine clamps out-of-contract values to 3.0. Changed the factory Bloom Threshold to 0.0; the soft-knee extraction now shortens at low thresholds so pure black never generates artificial light.
- Added a native Halation Color picker using the same frontend pattern as Leak Color. The default `#FF9959` is special-cased to preserve the original `(1.00, 0.32, 0.10)` red-dominant channel response exactly; custom colors select the normalized linear-RGB emission tint.
- Appended the `halation_color` STRING widget after `leak_radius`; all existing serialized widget indices remain unchanged. Reset to Defaults now includes both color values.
- Enlarged the composition preview and moved it to the top of the node. Kept the native 240 × 240 px Grain Detail preview centered at the bottom beneath the Film Grain controls.
- Added discreet, non-serialized separators before Bloom, Sabattier and Light Leak to clarify effect groups.
- Retained the DOM-based preview path, CPU-only preview rendering, preview cache/endpoints, 180 ms debounce, and original IMAGE processing order.
- Automated validation: 45 / 45 Python tests, JavaScript syntax check and mocked frontend interaction tests pass. The test verifies that the added DOM widgets do not alter serialized backend widget order. Exact ComfyUI host validation is pending; 0.2.2-alpha remains the host-verified rollback baseline.

## 0.2.2-alpha — Restore visible DOM Live Preview

- Restored the proven non-serialized DOM Live Preview panel used by 0.2.0-alpha. Both the 512 px maximum composition preview and native 240 × 240 px grain-detail crop are rendered as regular HTML image elements inside the node's DOM widget.
- Removed the canvas-rendered preview header and its custom widget-layout reservation. The 0.2.1 canvas path reserved space but did not display the preview in the user's ComfyUI host; the 0.2.0 DOM approach was visibly confirmed in the supplied reference screenshot.
- Kept the 0.2.1 native FLOAT sliders, bottom Film Grain controls, color picker, Reset to Defaults, 180 ms preview debounce, server preview endpoints, CPU-only preview rendering and original backend widget serialization order.
- No processing algorithms, preview cache contents, output tensors, node sockets or backend widget order changed.
- Added frontend regression checks for the actual DOM preview panel, both returned PNG images becoming visible, status updates, Film Grain control synchronization, reset, token handling and debounced refresh.
- Host validation is still required after installing this candidate; automated frontend tests do not replace visual confirmation in ComfyUI.

## 0.2.1-alpha — Unified sliders and enlarged preview header

- Changed every FLOAT input to ComfyUI's native slider display so Grain, Halation, Bloom, Sabattier and Light Leak numeric controls share the same interaction style as Light Leak Exposure. `leak_seed` remains an integer stepper.
- Moved the visible Film Grain controls to the bottom of the node, immediately below Leak Color and Reset to Defaults. The original backend widgets remain hidden in their original positions and retain their serialized values; the visible range controls write to those same widgets.
- Moved Live Preview from the bottom DOM panel into a dedicated canvas-rendered header region above the controls.
- The composition preview is displayed at its generated maximum 512 px long edge. The native 240 × 240 px grain detail is vertically centered in an adjacent column.
- Set the node's initial width to 800 px to fit the composition and detail previews side by side. The preview region reserves layout space without inserting a widget or changing the serialized backend widget order.
- Preserved the preview cache, endpoints, 180 ms debounce, CPU-only preview rendering, five processing engines, full-resolution IMAGE path, and Reset to Defaults behavior.
- Added automated checks for slider metadata, header layout, native preview dimensions, centered grain detail and bottom Film Grain slider synchronization.
- 41 / 41 Python tests pass; the mocked frontend interaction test passes. The new canvas-header layout still requires validation in the user's ComfyUI host.

## 0.2.0-alpha — Interactive Live Preview

- Added a non-serialized Live Preview panel inside Zuanfilm Retrolab.
- Full composition preview renders at up to 512 px long edge and reflects all five engines.
- Added a native-resolution 240 × 240 px centre crop for Film Grain inspection; grain is processed at source scale and uses the crop's absolute source-pixel coordinates plus a surrounding Gaussian context margin.
- Node execution creates a compact CPU cache containing only the full-frame proxy and grain crop context. The full-resolution source frame is not retained in the cache.
- Added ComfyUI endpoints `POST /retrolab/preview` and `DELETE /retrolab/preview/{token}`. Preview updates are rendered on CPU and return two PNG images; they do not queue the graph or rerun upstream nodes.
- Frontend widget changes refresh previews after a 180 ms debounce. Color-picker changes and Reset to Defaults also refresh the preview.
- Preview cache is bounded to six entries and 30-minute expiry; removing the node attempts to release its cache entry.
- Added source-coordinate support to deterministic grain-field generation for spatially registered detail crops. Crop-local normalization can cause a very small grain-amplitude difference from the full-frame crop, while the pixel-coordinate pattern remains aligned.
- Main full-resolution processing path, output IMAGE tensor, node sockets, and serialized widget order remain unchanged. Preview generation failures fail open and do not interrupt normal image processing.
- Added Python tests for cache sizing, preview render output, settings validation, coordinate alignment and preview route registration, plus a frontend smoke test for reset, token handling, dual preview updates and debounced widget refresh.
- 40 / 40 Python tests pass; frontend interaction smoke test passes. No additional pip dependencies.

## 0.1.5-alpha — Exposure slider visibility and reset action

- Explicitly set Light Leak Exposure to the ComfyUI `display: slider` widget with two-decimal precision, so its fill remains visible across the full 0–10 range, including when the value is exactly 10.0.
- Added a frontend **Reset to Defaults** button beneath the Leak Color picker. It restores every processing widget to the authoritative Python `INPUT_TYPES()` defaults, including color, character, position, exposure, radius and shared seed.
- The reset button is a non-serialized DOM control; no backend input was inserted or reordered, preserving existing workflow widget serialization.
- Reset writes use each widget's callback path and refresh the graph canvas.
- Added tests that the exposure slider is explicitly declared and that frontend reset values match every backend widget default.
- No new Python dependencies.

## 0.1.4-alpha — Border-to-border Light Leak influence

- Added `leak_radius` (0.25–10.0, default 1.0) to expand the inward influence of leaks from the selected border.
- Radius=1.0 preserves the 0.1.3 footprint; larger values scale each character's native spatial falloff / band width and film-base spectral attenuation together. Radius=10 supports border-to-border influence on typical images.
- Increased Light Leak Exposure maximum from 5.0 to 10.0; engine clamps external values to the public 0–10 contract.
- Appended `leak_radius` after the previously appended `leak_color` widget to preserve saved 0.1.3 workflow widget-value order. Existing workflows receive the default radius 1.0.
- Added tests for radius backward compatibility, opposite-border reach across all five characters, deterministic radius clamping and tenfold exposure scaling.
- 30 / 30 CPU tests pass; no additional Python dependencies.

## 0.1.3-alpha — Stronger Light Leak response and simplified character set

- Removed Mechanical Marks from the Light Leak selector and processing implementation.
- Expanded the Light Leak Exposure UI range from 0–2 to 0–5; the linear exposure mapping now honors the full range.
- Retained the existing per-character response coefficients and procedural geometry so increased exposure scales the established looks rather than changing their visual identity.
- Legacy saved `Mechanical Marks` and `Pinhole` values safely resolve to Soft Edge; unknown character values also fail safe.
- Updated CPU tests for the five supported characters, fivefold exposure scaling, and legacy workflow compatibility.
- Updated reference-photo preview, technical documentation, and validation record.
- No additional Python dependencies.

## 0.1.2-alpha — Expanded deterministic Light Leak characters

- Expanded Light Leak to six geometries: Soft Edge, First-Frame Burn, 120 Fat Roll, Light Piping, Holga 135 and Mechanical Marks.
- Added a serializable `#RRGGBB` incident-light color control with a native ComfyUI frontend color picker and a plain-hex fallback; appended the new backend widget after existing inputs to preserve saved widget-value order.
- Added seeded low-frequency boundary deformation, wavy Fat Roll bands, elongated Light Piping channels, and deterministic geometric mechanical marks using PyTorch / the Python standard library only.
- Added a near-white hot entry core to First-Frame Burn while retaining the user-selected incident color through its falloff.
- Preserved old Soft Edge and Sprocket behavior as available modes; legacy Pinhole / unknown mode values still safely fall back to Soft Edge.
- Added tests for all six characters, determinism, seed variation, color-picker hex parsing, custom light spectra and mode differentiation.
- No additional Python dependencies.

## 0.1.1-alpha — Stronger light response and simplified controls

- Doubled the public maximum Amount / Exposure range for Halation, Bloom and Light Leak from 1.0 to 2.0, retaining the same per-unit response coefficients.
- Removed the Anti-halation control; Halation now exposes its full annular response without a second attenuation control.
- Removed Pinhole from the Light Leak selector, retaining Soft Edge and Sprocket.
- Added a compatibility fallback that maps a legacy saved `Pinhole` character value to Soft Edge.
- Updated CPU tests and technical documentation for the simplified interface and doubled response headroom.

## 0.1.0-alpha — Initial implementation

- Added the single `ZuanfilmRetroLab` ComfyUI node.
- Added five independent Torch-only engines: Film Grain, Halation, Bloom, Sabattier and Light Leak.
- Added compact creative controls per engine and one shared deterministic Pattern Seed.
- Added a fixed finishing pipeline: Bloom → Halation → Light Leak → Sabattier → Film Grain.
- Added exact identity bypass when all effect amounts are zero.
- Added batch processing, alpha/extra-channel preservation, input device/dtype preservation and deterministic per-frame grain variation.
- Added 19 CPU unit tests for identity, determinism, effect behavior, bounded output, hue-preserving Sabattier recombination, node registration and batch handling.
- Added 1024², 2048² and 4096² CPU performance smoke tests and documented the outstanding CUDA / ComfyUI host validation.
- No additional pip dependencies.

## 0.2.0-alpha — Interactive Live Preview

- Added a non-serialized Live Preview panel inside Zuanfilm Retrolab.
- Full composition preview renders at up to 512 px long edge and reflects all five engines.
- Added a native-resolution 240 × 240 px centre crop for Film Grain inspection; grain is processed at source scale and uses the crop's absolute source-pixel coordinates plus a surrounding Gaussian context margin.
- Node execution creates a compact CPU cache containing only the full-frame proxy and grain crop context. The full-resolution source frame is not retained in the cache.
- Added ComfyUI endpoints `POST /retrolab/preview` and `DELETE /retrolab/preview/{token}`. Preview updates are rendered on CPU and return two PNG images; they do not queue the graph or rerun upstream nodes.
- Frontend widget changes refresh previews after a 180 ms debounce. Color-picker changes and Reset to Defaults also refresh the preview.
- Preview cache is bounded to six entries and 30-minute expiry; removing the node attempts to release its cache entry.
- Added source-coordinate support to deterministic grain-field generation for spatially registered detail crops. Crop-local normalization can cause a very small grain-amplitude difference from the full-frame result, while the pixel-coordinate pattern remains aligned.
- Main full-resolution processing path, output IMAGE tensor, node sockets, and serialized widget order remain unchanged. Preview generation failures fail open and do not interrupt normal image processing.
- Added Python tests for cache sizing, preview render output, settings validation, coordinate alignment and preview route registration, plus a frontend smoke test for reset, token handling, dual preview updates and debounced widget refresh.
- 40 / 40 Python tests pass; frontend interaction smoke test passes. No additional pip dependencies.
