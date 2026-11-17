## 0.2.34-alpha — Correct minimum sizing for standard region separators

- Python: **120 / 120 tests passed** in the working tree.
- Frontend interaction smoke test verifies that each of the three standard separator elements is constrained to 1 px and its ComfyUI DOM widget explicitly reports `getMinHeight() = 1` and `getMaxHeight() = 1`. This covers the actual layout-sizing contract that previously fell back to ComfyUI's 50 graph-unit DOM-widget minimum.
- The 2 px white divider above Film Grain / Procedural Texture remains unchanged. The subtle border below Emulsion Response remains unchanged. Node sizing and user resizing behavior are untouched.
- JavaScript syntax (`node --check`) and Python compilation passed. The clean extracted archive also passed **120 / 120 Python tests**, the frontend interaction smoke test, JavaScript syntax validation and Python compilation.
- Actual ComfyUI host visual confirmation remains pending; the fix targets the documented ComfyUI DOM-widget min/max-height API.

## 0.2.33-alpha — Emulsion Response

- Python: **120 / 120 tests passed** in the working tree. Coverage includes exact 1.00× profile-output preservation, linear-light deviation scaling at 0.50× / 1.00× / 3.00×, clamping and invalid-value normalization, exact Neutral bypass at all response values, pipeline settings integration, preview normalization, and separation from the Grain & Texture Detail preview.
- Frontend: mocked ComfyUI interaction test passed. It verifies the Emulsion Response control directly below the emulsion selector, range 0.50–3.00, 0.01 step, 1.00 default, value readout, backend callback, localized Spanish label / tooltip and Reset to Defaults synchronization. Backend input is append-only.
- JavaScript syntax (`node --check`) and Python compilation passed.
- CUDA kernels and host performance were not re-benchmarked specifically for this change. The added profile math uses the existing PyTorch linear-RGB path; physical ComfyUI visual confirmation remains pending.

## 0.2.32-alpha — Consistent region separators and emphasized Grain/Texture preview boundary

- Python: **115 / 115 tests passed** in the working tree.
- Frontend interaction smoke test verifies three consistent 1 px primary region separators, removal of redundant section borders from Bloom/Halation, Anti-AI/Sabattier, Light Leak and Soft Focus/Vignette, and one 2 px white border above the Film Grain / Procedural Texture region.
- The distinct upper boundary identifies the lower Grain & Texture Detail region as a separate preview area. The internal divider above the native detail crop remains to separate the crop from its controls.
- JavaScript syntax (`node --check`) and Python compilation passed.
- No node geometry, processing, preview, CUDA, report, parameter or serialized widget-order changes. Visual host confirmation remains pending.

## 0.2.31-alpha — Compact preview header and localized color-picker tooltips

- Python: **115 / 115 tests passed** in the working tree.
- Frontend: interaction smoke test passed for the compact three-column preview toolbar, centered wordmark, shortened decorative guide rules, Live Preview / Before-After controls, locale selector and Reset placement.
- Frontend locale coverage verifies Halation Color and Leak Color native input, row and label hover descriptions refresh in all ten interface languages rather than retaining their creation-time language.
- JavaScript syntax (`node --check`) and Python compilation passed.
- The header consolidation and tooltip correction have not yet received visual confirmation in the user's ComfyUI host. Node geometry and sizing behavior were not changed.

## 0.2.28-alpha — One-time zoom-aware first-load width

- Frontend: moved initial geometry adjustment from the node prototype's `onNodeCreated` callback to ComfyUI's extension-level `nodeCreated` hook, which runs after the constructor and DOM widget setup.
- Initial presentation targets approximately 480 CSS pixels using the active canvas scale sampled once, with a 640–1600 graph-unit bound. This improves first-load readability at zoomed-out workspace scales without installing a persistent size floor.
- No geometry changes occur during later zoom, user resizing, rendering or workflow configuration. A configured saved node size remains authoritative.
- Frontend tests cover canvas scales 0.25, 0.30, 0.5, 0.75, 1.0 and 2.0, maximum graph-width cap, repeated-hook idempotence and unrestricted subsequent manual resizing.

## 0.2.27-alpha — Fixed initial graph-space width and user resizing

- Frontend: new nodes receive one fixed 640 graph-unit width during `onNodeCreated`; the value is independent of canvas zoom. Existing workflow dimensions load unchanged, and later manual resizing is not constrained.
- Removed zoom-derived graph-space width calculations and all persistent sizing interception, including the LiteGraph canvas `drawNode()` wrapper and delayed animation-frame corrections.
- Frontend regression test: at several zoom values, initial node width remains fixed; smaller user `setSize` and `onResize` requests remain intact; a simulated saved workflow width survives `onConfigure`; legacy and Vue-node draw calls do not mutate node geometry.
- Python: **108 / 108 tests passed**. Frontend interaction smoke test passed. JavaScript syntax (`node --check`) and Python compilation passed.
- Requires visual confirmation in the user's ComfyUI environment; no claim of physical host validation is made for this geometry change.

## 0.2.26-alpha — CUDA operation compatibility preflight and canvas-level first-load width guard

- Execution planner: added a small CUDA operation compatibility preflight covering float32 tensor math, int64 indexing arithmetic, convolution, max pooling, replicate padding, FFT, bilinear interpolation and error-function math. Successful checks are cached per PyTorch/runtime/device signature; failures are not cached and select CPU before image transfer. The text report exposes checked/compatible/cache-hit status and error details.
- Frontend: restored an 800 CSS-pixel minimum visible width and added a per-canvas `drawNode()` guard. It uses the active graph canvas zoom immediately before LiteGraph renders each Retrolab node, including the Vue-node renderer path that can bypass `onDrawForeground`. Existing `setSize`, resize, configure, lifecycle and per-node draw safeguards remain in place. No creative controls, backend widget ordering or photographic algorithms changed.
- Frontend regression coverage simulates delayed first-load zoom changes and both legacy and Vue-node rendering paths; the 800 CSS-pixel floor is tested at 0.5× and 0.25× zoom.
- Automated validation: **108 / 108 Python tests passed**; frontend interaction smoke test passed; JavaScript syntax check passed. Validation ran in the CPU-only development environment, so the new preflight's actual CUDA kernels still require confirmation on the user's NVIDIA host.
- ComfyUI visual confirmation of the canvas-level first-load width fix remains pending.

# Validation Record

## 0.2.30-alpha — Emulsion popup placement, expanded procedural texture and Film Grain blending

- Python: **115 / 115 tests passed** in the working tree.
- Frontend: interaction smoke test passed for the viewport-level Color Film Emulsion popup opening below its trigger, 51 options, selection synchronization, outside-scroll dismissal, continued scrolling inside the popup, all ten interface languages, and the two aligned Film Grain / Texture Blend Mode selectors.
- Python coverage verifies the shared twelve-mode grain compositing contract, deterministic/bounded/distinct output across modes, Additive backward-compatible default, pipeline integration and the expanded Noise Amount maximum of 2.0.
- JavaScript syntax (`node --check`) and Python compilation passed.
- The Emulsion popup, wider Noise Amount range and Grain Blend Mode have not yet received visual confirmation in the user's ComfyUI host. Existing node geometry behavior is unchanged.

## 0.2.29-alpha — Compact Light Leak layout

- Python: **108 / 108 tests passed** in the working tree.
- Frontend interaction smoke test validates the six Light Leak controls as a two-column / three-row grid, exact Mode / Position, Exposure / Seed, Radius / Color ordering, translated selectors, numeric ranges, callback writes, reset synchronization and preserved backend widget ordering.
- JavaScript syntax (`node --check`) and Python compilation passed.
- No initial node sizing changes; the 0.2.28 sizing behavior is intentionally untouched. Actual ComfyUI host visual confirmation remains pending.

## 0.2.25-alpha — Structured execution report output

- Added an append-only ComfyUI `STRING` output named `report` at output index 1; the existing `IMAGE` output remains at index 0. No creative controls, widgets, frontend panels, dependencies or automatic file-writing behavior were added.
- Added `process_batch_with_report`, preserving the tensor-only `process_batch` API without telemetry overhead for callers that do not request a report.
- Report schema `zuanfilm.retrolab.execution-report.v1` records image tensor metadata, resolution, active stages, planner decision, selected and actual processing devices, CUDA availability and query errors, GPU identity, preflight VRAM/estimated working set/reserve, before/after allocator and global-memory snapshots when available, wall time, average time per frame, software environment and CUDA OOM-to-CPU recovery details.
- Report telemetry is best-effort and does not alter image processing. The working-set value is an estimate, and memory counters are snapshots rather than a measured per-node peak. Identity bypass is reported without probing hardware.
- Automated validation: **105 / 105 Python tests passed** in both the working tree and a clean extracted archive; frontend interaction smoke test passed; JavaScript syntax and Python compilation passed. ZIP integrity passed with 35 members, no bytecode/cache payloads and SHA-256 verification.
- Actual NVIDIA CUDA execution and the user's ComfyUI host validation remain necessary; this development environment uses CPU-only PyTorch.

## 0.2.24-alpha — Automatic execution and first-load node sizing

- Python: **101 / 101 tests passed**, including the 90-test 0.2.23 baseline and eleven execution-planner/recovery tests. Coverage includes CPU-only PyTorch selection, avoiding transfer for small/simple work, CUDA routing with mocked available VRAM, refusal under memory pressure, memory estimate scaling, identity bypass, whole-batch CPU retry after simulated CUDA out-of-memory, non-OOM error propagation and channel/dtype/shape preservation.
- Frontend: mocked ComfyUI interaction test passed. Added regression coverage proving that a legacy narrow width passed through LiteGraph `setSize` is clamped both after construction and after workflow restoration. Existing zoom-aware width, widget-order, localization, preview and control tests also pass.
- JavaScript syntax (`node --check`) and Python compilation passed.
- Package validation: clean extraction passed, ZIP integrity passed, **40 archive members** verified, no generated bytecode/cache files included, and SHA-256 verification passed.
- **Not yet host-verified:** actual CUDA processing and speed on NVIDIA hardware; live ComfyUI GPU memory pressure behavior; CPU-only ComfyUI execution; first-load node geometry in the actual frontend. The local test environment uses CPU-only PyTorch, so CUDA selection and fallback are covered by mocks rather than real GPU execution.

## 0.2.23-alpha — Zoom-aware first-load layout, positive Exposure refinement and evergreen README

- Python: **90 / 90 tests passed** in both the working tree and a clean extracted copy of the release archive. Coverage includes Exposure identity bypass, preserved negative darkening, exact 5% / 95% endpoints, the positive 2.5-power ease-in midpoint, monotonicity, bounded output, preview setting clamping and the append-only node contract.
- Frontend: mocked ComfyUI interaction test passed in both trees. It verifies a 512 CSS-pixel minimum visible width converted into graph units at 0.5× and 0.25× zoom, ComfyUI `nodeCreated` and `loadedGraphNode` hooks, graph insertion, interactive resize and workflow restoration. It also verifies the right-anchored White Balance wheel/readout group with a 1 px gap, Exposure placement and ten-language labels/tooltips.
- JavaScript syntax (`node --check`) and Python compilation (`compileall`) passed in the clean extracted archive.
- ZIP integrity passed: **38 members**, with no Python bytecode or cache directories in the distributable. SHA-256 is recorded alongside the release package.
- README restores the requested public introduction and keeps iteration-specific history in `CHANGELOG.md`.
- Actual ComfyUI host validation is pending. **0.2.20-alpha remains the latest user-host-verified baseline.**

## 0.2.21-alpha — Localized parameter guidance and texture compositing

- Python: **84 / 84 tests passed**, including twelve deterministic blend modes, Additive backward-default behavior, invalid-mode fallback, zero-scale safety/repeatability and bounded output.
- Preview: tests confirm blend-mode changes affect the centered native 240 × 240 detail crop only; the reduced Composition preview remains independent of Procedural Texture and Film Grain. Texture Scale accepts zero and stays bounded through request normalization.
- Frontend: mocked ComfyUI interaction test passed, including twelve selectable blend modes, backend and preview synchronization, scale minimum zero, language-aware hover descriptions for all ten locales (including Base WB), translated blend labels/options, and a 512 px initial and measured node width. Backend workflow widget ordering remains append-only.
- JavaScript syntax (`node --check`) and Python compilation (`compileall`) passed.
- Actual ComfyUI host validation identified a narrow first-load layout and excessive spacing around the White Balance wheel. The 0.2.23-alpha changes raise the width floor and tighten the wheel/readout group. The latest fully confirmed host baseline remains 0.2.20-alpha.

## 0.2.20-alpha — Control clarity and photographic imperfection atlas

- Python: **79 / 79 tests passed**, including the twelve deterministic procedural texture families, seed sensitivity and independence, texture-scale feature changes, warm Paper Foxing discolouration, Light Leak / Noise Seed bounds, Anti-AI behavior and full tensor contract preservation.
- Preview: tests verify Texture Scale is applied in the native-resolution detail renderer, procedural texture remains aligned to absolute source coordinates, and the reduced Composition preview still bypasses Procedural Texture and Film Grain.
- Frontend: mocked ComfyUI interaction test passed, including per-parameter hover details, Anti-AI's concise Strength label, Base WB and Light Leak Mode labels, both 0–999 integer seed sliders, all twelve localized texture choices, Texture Scale, Reset directly below the language selector and all ten language variants.
- JavaScript syntax (`node --check`) and Python compilation (`compileall`) passed.
- Clean extracted ZIP validation: ZIP integrity, member count, no bytecode/cache payloads, complete Python/frontend suites and SHA-256 verification passed.
- The user subsequently installed and confirmed 0.2.20-alpha working in the actual ComfyUI host; it became the host-verified baseline before 0.2.21-alpha.

## 0.2.19-alpha — Anti-AI normalization and deterministic procedural texture

- Python: **77 / 77 tests passed**, including Anti-AI tone/chroma behavior, exact zero bypass, all eight original procedural noise kernels, deterministic and seed-sensitive outputs, Noise Seed independence from Leak Seed and Film Grain, and full pipeline/device/channel preservation.
- Preview: tests verify that Anti-AI updates the processed Composition but not Grain Detail; Noise Texture and Noise Seed update the native-resolution detail crop while the reduced Composition preview bypasses Procedural Texture and Film Grain; absolute-coordinate crop alignment is tested.
- Frontend: mocked ComfyUI interaction test passed, including Bloom-left / Halation-right layout, Halation Color in the right column, Anti-AI-left / Sabattier-right layout, all eight original Noise Texture options, Noise Amount / Noise Seed callbacks, and all ten interface languages.
- 0.2.19-alpha was host-tested successfully before 0.2.20-alpha; see the following release record for the later confirmed baseline.

## 0.2.18-alpha — Emulsion / White Balance column separation and alignment

- Python: 70 / 70 tests passed.
- Frontend: mocked ComfyUI interaction test passed, including the Emulsion / White Balance vertical divider, exact match with the Halation / Bloom divider treatment, shared 82 px control-row centerline, vertically centered dropdown / Neutral WB / wheel, and right-aligned wheel/readout group.
- JavaScript syntax and Python compilation passed.
- No processing changes, White Balance math changes or backend widget changes.
- Actual ComfyUI visual validation is pending.

## 0.2.17-alpha — White Balance / Emulsion alignment polish

- Python: 70 / 70 tests passed.
- Frontend: mocked ComfyUI interaction test passed, including the localized Neutral WB button, right-aligned White Balance heading, shared heading baseline, Emulsion dropdown / Neutral WB control alignment and 3 px wheel-to-readout spacing.
- JavaScript syntax and Python compilation passed.
- No processing changes or backend widget changes.
- Actual ComfyUI visual validation is pending.


## 0.2.14-alpha — White Balance wheel visual refinement

- Frontend tests verify the White Balance heading shares a single horizontal footer with the Neutral reset button rather than occupying a detached heading row.
- Tests verify the wheel is wrapped in two independent graduated metallic layers: an outer white/silver chamfer and an inner silver/graphite bevel. The color field remains a circular interactive application element.
- Pointer mapping, keyboard arrows, Shift-step precision, Home-to-Neutral behavior, backend callbacks, localization and the underlying temperature/tint values remain unchanged.
- No ComfyUI input fields, serialized widgets, node processing code, preview cache or HTTP endpoints changed.
- Automated validation: **70 / 70 Python tests passed**; frontend interaction smoke test passed; JavaScript syntax and Python compilation passed. Clean extracted archive validation remains required before distribution. ComfyUI visual host confirmation is pending.

## 0.2.13-alpha — Emulsion / White Balance paired row

- The Color Film Emulsions selector occupies the left half of the first parameter row below Live Preview; the Resolve-inspired two-axis White Balance gain wheel occupies the right half. The selector is constrained to its half-width column so an open atlas list does not consume the full node width.
- White Balance wheel coordinates are appended after `color_emulsion` as hidden numeric backend widgets. Frontend tests verify pointer mapping, keyboard adjustment, neutral reset, accessible names, translations across all ten languages and preservation of the existing serialized widget order.
- The primary correction is tested for exact neutral identity, cool/warm and green/magenta response, finite bounded output and Rec.709-normalized luminance behavior. Preview regression verifies that white-balance changes affect the After Composition PNG but do not affect the original Before PNG or Grain Detail PNG.
- The centered maker mark is now `@zuanfilm`, larger than the previous `ZUANFILM` microtype while preserving its established position.
- Automated validation: **70 / 70 Python tests passed**; frontend interaction smoke test passed; JavaScript syntax and Python compilation passed. Clean extracted archive validation remains required before distribution. ComfyUI host validation is pending.

## 0.2.12-alpha — Color Film Emulsions Atlas

- Verified exactly 50 unique profiles and 51 backend choices including Neutral, grouped 10 / 10 / 10 / 20 across Cinema, Instant, Slide and 35mm Color Negative.
- Node contract tests verify the atlas input is appended after the previous backend inputs, with Neutral as default. Frontend tests verify the selector is placed directly after the Preview panel and before the existing processing controls.
- All ten language variants provide a translated atlas label, Neutral option and four translated family headings. Film stock names remain in their common product naming to support recognition and discovery.
- Core tests verify strict Neutral identity, finite bounded output for all 50 profiles, pairwise-distinct profile output, deterministic behavior and separation from Grain / Leak Seed. Preview tests verify selected emulsions affect Composition while leaving Grain Detail unchanged.
- Automated validation: **66 / 66 Python tests passed**; frontend interaction smoke test passed; JavaScript syntax and Python compilation passed. Clean extracted archive validation passed, including the complete Python and frontend suites, JavaScript syntax validation, Python compilation, ZIP integrity and SHA-256 verification. Live ComfyUI host validation remains pending before publishing.

## 0.2.11-alpha — Before / After preview and two-column Halation / Bloom

- The Live Preview Before / After switch displays the untouched reduced input proxy or the currently processed composition.
- The original proxy PNG is requested once per preview token, cached only in browser memory and omitted from subsequent slider-update requests. Tests verify original-source response omission when the frontend already has the proxy.
- Halation Amount, Ring Radius and Halation Color are presented in one column; Bloom Amount, Threshold and Radius are presented in the second column. The five original backend widget positions remain unchanged and hidden behind the custom DOM controls.
- Film Grain remains bypassed in the reduced Composition preview and isolated in the native-resolution Grain Detail view. The full-resolution ComfyUI IMAGE output still includes Film Grain.
- No new serialized ComfyUI widgets or input fields were added. The comparison switch is frontend-only.

**Automated validation:** 59 / 59 Python tests passed; the mocked frontend interaction test passed, including Before / After source switching, first-request original proxy transfer, translated labels, two-column control updates and serialized widget-order preservation. JavaScript syntax validation and Python compilation passed. The clean extracted distributable archive also passes the complete Python and frontend suites, compilation, ZIP integrity and SHA-256 verification. A live ComfyUI host check is still required for actual layout, switch interaction and visual comparison.

## 0.2.10-alpha — Grain-focused preview and Light Leak control refinements

- Reduced composition preview explicitly calls the processing pipeline with Film Grain disabled. Tests verify that changing Grain Amount, Grain Size or Color Grain cannot change the composition PNG. The native-resolution Grain Detail crop remains a Grain-only view, while full-resolution ComfyUI output continues to apply Film Grain in the final pipeline position.
- Film Grain defaults: Size `0.3`, Color Grain `0.09`. Highlight Confinement defaults to `0.09`.
- Public node controls are `leak_mode` (visible label Leak Mode), `leak_seed` (default `27`) and `leak_position` (default Bottom). The available Holga perforation mode is named **Holga 135**.
- Leak Seed affects only procedural Light Leak geometry. Film Grain uses an independent fixed seed and deterministic frame variation.
- Automated validation: **58 / 58 Python tests passed**; frontend interaction smoke test passed; JavaScript syntax and Python compilation passed. The preview regression suite verifies that Grain Amount, Grain Size and Color Grain affect only the native Grain Detail PNG, never the reduced Composition PNG. Real ComfyUI host validation remains pending.

## 0.2.9-alpha — Decorative RetroLab wordmark

- Frontend-only identity update in the existing composition-preview DOM panel.
- Centered ZUANFILM / RETROLAB lockup, fine horizontal guide rules, vertical registration tick and workshop descriptor.
- Native ComfyUI title, node key, backend inputs, saved widget order, image processing, preview API and language preference behavior remain unchanged.
- Uses a host-local retro slab-serif font stack; no external font or dependency.
- Validation: frontend smoke test, Python unit suite, JavaScript syntax, Python compilation and clean archive extraction.

## 0.2.29-alpha — Compact Light Leak layout

- Python suite: pending clean-run result for this archive.
- Frontend interaction smoke test: validates the six Light Leak controls as a two-column / three-row grid, the exact Mode / Position, Exposure / Seed, Radius / Color ordering, translated selectors, numeric ranges, callback writes, reset synchronization and preserved backend widget ordering.
- No initial node sizing changes; the existing 0.2.28 first-load sizing behavior is intentionally untouched.
- Actual ComfyUI host visual confirmation remains pending.

## Development test commands

From the repository root:

```bash
python -m unittest discover -s tests -v
node tests/test_frontend.mjs
```

Automated tests do not replace a real ComfyUI host run. GPU performance, frontend appearance and actual node registration must be verified in the target ComfyUI installation.


## 0.2.16-alpha — White Balance layout and neutral-center correction

- Python: 70 / 70 tests passed.
- Frontend: mocked ComfyUI interaction smoke test passed, including Emulsion/White Balance layout, right-aligned readouts, Neutral placement, exact zero-offset puck centering, pointer and keyboard interaction, ten translations and backend widget ordering.
- JavaScript syntax check and Python compilation passed.
- Changes are frontend-only; processing mathematics, backend inputs and workflow widget order are unchanged.
- Actual ComfyUI visual validation is pending.

- Automated validation for this iteration: **108 / 108 Python tests passed**; frontend interaction smoke test passed, JavaScript syntax check passed, and Python compilation passed. Validation ran in the CPU-only development environment; the sizing correction still requires visual confirmation in the user's ComfyUI host.
