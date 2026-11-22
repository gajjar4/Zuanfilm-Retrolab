import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    addDOMWidgetAt, applyInitialNodeWidth, contentHeight, createPreviewPanel, createVideoPlayerControls, deletePreviewToken,
    deleteVideoPreviewToken, installDarkroomStyle, makeText, pumpVideoRender, renderActiveCompositionPreview, resetVideoState,
    restartVideoRender, updateRangeFill,
} from "./zuanfilm_retrolab.js";

const NODE_NAME = "ZuanfilmRetroLabFX";
const FX_ENDPOINT = "/retrolab/fx/preview";
const LOOKS_FILE = "retrolab/fx_looks.json";
const PREVIEW_DEBOUNCE_MS = 160;
const CATEGORY_ORDER = ["Blur", "Light", "Retro", "Print", "Glitch", "Adjust", "Dither"];

const FX_STYLE = `
.zrl-fx-sec { display: flex; justify-content: space-between; align-items: center; padding: 2px 2px 0; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--zrl-muted); }
.zrl-fx-looks { display: flex; gap: 6px; overflow-x: auto; padding: 2px 0 4px; scrollbar-width: thin; }
.zrl-dr .zrl-fx-look { flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; gap: 3px; padding: 4px !important; font-size: 10px; color: var(--zrl-muted) !important; background: transparent !important; border: 1px solid transparent !important; position: relative; }
.zrl-dr .zrl-fx-look.on { border-color: var(--zrl-ac) !important; color: var(--zrl-text) !important; }
.zrl-fx-thumb { width: 52px; height: 52px; border-radius: 6px; background: #1b1917 center/cover no-repeat; }
.zrl-fx-look .zrl-fx-del { position: absolute; top: -2px; right: -2px; width: 16px; height: 16px; line-height: 14px; border-radius: 50%; font-size: 11px; background: #3a3631; color: var(--zrl-text); display: none; }
.zrl-fx-look:hover .zrl-fx-del { display: block; }
.zrl-dr .zrl-fx-save { flex: 0 0 64px; height: 70px; border-style: dashed !important; color: var(--zrl-muted) !important; font-size: 10.5px; }
.zrl-fx-save-input { flex: 0 0 120px; height: 30px; align-self: center; padding: 0 8px; border-radius: 8px; border: 1px solid var(--zrl-ac); background: var(--zrl-card); color: var(--zrl-text); font-size: 11px; }
.zrl-fx-stack { display: flex; flex-direction: column; gap: 4px; padding: 2px 0; }
.zrl-fx-row { display: grid; grid-template-columns: 14px 14px minmax(0, 1fr) 92px 40px 18px; gap: 7px; align-items: center; height: 34px; padding: 0 8px; border-radius: 9px; background: var(--zrl-card); border: 1px solid transparent; cursor: pointer; }
.zrl-fx-row.sel { border-color: var(--zrl-ac); background: #34302b; }
.zrl-fx-row.off { opacity: 0.45; }
.zrl-fx-row.drop { box-shadow: 0 -2px 0 var(--zrl-ac); }
.zrl-fx-grip { color: var(--zrl-muted); cursor: grab; letter-spacing: -2px; user-select: none; }
.zrl-dr .zrl-fx-eye { width: 12px; height: 12px; padding: 0 !important; border-radius: 50% !important; border: 0 !important; background: var(--zrl-ac) !important; }
.zrl-dr .zrl-fx-row.off .zrl-fx-eye { background: #4a453f !important; }
.zrl-fx-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 12px; }
.zrl-fx-name small { color: var(--zrl-muted); font-size: 10px; margin-left: 6px; }
.zrl-dr .zrl-fx-row select { height: 22px !important; font-size: 10.5px !important; padding: 0 4px !important; border-radius: 6px !important; }
.zrl-fx-op { text-align: right; color: var(--zrl-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.zrl-dr .zrl-fx-x { padding: 0 !important; border: 0 !important; background: transparent !important; color: var(--zrl-muted) !important; font-size: 15px; }
.zrl-dr .zrl-fx-add { height: 32px; border-style: dashed !important; color: var(--zrl-muted) !important; background: transparent !important; }
.zrl-fx-picker { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 4px; padding: 8px; border-radius: 10px; background: #1e1c1a; border: 1px solid var(--zrl-line); }
.zrl-fx-picker h5 { grid-column: 1 / -1; margin: 6px 2px 0; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--zrl-muted); font-weight: 600; }
.zrl-dr .zrl-fx-picker button { height: 28px; padding: 0 8px !important; text-align: left; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.zrl-fx-params { display: flex; flex-direction: column; gap: 4px; padding: 6px 4px 4px; border-top: 1px solid var(--zrl-line); }
.zrl-fx-sl { display: grid; grid-template-columns: 112px minmax(0, 1fr) 46px; gap: 10px; align-items: center; height: 24px; }
.zrl-fx-sl label, .zrl-fx-field label { color: #cdc5b8; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.zrl-fx-sl output { text-align: right; font-size: 11px; font-variant-numeric: tabular-nums; }
.zrl-fx-field { display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 10px; align-items: center; min-height: 28px; }
.zrl-fx-field input[type=text] { height: 26px; padding: 0 8px; border-radius: 8px; border: 1px solid var(--zrl-line); background: var(--zrl-card); color: var(--zrl-text); font-size: 11px; }
.zrl-fx-colour { display: flex; align-items: center; gap: 8px; font-family: monospace; font-size: 11px; color: var(--zrl-muted); }
.zrl-fx-chips { display: flex; flex-wrap: wrap; gap: 5px; padding: 2px 0; }
.zrl-dr .zrl-fx-chip { height: 26px; padding: 0 9px !important; font-size: 11px; display: inline-flex; align-items: center; gap: 6px; }
.zrl-dr .zrl-fx-chip.on { border-color: var(--zrl-ac) !important; background: #34302b !important; }
.zrl-fx-swatch { display: inline-flex; border-radius: 3px; overflow: hidden; }
.zrl-fx-swatch b { width: 8px; height: 13px; display: block; }
.zrl-fx-wheels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.zrl-fx-wheel { display: flex; flex-direction: column; align-items: center; gap: 5px; font-size: 10px; color: var(--zrl-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.zrl-fx-disc { position: relative; width: 84px; height: 84px; border-radius: 50%; cursor: crosshair; touch-action: none;
  background: radial-gradient(circle, rgba(35,34,32,0.9) 0 14%, rgba(35,34,32,0) 68%), conic-gradient(from 90deg, #e7b56a, #e07a7a, #b77ad6, #6f9fe0, #6fd0c2, #9fd06f, #e7b56a); }
.zrl-fx-disc i { position: absolute; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%; background: #fff; border: 2px solid #1c1712; pointer-events: none; }
.zrl-fx-wheel input[type=range] { width: 84px; }
.zrl-fx-handles { position: absolute; inset: 0; z-index: 2; pointer-events: none; }
.zrl-fx-handles svg { width: 100%; height: 100%; overflow: visible; }
.zrl-fx-handles .zrl-fx-h { pointer-events: all; cursor: grab; }
.zrl-fx-hist { position: absolute; left: 8px; top: 40px; width: 120px; height: 44px; border-radius: 8px; z-index: 3; }
.zrl-fx-hint { color: var(--zrl-muted); font-size: 10.5px; padding: 2px; }
`;

const BUILTIN_LOOKS = {
    "Anamorphic": [["curves", { contrast: 0.25 }], ["anamorphic_streaks", { amount: 0.8 }], ["chromatic_aberration", { amount: 2 }], ["tilt_shift", { blur: 5, band: 0.32 }]],
    "Tokyo VHS": [["chromatic_aberration", { amount: 5 }], ["vhs", {}], ["zoom_blur", { amount: 0.05 }, { opacity: 0.6 }]],
    "Riso poster": [["curves", { contrast: 0.4 }], ["risograph", {}]],
    "Game Boy": [["dither", { algorithm: "Bayer 4×4", palette: "Game Boy", pixel_size: 4 }]],
    "Mac 1-bit": [["dither", { algorithm: "Atkinson", palette: "1-bit", pixel_size: 2, contrast: 0.3 }]],
    "PICO-8": [["dither", { algorithm: "Floyd–Steinberg", palette: "PICO-8", pixel_size: 4 }]],
    "Arcade CRT": [["dither", { algorithm: "Bayer 2×2", palette: "EGA", pixel_size: 3 }], ["crt", {}]],
    "Speed": [["directional_blur", { angle: 170, distance: 60 }], ["chromatic_aberration", { amount: 3 }], ["film_frame", {}]],
    "Glitch": [["pixel_sort", {}], ["block_shift", {}], ["chromatic_aberration", { amount: 8 }]],
    "Polaroid": [["curves", { contrast: -0.1, black: 0.08 }], ["gradient_map", { mix: 0.25 }], ["film_frame", { style: "Polaroid", text: "" }]],
    "Newsprint": [["halftone", { mode: "CMYK", cell: 7 }]],
    "Dreamy": [["lens_blur", { radius: 8, focus: 0.45 }], ["prism", { mix: 0.3 }]],
};

let catalogPromise = null;
function loadCatalog() {
    catalogPromise ||= api.fetchApi("/retrolab/fx/catalog").then((response) => response.json());
    return catalogPromise;
}

function el(tag, className = "", text = "") {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text) element.textContent = text;
    return element;
}

function newLayer(catalog, type, params = {}, extra = {}) {
    const defaults = {};
    for (const param of catalog.effects[type].params) defaults[param.key] = param.default;
    return { type, on: true, opacity: 1, blend: "Normal", ...extra, params: { ...defaults, ...params } };
}

function lookStack(catalog, recipe) {
    return recipe.map(([type, params, extra]) => newLayer(catalog, type, params, extra || {}));
}

function stackWidget(node) {
    return node.widgets?.find((widget) => widget.name === "fx_stack");
}

function readStack(node) {
    try {
        const parsed = JSON.parse(stackWidget(node)?.value || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
        return [];
    }
}

function commit(node, { rerender = true } = {}) {
    const fx = node._fx;
    const widget = stackWidget(node);
    widget.value = JSON.stringify(fx.stack);
    widget.callback?.(widget.value, app.canvas, node);
    if (rerender) {
        renderStack(node);
        renderParams(node);
    }
    drawHandles(node);
    schedulePreview(node);
    node.setDirtyCanvas?.(true, true);
}

function schedulePreview(node, immediate = false) {
    clearTimeout(node._fx.timer);
    const run = () => (node._zrlVideo ? restartVideoRender(node) : requestPreview(node));
    if (immediate) run();
    else node._fx.timer = setTimeout(run, PREVIEW_DEBOUNCE_MS);
}

async function renderStackImage(token, stack, includeOriginal) {
    const response = await api.fetchApi(FX_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, stack, include_original: includeOriginal }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result?.error || `Preview failed (HTTP ${response.status}).`);
    return result;
}

async function requestPreview(node) {
    const ui = node._zrlPreviewUI;
    const token = node._zrlPreviewToken;
    if (!token) return;
    const requestId = (node._fx.requestId || 0) + 1;
    node._fx.requestId = requestId;
    ui.status.textContent = "Rendering…";
    try {
        const result = await renderStackImage(token, node._fx.stack, !node._zrlPreviewOriginalSrc);
        if (token !== node._zrlPreviewToken || requestId !== node._fx.requestId) return;
        if (result.original_png) node._zrlPreviewOriginalSrc = `data:image/png;base64,${result.original_png}`;
        node._zrlPreviewProcessedSrc = `data:image/png;base64,${result.full_png}`;
        ui.fullImage.style.visibility = "visible";
        renderActiveCompositionPreview(node);
        ui.status.textContent = `Source: ${result.source_size[0]} × ${result.source_size[1]}`;
        ui.fullImage.onload = () => { drawHandles(node); drawHistogram(node); };
        if (!node._fx.thumbToken || node._fx.thumbToken !== token) renderLookThumbs(node, token);
    } catch (error) {
        if (requestId === node._fx.requestId) ui.status.textContent = `Preview unavailable: ${error?.message || error}`;
    }
}

// ---------- looks ----------

async function loadUserLooks(node) {
    try {
        const response = await api.getUserData(LOOKS_FILE);
        if (response.ok) node._fx.userLooks = await response.json();
    } catch (_) {
        // No saved looks yet.
    }
    renderLooks(node);
}

async function saveUserLooks(node) {
    await api.storeUserData(LOOKS_FILE, node._fx.userLooks, { overwrite: true, stringify: true, throwOnError: false });
}

function allLooks(node) {
    const catalog = node._fx.catalog;
    const looks = [["Clean", []]];
    for (const [name, recipe] of Object.entries(BUILTIN_LOOKS)) looks.push([name, lookStack(catalog, recipe)]);
    for (const [name, stack] of Object.entries(node._fx.userLooks)) looks.push([name, stack, true]);
    return looks;
}

async function renderLookThumbs(node, token) {
    node._fx.thumbToken = token;
    for (const [name, stack] of allLooks(node)) {
        if (node._zrlPreviewToken !== token) return;
        try {
            const result = await renderStackImage(token, stack, false);
            node._fx.thumbs.set(name, `data:image/png;base64,${result.full_png}`);
            const thumb = node._fx.lookEls.get(name);
            if (thumb) thumb.style.backgroundImage = `url(${node._fx.thumbs.get(name)})`;
        } catch (_) {
            return;
        }
    }
}

function renderLooks(node) {
    const fx = node._fx;
    if (!fx.catalog) return;
    const host = fx.ui.looks;
    host.replaceChildren();
    fx.lookEls = new Map();
    for (const [name, stack, isUser] of allLooks(node)) {
        const button = el("button", `zrl-fx-look${fx.look === name ? " on" : ""}`);
        button.type = "button";
        const thumb = el("span", "zrl-fx-thumb");
        if (fx.thumbs.has(name)) thumb.style.backgroundImage = `url(${fx.thumbs.get(name)})`;
        fx.lookEls.set(name, thumb);
        button.append(thumb, el("span", "", name));
        button.addEventListener("click", () => {
            fx.stack = JSON.parse(JSON.stringify(stack));
            fx.selected = Math.max(0, fx.stack.length - 1);
            fx.look = name;
            renderLooks(node);
            commit(node);
        });
        if (isUser) {
            const remove = el("span", "zrl-fx-del", "×");
            remove.title = "Delete look";
            remove.addEventListener("click", (event) => {
                event.stopPropagation();
                delete fx.userLooks[name];
                fx.thumbs.delete(name);
                saveUserLooks(node);
                renderLooks(node);
            });
            button.append(remove);
        }
        host.append(button);
    }
    const save = el("button", "zrl-fx-save", "＋ Save look");
    save.type = "button";
    save.addEventListener("click", () => {
        const input = el("input", "zrl-fx-save-input");
        input.placeholder = "Look name";
        save.replaceWith(input);
        input.focus();
        input.addEventListener("keydown", (event) => {
            event.stopPropagation();
            if (event.key === "Escape") renderLooks(node);
            if (event.key !== "Enter" || !input.value.trim()) return;
            const name = input.value.trim().slice(0, 32);
            fx.userLooks[name] = JSON.parse(JSON.stringify(fx.stack));
            fx.look = name;
            fx.thumbs.set(name, node._zrlPreviewProcessedSrc || "");
            saveUserLooks(node);
            renderLooks(node);
        });
        input.addEventListener("blur", () => renderLooks(node));
    });
    host.append(save);
}

// ---------- stack ----------

function renderStack(node) {
    const fx = node._fx;
    const host = fx.ui.stack;
    host.replaceChildren();
    fx.ui.count.textContent = `${fx.stack.length} layer${fx.stack.length === 1 ? "" : "s"}`;
    fx.stack.forEach((layer, index) => {
        const effect = fx.catalog.effects[layer.type];
        const row = el("div", `zrl-fx-row${index === fx.selected ? " sel" : ""}${layer.on ? "" : " off"}`);
        const grip = el("span", "zrl-fx-grip", "⋮⋮");
        grip.draggable = true;
        const eye = el("button", "zrl-fx-eye");
        eye.type = "button";
        eye.title = layer.on ? "Hide layer" : "Show layer";
        const name = el("span", "zrl-fx-name", effect.name);
        name.append(el("small", "", effect.category));
        const blend = el("select");
        for (const mode of fx.catalog.blend_modes) {
            const option = el("option", "", mode);
            option.value = mode;
            blend.append(option);
        }
        blend.value = layer.blend;
        const opacity = el("span", "zrl-fx-op", `${Math.round(layer.opacity * 100)}%`);
        const remove = el("button", "zrl-fx-x", "×");
        remove.type = "button";
        remove.title = "Remove layer";
        row.append(grip, eye, name, blend, opacity, remove);
        row.addEventListener("click", (event) => {
            if (event.target.closest("button, select")) return;
            fx.selected = index;
            renderStack(node);
            renderParams(node);
            drawHandles(node);
        });
        eye.addEventListener("click", () => { layer.on = !layer.on; fx.look = null; commit(node); });
        remove.addEventListener("click", () => {
            fx.stack.splice(index, 1);
            fx.selected = Math.min(fx.selected, fx.stack.length - 1);
            fx.look = null;
            commit(node);
        });
        blend.addEventListener("change", () => { layer.blend = blend.value; commit(node, { rerender: false }); });
        grip.addEventListener("dragstart", (event) => { event.dataTransfer.setData("text/plain", String(index)); });
        row.addEventListener("dragover", (event) => { event.preventDefault(); row.classList.add("drop"); });
        row.addEventListener("dragleave", () => row.classList.remove("drop"));
        row.addEventListener("drop", (event) => {
            event.preventDefault();
            const from = Number(event.dataTransfer.getData("text/plain"));
            if (!Number.isInteger(from) || from === index) return;
            const [moved] = fx.stack.splice(from, 1);
            fx.stack.splice(index, 0, moved);
            fx.selected = index;
            fx.look = null;
            commit(node);
        });
        host.append(row);
    });
    if (!fx.stack.length) host.append(el("div", "zrl-fx-hint", "Pick a look above or add an effect."));
}

function togglePicker(node) {
    const fx = node._fx;
    const picker = fx.ui.picker;
    if (picker.style.display !== "none") {
        picker.style.display = "none";
        return;
    }
    picker.replaceChildren();
    for (const category of CATEGORY_ORDER) {
        const entries = Object.entries(fx.catalog.effects).filter(([, effect]) => effect.category === category);
        if (!entries.length) continue;
        picker.append(el("h5", "", category));
        for (const [type, effect] of entries) {
            const button = el("button", "", effect.name);
            button.type = "button";
            button.addEventListener("click", () => {
                fx.stack.push(newLayer(fx.catalog, type));
                fx.selected = fx.stack.length - 1;
                fx.look = null;
                picker.style.display = "none";
                commit(node);
            });
            picker.append(button);
        }
    }
    picker.style.display = "grid";
}

// ---------- parameter panel ----------

function slider(host, label, min, max, step, value, onInput, format) {
    const row = el("div", "zrl-fx-sl");
    const input = el("input");
    input.type = "range";
    Object.assign(input, { min: String(min), max: String(max), step: String(step), value: String(value) });
    const output = el("output");
    const show = () => {
        updateRangeFill(input);
        output.textContent = format ? format(Number(input.value)) : (step >= 1 ? String(Math.round(input.value)) : Number(input.value).toFixed(2));
    };
    input.addEventListener("input", () => { show(); onInput(Number(input.value)); });
    show();
    row.append(el("label", "", label), input, output);
    host.append(row);
    return row;
}

function field(host, label, control) {
    const row = el("div", "zrl-fx-field");
    row.append(el("label", "", label), control);
    host.append(row);
}

function chips(host, options, value, onPick, decorate) {
    const wrap = el("div", "zrl-fx-chips");
    for (const option of options) {
        const chip = el("button", `zrl-fx-chip${option === value ? " on" : ""}`);
        chip.type = "button";
        if (decorate) decorate(chip, option);
        chip.append(document.createTextNode(option));
        chip.addEventListener("click", () => onPick(option));
        wrap.append(chip);
    }
    host.append(wrap);
}

function wheel(host, label, layer, prefix, onChange) {
    const box = el("div", "zrl-fx-wheel");
    const disc = el("div", "zrl-fx-disc");
    const puck = el("i");
    disc.append(puck);
    const place = () => {
        puck.style.left = `${50 + layer.params[`${prefix}_x`] * 42}%`;
        puck.style.top = `${50 + layer.params[`${prefix}_y`] * 42}%`;
    };
    const move = (event) => {
        const rect = disc.getBoundingClientRect();
        let x = ((event.clientX - rect.left) / rect.width - 0.5) / 0.42;
        let y = ((event.clientY - rect.top) / rect.height - 0.5) / 0.42;
        const length = Math.hypot(x, y);
        if (length > 1) { x /= length; y /= length; }
        layer.params[`${prefix}_x`] = Math.round(x * 100) / 100;
        layer.params[`${prefix}_y`] = Math.round(y * 100) / 100;
        place();
        onChange();
    };
    let dragging = false;
    disc.addEventListener("pointerdown", (event) => { dragging = true; disc.setPointerCapture(event.pointerId); move(event); });
    disc.addEventListener("pointermove", (event) => { if (dragging) move(event); });
    disc.addEventListener("pointerup", () => { dragging = false; });
    disc.addEventListener("dblclick", () => { layer.params[`${prefix}_x`] = 0; layer.params[`${prefix}_y`] = 0; place(); onChange(); });
    place();
    const level = el("input");
    level.type = "range";
    Object.assign(level, { min: "-1", max: "1", step: "0.01", value: String(layer.params[prefix]) });
    updateRangeFill(level);
    level.addEventListener("input", () => { layer.params[prefix] = Number(level.value); updateRangeFill(level); onChange(); });
    box.append(el("span", "", label), disc, level);
    host.append(box);
}

function renderParams(node) {
    const fx = node._fx;
    const host = fx.ui.params;
    host.replaceChildren();
    const layer = fx.stack[fx.selected];
    if (!layer) {
        host.hidden = true;
        return;
    }
    host.hidden = false;
    const effect = fx.catalog.effects[layer.type];
    const head = el("div", "zrl-fx-sec", effect.name);
    if (effect.handle) head.append(el("span", "", "drag the handle on the photo"));
    host.append(head);
    const changed = () => { fx.look = null; commit(node, { rerender: false }); };
    slider(host, "Opacity", 0, 1, 0.01, layer.opacity, (value) => {
        layer.opacity = value;
        fx.ui.stack.children[fx.selected]?.querySelector(".zrl-fx-op")?.replaceChildren(`${Math.round(value * 100)}%`);
        changed();
    }, (value) => `${Math.round(value * 100)}%`);

    if (layer.type === "hsl") {
        const bands = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"];
        fx.hslBand = bands.includes(fx.hslBand) ? fx.hslBand : "red";
        chips(host, bands, fx.hslBand, (band) => { fx.hslBand = band; renderParams(node); });
        for (const [part, label] of [["hue", "Hue"], ["sat", "Saturation"], ["lum", "Luminance"]]) {
            const key = `${fx.hslBand}_${part}`;
            slider(host, label, -1, 1, 0.01, layer.params[key], (value) => { layer.params[key] = value; changed(); });
        }
        return;
    }
    if (layer.type === "color_wheels") {
        const wheels = el("div", "zrl-fx-wheels");
        wheel(wheels, "Lift", layer, "lift", changed);
        wheel(wheels, "Gamma", layer, "gamma", changed);
        wheel(wheels, "Gain", layer, "gain", changed);
        host.append(wheels, el("div", "zrl-fx-hint", "Drag a puck to tint; double-click to reset. Sliders set brightness."));
        return;
    }
    const handled = effect.handle ? new Set(["cx", "cy"]) : new Set();
    for (const param of effect.params) {
        if (handled.has(param.key)) continue;
        const value = layer.params[param.key];
        if (layer.type === "dither" && param.key === "levels" && layer.params.palette !== "Original levels") continue;
        if (param.kind === "float" || param.kind === "int") {
            slider(host, param.label, param.min, param.max, param.step, value, (next) => { layer.params[param.key] = next; changed(); });
        } else if (param.kind === "choice" && layer.type === "dither" && param.key === "palette") {
            host.append(el("div", "zrl-fx-sec", "Palette"));
            chips(host, param.options, value, (next) => { layer.params.palette = next; changed(); renderParams(node); }, (chip, option) => {
                const colours = fx.catalog.palettes[option];
                if (!colours) return;
                const swatch = el("span", "zrl-fx-swatch");
                for (const colour of colours.slice(0, 8)) {
                    const bar = el("b");
                    bar.style.background = colour;
                    swatch.append(bar);
                }
                chip.append(swatch);
            });
        } else if (param.kind === "choice" && (layer.type === "dither" || param.options.length <= 4)) {
            host.append(el("div", "zrl-fx-sec", param.label));
            chips(host, param.options, value, (next) => { layer.params[param.key] = next; changed(); renderParams(node); });
        } else if (param.kind === "choice") {
            const select = el("select");
            for (const option of param.options) select.append(Object.assign(el("option", "", option), { value: option }));
            select.value = value;
            select.addEventListener("change", () => { layer.params[param.key] = select.value; changed(); });
            field(host, param.label, select);
        } else if (param.kind === "color") {
            const wrap = el("span", "zrl-fx-colour");
            const picker = el("input");
            picker.type = "color";
            picker.value = value;
            const hex = el("span", "", value);
            picker.addEventListener("input", () => { layer.params[param.key] = picker.value.toUpperCase(); hex.textContent = picker.value.toUpperCase(); changed(); });
            wrap.append(picker, hex);
            field(host, param.label, wrap);
        } else if (param.kind === "bool") {
            const box = el("input");
            box.type = "checkbox";
            box.checked = Boolean(value);
            box.style.accentColor = "#E7B56A";
            box.addEventListener("change", () => { layer.params[param.key] = box.checked; changed(); });
            field(host, param.label, box);
        } else if (param.kind === "text") {
            if (layer.type === "dither" && param.key === "custom" && layer.params.palette !== "Custom") continue;
            const input = el("input");
            input.type = "text";
            input.value = value;
            input.addEventListener("keydown", (event) => event.stopPropagation());
            input.addEventListener("input", () => { layer.params[param.key] = input.value; changed(); });
            field(host, param.label, input);
        }
    }
}

// ---------- on-photo handles ----------

function imageRect(node) {
    const ui = node._zrlPreviewUI;
    const card = ui.card.getBoundingClientRect();
    const naturalW = ui.fullImage.naturalWidth || 1;
    const naturalH = ui.fullImage.naturalHeight || 1;
    const scale = Math.min(card.width / naturalW, card.height / naturalH);
    const width = naturalW * scale;
    const height = naturalH * scale;
    return { left: (card.width - width) / 2, top: (card.height - height) / 2, width, height, card };
}

function drawHandles(node) {
    const fx = node._fx;
    const svg = fx.ui.svg;
    svg.replaceChildren();
    const layer = fx.stack[fx.selected];
    const kind = layer && layer.on && fx.catalog?.effects[layer.type]?.handle;
    if (!kind || node._zrlPreviewUI.fullImage.style.visibility !== "visible") return;
    const ns = "http://www.w3.org/2000/svg";
    const r = imageRect(node);
    const zoom = r.card.width / (node._zrlPreviewUI.card.offsetWidth || r.card.width);
    const add = (tag, attrs) => {
        const shape = document.createElementNS(ns, tag);
        for (const [key, value] of Object.entries(attrs)) shape.setAttribute(key, value);
        svg.append(shape);
        return shape;
    };
    const X = (r.left + layer.params.cx * r.width) / zoom;
    const Y = (r.top + layer.params.cy * r.height) / zoom;
    const accent = "#E7B56A";
    if (kind === "center") {
        add("circle", { cx: X, cy: Y, r: 22, fill: "none", stroke: accent, "stroke-width": 1.5, "stroke-dasharray": "4 4" });
        add("circle", { cx: X, cy: Y, r: 7, fill: accent, class: "zrl-fx-h", "data-h": "center" });
    } else if (kind === "angle") {
        const a = layer.params.angle * Math.PI / 180;
        const reach = 24 + layer.params.distance * Math.max(r.width, r.height) / 1024 / zoom;
        add("line", { x1: X - Math.cos(a) * reach, y1: Y - Math.sin(a) * reach, x2: X + Math.cos(a) * reach, y2: Y + Math.sin(a) * reach, stroke: accent, "stroke-width": 2 });
        add("circle", { cx: X, cy: Y, r: 6, fill: accent, class: "zrl-fx-h", "data-h": "center" });
        add("circle", { cx: X + Math.cos(a) * reach, cy: Y + Math.sin(a) * reach, r: 8, fill: "#1C1712", stroke: accent, "stroke-width": 2, class: "zrl-fx-h", "data-h": "angle" });
    } else if (kind === "band") {
        const a = layer.params.angle * Math.PI / 180;
        const offset = layer.params.band * 0.5 * r.height / zoom;
        const length = Math.max(r.width, r.height) / zoom;
        for (const side of [-1, 1]) {
            const cx = X - Math.sin(a) * offset * side;
            const cy = Y + Math.cos(a) * offset * side;
            add("line", { x1: cx - Math.cos(a) * length, y1: cy - Math.sin(a) * length, x2: cx + Math.cos(a) * length, y2: cy + Math.sin(a) * length, stroke: accent, "stroke-width": 1.5, "stroke-dasharray": "6 5" });
        }
        add("circle", { cx: X, cy: Y, r: 7, fill: accent, class: "zrl-fx-h", "data-h": "center" });
        add("circle", { cx: X - Math.sin(a) * offset, cy: Y + Math.cos(a) * offset, r: 6, fill: "#1C1712", stroke: accent, "stroke-width": 2, class: "zrl-fx-h", "data-h": "band" });
    }
}

function bindHandles(node) {
    const fx = node._fx;
    const svg = fx.ui.svg;
    let drag = null;
    svg.addEventListener("pointerdown", (event) => {
        const handle = event.target.getAttribute?.("data-h");
        if (!handle) return;
        drag = handle;
        svg.setPointerCapture(event.pointerId);
        event.stopPropagation();
    });
    svg.addEventListener("pointermove", (event) => {
        if (!drag) return;
        const layer = fx.stack[fx.selected];
        const r = imageRect(node);
        const zoom = r.card.width / (node._zrlPreviewUI.card.offsetWidth || r.card.width);
        const x = (event.clientX - r.card.left - r.left) / r.width;
        const y = (event.clientY - r.card.top - r.top) / r.height;
        if (drag === "center") {
            layer.params.cx = Math.min(1, Math.max(0, Math.round(x * 1000) / 1000));
            layer.params.cy = Math.min(1, Math.max(0, Math.round(y * 1000) / 1000));
        } else if (drag === "angle") {
            const dx = (x - layer.params.cx) * r.width;
            const dy = (y - layer.params.cy) * r.height;
            layer.params.angle = Math.round(((Math.atan2(dy, dx) * 180 / Math.PI) + 360) % 180);
            const reachPx = Math.max(0, Math.hypot(dx, dy) - 24 * zoom);
            layer.params.distance = Math.min(200, Math.round(reachPx * 1024 / Math.max(r.width, r.height)));
        } else if (drag === "band") {
            const a = layer.params.angle * Math.PI / 180;
            const dx = (x - layer.params.cx) * r.width;
            const dy = (y - layer.params.cy) * r.height;
            const along = Math.abs(-dx * Math.sin(a) + dy * Math.cos(a));
            layer.params.band = Math.min(0.9, Math.max(0.02, Math.round(along * 2 / r.height * 100) / 100));
        }
        fx.look = null;
        commit(node, { rerender: false });
        renderParams(node);
    });
    svg.addEventListener("pointerup", () => { drag = null; });
}

// ---------- histogram, A/B, dice, hold ----------

function drawHistogram(node) {
    const fx = node._fx;
    const canvas = fx.ui.hist;
    if (canvas.hidden) return;
    const image = node._zrlPreviewUI.fullImage;
    if (!image.naturalWidth) return;
    const probe = document.createElement("canvas");
    probe.width = 128;
    probe.height = Math.max(1, Math.round(128 * image.naturalHeight / image.naturalWidth));
    const ctx = probe.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, probe.width, probe.height);
    const data = ctx.getImageData(0, 0, probe.width, probe.height).data;
    const bins = new Array(64).fill(0);
    for (let i = 0; i < data.length; i += 4) bins[Math.min(63, ((0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 4) | 0)] += 1;
    const peak = Math.max(...bins, 1);
    const draw = canvas.getContext("2d");
    draw.clearRect(0, 0, canvas.width, canvas.height);
    draw.fillStyle = "rgba(18,16,14,0.78)";
    draw.fillRect(0, 0, canvas.width, canvas.height);
    draw.fillStyle = "#E7B56A";
    bins.forEach((count, index) => {
        const h = count / peak * (canvas.height - 10);
        draw.fillRect(6 + index * 3.6, canvas.height - 5 - h, 3, h);
    });
}

function iconButton(label, title) {
    const button = el("button", "zrl-glass zrl-icon", label);
    button.type = "button";
    button.title = title;
    return button;
}

function randomStack(catalog) {
    const types = Object.keys(catalog.effects).filter((type) => !["film_frame", "hsl", "color_wheels", "jpeg_crunch"].includes(type));
    const pick = (list) => list[Math.floor(Math.random() * list.length)];
    const stack = [];
    for (let i = 0; i < 2 + Math.floor(Math.random() * 2); i += 1) {
        const type = pick(types);
        const layer = newLayer(catalog, type);
        for (const param of catalog.effects[type].params) {
            if (param.kind === "float" && !["cx", "cy"].includes(param.key)) {
                const value = param.min + (param.max - param.min) * (0.2 + Math.random() * 0.45);
                layer.params[param.key] = Math.round(value / param.step) * param.step;
            } else if (param.kind === "choice" && param.key !== "palette") {
                layer.params[param.key] = pick(param.options);
            } else if (param.key === "palette") {
                layer.params.palette = pick(param.options.filter((option) => option !== "Custom" && option !== "Original levels"));
            }
        }
        stack.push(layer);
    }
    return stack;
}

function addPreviewTools(node) {
    const fx = node._fx;
    const ui = node._zrlPreviewUI;
    ui.languageIcon.parentElement.style.display = "none";
    const ab = iconButton("A", "A/B: snapshot these settings, then flip between them");
    const histButton = iconButton("▁▃▆", "Histogram");
    const dice = iconButton("🎲", "Surprise me: random look");
    const hold = iconButton("◐", "Hold to see the original");
    const reset = iconButton("↺", "Clear the effect stack");
    ui.resetSlot.prepend(hold, ab, histButton, dice);
    ui.resetSlot.append(reset);
    ab.addEventListener("click", () => {
        if (!fx.snapshot) {
            fx.snapshot = JSON.parse(JSON.stringify(fx.stack));
            ab.textContent = "B";
            ab.title = "Showing B. Click to flip to A.";
            return;
        }
        const current = fx.stack;
        fx.stack = fx.snapshot;
        fx.snapshot = current;
        ab.textContent = ab.textContent === "B" ? "A" : "B";
        fx.selected = Math.min(fx.selected, fx.stack.length - 1);
        fx.look = null;
        commit(node);
    });
    histButton.addEventListener("click", () => {
        fx.ui.hist.hidden = !fx.ui.hist.hidden;
        histButton.classList.toggle("on", !fx.ui.hist.hidden);
        drawHistogram(node);
    });
    dice.addEventListener("click", () => {
        fx.stack = randomStack(fx.catalog);
        fx.selected = 0;
        fx.look = null;
        renderLooks(node);
        commit(node);
    });
    const release = () => {
        if (!fx.holding) return;
        fx.holding = false;
        ui.compare.setMode(fx.holdMode);
        renderActiveCompositionPreview(node);
    };
    hold.addEventListener("pointerdown", (event) => {
        event.stopPropagation();
        fx.holding = true;
        fx.holdMode = ui.compare.mode();
        ui.compare.setMode("before");
        renderActiveCompositionPreview(node);
    });
    hold.addEventListener("pointerup", release);
    hold.addEventListener("pointerleave", release);
    reset.addEventListener("click", () => {
        fx.stack = [];
        fx.selected = 0;
        fx.look = "Clean";
        renderLooks(node);
        commit(node);
    });
}

// ---------- video mode ----------

function enterVideoMode(node, token) {
    if (!node._zrlVideo) {
        node._zrlVideo = {
            generation: 0, playing: false, timer: null, pumping: false,
            endpoint: FX_ENDPOINT, payload: () => ({ stack: node._fx.stack }),
        };
        node._fx.player ||= createVideoPlayerControls(node);
        node._zrlVideoPlayerUI = node._fx.player;
        node._zrlPreviewUI.panel.append(node._fx.player.row);
    }
    node._fx.player.row.style.display = "flex";
    resetVideoState(node, token);
    pumpVideoRender(node);
}

function exitVideoMode(node) {
    if (!node._zrlVideo) return;
    deleteVideoPreviewToken(node._zrlVideo.token);
    resetVideoState(node);
    node._fx.player.row.style.display = "none";
    delete node._zrlVideo;
}

// ---------- node wiring ----------

function buildFxNode(node) {
    installDarkroomStyle();
    if (!document.getElementById("zrl-fx-style")) {
        const style = el("style");
        style.id = "zrl-fx-style";
        style.textContent = FX_STYLE;
        document.head.append(style);
    }
    node.properties = node.properties || {};
    const widget = stackWidget(node);
    widget.hidden = true;
    node._fx = { stack: readStack(node), selected: 0, catalog: null, userLooks: {}, thumbs: new Map(), lookEls: new Map(), look: null, snapshot: null, ui: {} };
    node._zrlPreviewToken = null;
    node._zrlPreviewOriginalSrc = null;
    node._zrlPreviewProcessedSrc = null;

    const previewUI = createPreviewPanel();
    node._zrlPreviewUI = previewUI;
    previewUI.compare.setOnChange(() => renderActiveCompositionPreview(node));
    previewUI.loupe.remove();
    previewUI.status.textContent = "Queue the node once to create a preview source.";
    const handles = el("div", "zrl-ov zrl-fx-handles");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    handles.append(svg);
    const hist = el("canvas", "zrl-fx-hist");
    hist.width = 240;
    hist.height = 88;
    hist.hidden = true;
    previewUI.card.append(handles, hist);
    node._fx.ui.svg = svg;
    node._fx.ui.hist = hist;
    addDOMWidgetAt(node, "zrl_fx_preview", previewUI.panel, "fx_stack", { getMinHeight: () => 300 });

    const looksSection = el("div", "zrl-dr");
    const looksHead = el("div", "zrl-fx-sec", "Looks");
    looksHead.append(el("span", "", "saved in your ComfyUI user folder"));
    const looks = el("div", "zrl-fx-looks");
    looksSection.append(looksHead, looks);
    node._fx.ui.looks = looks;
    addDOMWidgetAt(node, "zrl_fx_looks", looksSection, "fx_stack", contentHeight(looksSection));

    const stackSection = el("div", "zrl-dr zrl-fx-stack");
    const stackHead = el("div", "zrl-fx-sec", "Effect stack");
    const count = el("span", "", "");
    stackHead.append(count);
    const rows = el("div", "zrl-fx-stack");
    const add = el("button", "zrl-fx-add", "＋ Add effect");
    add.type = "button";
    const picker = el("div", "zrl-fx-picker");
    picker.style.display = "none";
    stackSection.append(stackHead, rows, add, picker);
    Object.assign(node._fx.ui, { stack: rows, count, picker });
    addDOMWidgetAt(node, "zrl_fx_stack", stackSection, "fx_stack", contentHeight(stackSection));

    const params = el("div", "zrl-dr zrl-fx-params");
    node._fx.ui.params = params;
    addDOMWidgetAt(node, "zrl_fx_params", params, "fx_stack", contentHeight(params));

    for (const section of [previewUI.panel, looksSection, stackSection, params]) {
        section.addEventListener("pointerdown", (event) => event.stopPropagation());
        section.addEventListener("mousedown", (event) => event.stopPropagation());
    }
    add.addEventListener("click", () => togglePicker(node));
    // The card resizes as panels below it grow or shrink; keep handles on the photo.
    new ResizeObserver(() => drawHandles(node)).observe(previewUI.card);
    bindHandles(node);
    addPreviewTools(node);

    loadCatalog().then((catalog) => {
        node._fx.catalog = catalog;
        node._fx.stack = node._fx.stack.filter((layer) => catalog.effects[layer?.type]);
        renderLooks(node);
        renderStack(node);
        renderParams(node);
        loadUserLooks(node);
    }).catch((error) => {
        rows.replaceChildren(el("div", "zrl-fx-hint", `FX catalog unavailable: ${error?.message || error}`));
    });
}

app.registerExtension({
    name: "zuanfilm.retrolab.fx",

    nodeCreated(node, extensionApp) {
        if (node?.comfyClass === NODE_NAME || node?.type === NODE_NAME) applyInitialNodeWidth(node, extensionApp);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalCreated?.apply(this, args);
            if (typeof this.addDOMWidget === "function" && stackWidget(this)) buildFxNode(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = originalConfigure?.apply(this, args);
            if (!this._fx) return result;
            this._fx.stack = readStack(this);
            this._fx.selected = 0;
            if (this._fx.catalog) {
                this._fx.stack = this._fx.stack.filter((layer) => this._fx.catalog.effects[layer?.type]);
                renderStack(this);
                renderParams(this);
            }
            return result;
        };

        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message, ...args) {
            const result = originalExecuted?.call(this, message, ...args);
            if (!this._fx) return result;
            const pick = (key) => {
                const value = message?.[key] ?? message?.ui?.[key];
                return Array.isArray(value) ? value[0] : value;
            };
            const videoToken = pick("retrolab_video_preview_id");
            const imageToken = pick("retrolab_preview_id");
            this._zrlPreviewUI.compare.setMode("split");
            if (typeof videoToken === "string" && videoToken.length === 32) {
                deletePreviewToken(this._zrlPreviewToken);
                this._zrlPreviewToken = null;
                if (this._zrlVideo?.token && this._zrlVideo.token !== videoToken) deleteVideoPreviewToken(this._zrlVideo.token);
                enterVideoMode(this, videoToken);
            } else if (typeof imageToken === "string" && imageToken.length === 32) {
                exitVideoMode(this);
                if (this._zrlPreviewToken && this._zrlPreviewToken !== imageToken) deletePreviewToken(this._zrlPreviewToken);
                this._zrlPreviewToken = imageToken;
                this._zrlPreviewOriginalSrc = null;
                this._zrlPreviewProcessedSrc = null;
                this._fx.thumbToken = null;
                schedulePreview(this, true);
            }
            return result;
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            if (this._fx) {
                clearTimeout(this._fx.timer);
                deletePreviewToken(this._zrlPreviewToken);
                exitVideoMode(this);
            }
            return originalRemoved?.apply(this, args);
        };
    },
});
