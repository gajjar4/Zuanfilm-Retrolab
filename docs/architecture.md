# Zuanfilm Retrolab — Engineering Architecture

**Document purpose:** implementation-level reference for the released Zuanfilm Retrolab ComfyUI node.  
**Baseline:** v1.0.0 public release; checked against the source modules on the repository default branch.  
**Scope:** tensor contract, processing order, mathematical transforms, PyTorch operations, CUDA execution, memory policy, preview behavior and validation boundaries.

This document describes the implementation that exists in the repository. It is not a proposal for a future renderer, a generic description of film photography, or a claim that the named film stocks are chemically simulated.

---

## 1. Product and engineering scope

Zuanfilm Retrolab is a post-rendering photographic finishing node for ComfyUI. It accepts a decoded ComfyUI IMAGE tensor, applies primary corrections and a fixed sequence of creative finishing operations, and returns an IMAGE tensor plus a structured diagnostic STRING.

The engineering principles are:

1. **Creative controls first.** Hardware selection, CUDA availability, VRAM budgeting and fallback are internal runtime decisions. They are not exposed as creative controls.
2. **One deterministic image pipeline.** CPU and CUDA routes execute the same PyTorch stage functions and float32 image mathematics.
3. **Bounded working memory.** Batch frames are processed sequentially. Broad spatial filters reduce their working resolution when needed. CUDA admission uses a conservative one-frame working-set estimate and reserves GPU headroom.
4. **Failure isolation.** A failed preview must not break full-resolution image output. A recognized CUDA allocation failure discards the incomplete CUDA batch and retries the entire batch on CPU. Unrelated errors are not silently converted into fallback.

Retrolab is not a RAW development application, camera-sensor model, spectral film simulation, scanner emulation or calibrated chemical-process simulator. The emulsion atlas contains independently authored creative RGB transforms inspired by public film-family references.

## 2. Public node contract

| Property | Implementation |
|---|---|
| Python class | ZuanfilmRetroLab |
| Node key | ZuanfilmRetroLab |
| Display name | Zuanfilm Retrolab |
| Category | Zuanfilm/RetroLab |
| Input | ComfyUI IMAGE tensor, BHWC |
| Image output | IMAGE, output index 0 |
| Diagnostic output | STRING named report, output index 1 |
| Node definition | nodes/zuanfilm_retrolab.py |
| Pipeline orchestration | core/pipeline.py |
| Runtime planning | core/execution.py |
| Preview API and cache | preview.py |
| Frontend | js/zuanfilm_retrolab.js |

The input image has shape:

\[
X\in\mathbb{R}^{B\times H\times W\times C},\qquad C\ge3
\]

where \(B\) is batch size, \(H\) is image height, \(W\) is image width, and the first three channels are interpreted as RGB. Channels after RGB are copied through unchanged.

In normal execution, the output preserves batch size, height, width, channel count, input dtype and input output-device placement. An exceptional CUDA-to-CPU recovery returns the recovered output on CPU. Non-contiguous input is accepted; the pipeline does not require a contiguous source tensor.

## 3. Source layout and responsibilities

| Path | Responsibility |
|---|---|
| nodes/zuanfilm_retrolab.py | ComfyUI schema, defaults, input normalization, settings creation, execution and preview-token delivery |
| core/pipeline.py | Settings dataclass, identity detection, fixed stage order, per-frame processing, batch processing and report generation |
| core/execution.py | CPU/CUDA selection, CUDA operation compatibility preflight, working-set estimate and OOM classification |
| core/color.py | sRGB transfer functions and linear Rec.709 luminance |
| core/filters.py | Separable Gaussian kernels and resolution-aware blur |
| core/white_balance.py | Two-axis temperature/tint gain-wheel transform |
| core/anti_ai.py | Global tonal and chroma normalization |
| core/exposure.py | Normalized linear-light exposure/lightness control |
| core/bloom.py | Soft-knee highlight extraction and multi-scale glow |
| core/halation.py | Annular Fresnel/TIR-inspired kernel, FFT convolution and color weighting |
| core/light_leak.py | Seeded edge geometry, procedural profiles and spectral attenuation |
| core/emulsions.py | 50 immutable creative profile definitions and emulsion transform |
| core/sabattier.py | Solarization curve, highlight confinement and optional Mackie contours |
| core/soft_focus.py | Center-preserving, corner-weighted linear-light blur |
| core/vignette.py | Aspect-correct radial exposure attenuation |
| core/noise_texture.py | Twelve deterministic procedural texture families |
| core/grain.py | Coordinate-hashed Gaussian grain, size correlation and color-grain mixing |
| core/blending.py | Shared 12-mode signed-layer compositor for texture and grain |
| preview.py | Reduced CPU preview cache, request validation, rendering endpoints and PNG encoding |
| js/zuanfilm_retrolab.js | DOM controls, localization, preview interaction and frontend presentation |

The image engines are implemented with PyTorch. Pillow is used by the preview path to encode small PNG responses and is supplied by the ComfyUI runtime. No additional numerical, noise, image-processing or CUDA package is required by Retrolab.

## 4. High-level data flow

~~~mermaid
flowchart TD
    A[ComfyUI IMAGE BHWC] --> B[ZuanfilmRetroLab.execute]
    B --> C[Normalize UI values]
    C --> D[RetroLabSettings]
    D --> E[process_batch_with_report]
    E --> F{Identity settings?}
    F -->|Yes| G[Return original tensor]
    F -->|No| H[select_execution_plan]
    H --> I[One frame at a time]
    I --> J[RGB float32 on processing device]
    J --> K[12-stage image pipeline]
    K --> L[Copy RGB to output tensor]
    I --> M[Copy channels after RGB unchanged]
    L --> N[ComfyUI IMAGE]
    M --> N
    E --> O[JSON execution report]
    O --> P[STRING report]
    B --> Q[Create bounded CPU preview source]
    Q --> R[Preview token in node UI payload]
    N --> S[ComfyUI node outputs]
    P --> S
~~~

The processing order is fixed in process_single_image:

~~~mermaid
flowchart LR
    A[Input RGB] --> B[White Balance]
    B --> C[Anti-AI]
    C --> D[Exposure]
    D --> E[Bloom]
    E --> F[Halation]
    F --> G[Light Leak]
    G --> H[Color Film Emulsion]
    H --> I[Sabattier]
    I --> J[Soft Focus]
    J --> K[Vignette]
    K --> L[Procedural Texture]
    L --> M[Film Grain]
    M --> N[Output RGB]
~~~

| Order | Stage | Signal domain | Role |
|---:|---|---|---|
| 00 | White Balance | Linear RGB internally | Primary temperature/tint correction |
| 01 | Anti-AI | Linear RGB plus encoded-sRGB hue mask | Global tonal and chroma normalization |
| 02 | Exposure | Linear RGB | Normalized darken/lighten control |
| 03 | Bloom | Linear RGB | Soft-knee highlight glow |
| 04 | Halation | Linear RGB; FFT spatial convolution | Annular highlight scattering |
| 05 | Light Leak | Linear RGB | Seeded edge illumination and spectral falloff |
| 06 | Color Film Emulsion | Linear RGB | Selected profile tone and color response |
| 07 | Sabattier | Linear RGB | Non-monotonic luminance reversal and optional edge contours |
| 08 | Soft Focus | Linear RGB | Spatially varying optical softness |
| 09 | Vignette | Linear RGB | Radial exposure falloff |
| 10 | Procedural Texture | Encoded sRGB carrier | Deterministic image-space surface texture |
| 11 | Film Grain | Encoded sRGB carrier | Final deterministic photographic grain |

The order is intentional. White Balance is a primary correction. Anti-AI and Exposure prepare the image before highlight-dependent light effects. Bloom, Halation and Light Leak operate before the film-emulsion transform. Sabattier shapes the emulsion-rendered luminance. Soft Focus and Vignette are spatial finishing operations. Procedural Texture follows optical blur so its fine detail is not softened. Film Grain is last for the same reason.

The pipeline skips inactive stages. A zero value is not merely a very small effect: the corresponding engine is normally not called.

## 5. Tensor conventions, color math and shared primitives

### 5.1 Working tensor shapes

The ComfyUI image enters in BHWC layout. The batch executor selects one frame:

\[
X_b\in\mathbb{R}^{H\times W\times C}
\]

Only RGB is transferred to the chosen processing device. RGB is converted to float32 and clamped to the supported display-referred interval before the first active image transform:

\[
X_{\mathrm{RGB}}^{32}
=
\operatorname{float32}\left(
\operatorname{clamp}(X[...,0:3],0,1)
\right)
\]

Most stage functions accept one HWC RGB frame. Spatial filters temporarily convert HWC to NCHW/BCHW where required by PyTorch convolution, pooling and interpolation:

\[
HWC\rightarrow CHW\rightarrow 1CHW
\]

At the end of a frame, RGB is copied into a preallocated output tensor, converted back to the input dtype and output device. Additional channels are copied directly from the source frame without transfer to the processing device.

### 5.2 sRGB transfer functions

Input RGB values are interpreted as sRGB-encoded display-referred values. Operations modelling light addition, attenuation, exposure or luminance are performed in linear RGB.

For encoded channel \(c_s\in[0,1]\), core/color.py uses:

\[
c_l=
\begin{cases}
c_s/12.92 & c_s\le0.04045\\
\left(\frac{c_s+0.055}{1.055}\right)^{2.4} & c_s>0.04045
\end{cases}
\]

The inverse conversion for a non-negative linear channel \(c_l\) is:

\[
c_s=
\begin{cases}
12.92c_l & c_l\le0.0031308\\
1.055c_l^{1/2.4}-0.055 & c_l>0.0031308
\end{cases}
\]

The source functions clamp encoded input, clamp negative linear input to zero for encoding, and bound final encoded output to \([0,1]\).

**PyTorch mapping:** torch.clamp, torch.where, division, torch.pow, multiplication and addition. These are device-dispatched tensor expressions and run on CUDA when their tensors are CUDA-resident.

### 5.3 Linear Rec.709 luminance

luminance_rec709 expects linear RGB:

\[
Y=0.2126R_l+0.7152G_l+0.0722B_l
\]

It is implemented as channel selections and weighted tensor additions. It is used by Anti-AI, Bloom, Sabattier, Procedural Texture and Film Grain.

### 5.4 Smoothstep

Several effects use the same smooth transition:

\[
t=\operatorname{clamp}\left(
\frac{x-e_0}{\max(e_1-e_0,\epsilon)},0,1
\right)
\]

\[
S(t)=t^2(3-2t)
\]

Anti-AI uses this for contrast weighting, highlight weighting, saturation gates and hue-mask saturation. It is implemented with tensor clamp, division, multiplication and subtraction.

### 5.5 Gaussian filtering

core/filters.py supplies the shared Gaussian primitives.

For standard deviation \(\sigma\), the 1D kernel radius is:

\[
r=\max(1,\lceil3\sigma\rceil)
\]

For coordinates \(i\in[-r,r]\):

\[
g_i=\exp\left(-\frac{1}{2}\left(\frac{i}{\sigma}\right)^2\right),
\qquad
k_i=\frac{g_i}{\sum_jg_j}
\]

For very small \(\sigma\), a one-element identity kernel is returned. A separable 2D blur is computed as horizontal convolution followed by vertical convolution. The tensor is padded using replicate padding, and channel groups are convolved independently using torch.nn.functional.conv2d with groups equal to the channel count.

For large blur radii, gaussian_blur_scaled limits working-filter size and edge length:

\[
f_\sigma=\left\lceil\frac{\sigma_{\mathrm{full}}}{10}\right\rceil,
\qquad
f_{\mathrm{edge}}=\left\lceil\frac{\max(H,W)}{2048}\right\rceil
\]

\[
f=\max(1,f_\sigma,f_{\mathrm{edge}})
\]

When \(f>1\), the image is reduced with area interpolation, blurred at \(\sigma_{\mathrm{full}}/f\), then enlarged with bilinear interpolation and align_corners=False. Area reduction is used for the broad source field.

**PyTorch mapping:** torch.arange, torch.exp, tensor reductions, torch.nn.functional.pad, grouped conv2d, interpolate(mode="area") and interpolate(mode="bilinear").

## 6. Stage implementation and equations

The following subsections describe the implementation in the corresponding core module. Symbols introduced in a subsection are local to that stage.

### 6.1 White Balance — core/white_balance.py

The wheel is a creative two-axis gain correction, not a Kelvin-calibrated RAW white-balance transform.

Controls: Temperature \(T\in[-1,1]\), cool to warm; Tint \(t\in[-1,1]\), green to magenta. Both zero means exact bypass.

After sRGB-to-linear conversion, channel offsets are:

\[
e_R=0.38T+0.22t,\quad
e_G=-0.30t,\quad
e_B=-0.38T+0.22t
\]

Channel gains are powers of two:

\[
g_R=2^{e_R},\quad g_G=2^{e_G},\quad g_B=2^{e_B}
\]

To avoid an unintended global exposure shift on neutral gray, the gains are normalized by their Rec.709-weighted neutral luminance:

\[
g_Y=0.2126g_R+0.7152g_G+0.0722g_B
\]

\[
\tilde g_c=g_c/\max(g_Y,10^{-8})
\]

The output is:

\[
RGB_{\mathrm{out}}=RGB_l\odot(\tilde g_R,\tilde g_G,\tilde g_B)
\]

followed by linear-to-sRGB conversion.

**PyTorch mapping:** transfer functions, torch.tensor for three channel gains on the current device, scalar tensor arithmetic, view(1,1,3) broadcasting and elementwise multiplication.

### 6.2 Anti-AI normalization — core/anti_ai.py

Anti-AI is a deterministic global tonal/chroma correction. It is not an AI detector, local retouching system or content-aware segmentation pass.

The UI amount \(A\in[0,100]\) is normalized:

\[
s=\operatorname{clamp}(A/100,0,1)
\]

At \(s=0\), the original stage input is returned unchanged.

Let \(Y\) be linear Rec.709 luminance. Contrast weighting is:

\[
w_c=S(0.015,0.96,Y)
\]
\[
Y_c=Y+(0.46-Y)(0.14s\,w_c)
\]

Three tonal masks are computed:

\[
w_t=\exp\left[-\frac{1}{2}\left(\frac{Y}{0.105}\right)^2\right]
\]
\[
w_m=\exp\left[-\frac{1}{2}\left(\frac{Y-0.46}{0.25}\right)^2\right]
\]
\[
w_h=S(0.74,1,Y)^2
\]

Mapped luminance is:

\[
Y_m=\operatorname{clamp}
\left(
Y_c+0.012s\,w_t+0.026s\,w_m-0.038s\,w_h,\ 0,\ 1
\right)
\]

The transform first computes a common RGB scale:

\[
q=\frac{Y_m}{\max(Y,10^{-6})}
\]

To prevent the scale from driving the largest linear RGB channel above 1 before chroma processing, it is capped:

\[
q\leftarrow\min\left(q,\frac{1}{\max(\max_c RGB_{l,c},10^{-6})}\right)
\]

\[
RGB_t=RGB_lq
\]

The implementation recomputes \(Y_t\) and obtains a zero-centered chroma residual:

\[
C=RGB_t-Y_t
\]

A soft red/orange mask is calculated in encoded sRGB. Per pixel, the code derives HSV-like hue from channel maximum/minimum and channel differences, wraps the red hue boundary circularly, applies a Gaussian hue-distance weight around hue \(0.035\) turns, and gates it by a smooth saturation mask from 0.025 to 0.22:

\[
w_{ro}=w_{\mathrm{hue}}w_{\mathrm{sat}}
\]

A shadow mask is:

\[
w_s=1-S(0.025,0.34,Y)
\]

Chroma multipliers are:

\[
m_g=1-0.08s,\quad
m_{ro}=1-0.34s\,w_{ro},\quad
m_s=1-0.08s\,w_s
\]

\[
m_C=\operatorname{clamp}(m_gm_{ro}m_s,0,1)
\]

The final linear result before encoding is:

\[
RGB_{\mathrm{out}}=Y_t+C\,m_C
\]

This structure changes luminance through a shared RGB ratio, then changes chroma around the recomputed luminance. It avoids spatially varying masks and local halos.

**PyTorch mapping:** sRGB conversion, amax/amin, unbind, where, remainder, abs, minimum, exp, square, smoothstep, clamp, clamp_min, torch.minimum, broadcasting and linear-to-sRGB conversion.

### 6.3 Exposure — core/exposure.py

Exposure is a normalized photographic lightness control, not a camera EV model. The public control \(a\) is clamped to \([-100,100]\). Zero is an exact bypass.

In linear RGB, negative values move toward black:

\[
p_-=0.95\frac{|a|}{100}
\]
\[
RGB_{\mathrm{out}}=RGB_l(1-p_-)
\]

At \(-100\), 5% of the source linear-light value remains.

Positive values move toward white using a 2.5-power ease-in:

\[
p_+=0.95\left(\frac{a}{100}\right)^{2.5}
\]
\[
RGB_{\mathrm{out}}=RGB_l+(1-RGB_l)p_+
\]

At \(+100\), the transform covers 95% of the remaining distance to white. The result is converted to sRGB.

**PyTorch mapping:** sRGB transfer, scalar power mapping and elementwise multiply/add.

### 6.4 Bloom — core/bloom.py

Bloom is a color-preserving linear-light glow produced from a soft-knee highlight field.

Let \(Y\) be linear Rec.709 luminance and \(T\) the threshold. The fixed knee width is \(k=0.12\), shortened at low thresholds:

\[
k_e=\min(0.12,2\max(T,0))
\]

If \(k_e\le10^{-8}\), extraction is \(\max(Y,0)\). Otherwise:

\[
x=\operatorname{clamp}\left(Y-(T-k_e/2),0,k_e\right)
\]
\[
E(Y)=\frac{x^2}{2k_e}+\max(Y-(T+k_e/2),0)
\]

The color channels are retained by scaling RGB with the luminance extraction ratio:

\[
q=\frac{E(Y)}{\max(Y,10^{-5})},\qquad H=RGB_lq
\]

The spatial radius is scaled relative to a 1024-pixel long edge:

\[
r_{\mathrm{px}}=r_{\mathrm{control}}\frac{\max(H_{\mathrm{image}},W_{\mathrm{image}})}{1024}
\]

The extracted highlight image is blurred twice:

\[
B_c=G(r_{\mathrm{px})*H,\qquad
B_t=G(2.5r_{\mathrm{px}})*H
\]
\[
B=0.65B_c+0.35B_t
\]

where \(G(\sigma)\) denotes the shared scaled Gaussian blur. The output is:

\[
RGB_{\mathrm{out}}=
\operatorname{sRGB}\left(
\operatorname{clamp}(RGB_l+0.38aB,0,1)
\right)
\]

where \(a\in[0,3]\) is Bloom Amount. A completely black input extracts zero and cannot acquire an artificial glow.

**PyTorch mapping:** luminance, clamp, clamp_min, soft-knee tensor arithmetic, channel broadcasting, NCHW permutation, scaled Gaussian convolution and interpolation.

### 6.5 Halation — core/halation.py

Halation creates an annular response with an outward scattering tail. It uses a spatial kernel and FFT convolution. The Fresnel formulation is a creative optical approximation, not a calibrated film-base simulation.

#### Kernel construction

Let \(r\) be radial distance from kernel center, \(R\) the ring radius and \(L\) the tail length. The implementation uses refractive index \(n=1.5\):

\[
d=R\sqrt{n^2-1},\quad
u=r/d,\quad
\theta_i=\arctan(u)
\]
\[
\sin\theta_t=n\sin\theta_i
\]

For the subcritical region \(\sin\theta_t<1\), the transmitted angle is computed from the clamped arcsine. The implementation computes s- and p-polarized Fresnel amplitude ratios:

\[
r_s=\frac{n\cos\theta_i-\cos\theta_t}
{n\cos\theta_i+\cos\theta_t}
\]
\[
r_p=\frac{n\cos\theta_t-\cos\theta_i}
{n\cos\theta_t+\cos\theta_i}
\]

It uses their average squared reflectance. In the total-internal-reflection region, reflectance is set to 1:

\[
F(r)=
\begin{cases}
(r_s^2+r_p^2)/2 & n\sin\theta_i<1\\
1 & \text{otherwise}
\end{cases}
\]

The annular component is:

\[
K_r(r)=\frac{F(r)}{(1+u^2)^2}
\]

and is normalized to unit sum.

The tail uses a radial ramp and exponential outward decay:

\[
w(r)=\min(r/R,1)^2
\]
\[
K_t(r)=
\frac{\exp[-\max(r-R,0)/L]\,w(r)}{r+0.5}
\]

It is also normalized. The combined kernel is:

\[
K=0.65K_r+0.35K_t
\]

followed by a final unit-sum normalization. Kernel extent uses the larger of approximately \(4R\) and \(6L\), with a minimum half-size of 8 pixels.

#### Working resolution and FFT

If the source long edge exceeds 1536 pixels, the stage computes:

\[
f=\left\lceil\frac{\max(H,W)}{1536}\right\rceil
\]

The RGB linear image is reduced using max pooling with kernel/stride \(f\), ceil_mode=True, then resized to calculated work dimensions with nearest interpolation. Max pooling preserves small bright sources better than averaging before halation generation.

The ring radius is scaled to the working image's 1024-pixel reference edge. Tail length is fixed at 40 reference pixels. For each RGB channel, the implementation computes a linear convolution by zero-padded real FFT:

\[
S=\operatorname{rFFT2}(I,s=(H_f,W_f))
\]
\[
K_f=\operatorname{rFFT2}(K,s=(H_f,W_f))
\]
\[
C=\operatorname{irFFT2}(S\cdot K_f,s=(H_f,W_f))
\]

The FFT dimensions are the next lengths at least as large as the full linear-convolution size, with prime factors limited to 2, 3 and 5. The result is cropped to the original field dimensions to obtain same-sized convolution.

The selected Halation Color is parsed from sRGB hex, converted to linear RGB and normalized by its largest channel. The default color is special-cased to preserve exact legacy channel weights:

\[
w_c=(1.00,0.32,0.10)
\]

Each channel's convolved field is multiplied by its channel weight. If the working image was reduced, the halo is bilinearly resized to source dimensions.

Finally:

\[
RGB_{\mathrm{out}}=
\operatorname{sRGB}\left(
\operatorname{clamp}(RGB_l+0.25aH,0,1)
\right)
\]

where \(a\in[0,2]\) and \(H\) is the color-weighted halo field.

**PyTorch mapping:** meshgrid, trigonometric functions, sqrt, atan, asin, cos, exp, where, reductions, kernel normalization, max_pool2d, interpolate(mode="nearest"/"bilinear"), torch.fft.rfft2, complex pointwise multiplication and torch.fft.irfft2. On CUDA, FFT execution is provided through PyTorch's CUDA FFT backend (cuFFT where supported); max pooling and interpolation use their CUDA implementations.

The kernel cache is limited to six entries, keyed by rounded radius/tail/balance and device.

### 6.6 Light Leak — core/light_leak.py

Light Leak builds a deterministic image-space profile from an edge or corner. Geometry is generated once for a given image shape, mode, position and seed, then applied equally to every batch frame. There is no per-frame leak variation.

Controls include five modes, eight positions, Exposure \(e\in[0,10]\), Seed and Influence Radius \(q\in[0.25,10]\). The default leak color is #C2AF8E.

#### Seeded boundary fields

Low-frequency 2D warps are produced from small seeded CPU Torch random grids. Two coarse grids are generated, moved to the image device, enlarged with bicubic interpolation and combined as:

\[
W=0.76W_a+0.24W_b
\]

The field is mean-centered, divided by its maximum absolute value, clamped to \([-1,1]\), and scaled by a mode-specific amplitude relative to the 1024-pixel long edge. Seeded 1D edge variation is similarly generated at low resolution and linearly interpolated along the selected edge.

This is an intentional mixed computation: random control grids and Python random geometry are created on CPU, while full-resolution coordinate fields, profiles, attenuation and compositing are evaluated on the current image device. When that device is CUDA, the full image-sized tensor expressions execute on CUDA.

#### Edge and path coordinates

For a selected straight edge, the distance field is the pixel distance from that edge and the along-edge coordinate is the perpendicular image coordinate. For a selected corner, distance is Euclidean distance from that corner. The selected edge/corner establishes the origin of inward propagation.

Let \(d\) be distance modified by the seeded warp. Each mode computes a scalar spatial profile \(P(x,y)\), optical path distance \(d(x,y)\), and fixed mode gain \(g_m\).

#### Mode profile equations

**Soft Edge**

\[
d=\max(d_0+W,0),\quad
P=\exp\left[-\frac{d}{200sq}\right],\quad g_m=0.30
\]

where \(s=\max(H,W)/1024\). Its warp uses an 85-reference-pixel amplitude and 380-reference-pixel feature scale.

**First-Frame Burn**

The warped path uses a 155-pixel warp amplitude and a 260-pixel feature scale. Define \(w=95sq\):

\[
P_b=\exp\left[-\left(\frac{d}{1.75w}\right)^{1.28}\right]
\]
\[
P_h=\exp\left[-\frac{d}{\max(0.34w,1)}\right]
\]
\[
M=0.84+0.16\cos\left(2\pi\frac{a}{L_a}+0.013(seed\bmod997)\right)
\]
\[
P=(0.70P_b+0.30P_h)M,\quad g_m=0.55
\]

A separate near-white hot-core mixture is later applied to the incident color.

**120 Fat Roll**

A low-frequency 2D warp and 1D edge line alter the propagation path. Let the edge line be \(l\):

\[
d=\max(d_0+W+30sl,0)
\]
\[
w=(72+28(l+1))sq
\]
\[
P_m=\exp(-d/w)
\]
\[
P_s=\exp\left[-\left(\frac{d-2.15w}{\max(0.62w,1)}\right)^2\right]
\]
\[
P=(0.72P_m+0.28P_s)\left(0.78+0.22\frac{l+1}{2}\right),
\quad g_m=0.43
\]

**Light Piping**

A 52-pixel warp and a seeded 1D line produce the path and streak placement:

\[
d=\max(d_0+W,0)
\]
\[
T=\exp\left[-\frac{d}{0.88\max(H,W)q}\right]
\]
\[
P_v=0.20T\left(0.62+0.38\frac{l+1}{2}\right)
\]

Five irregular longitudinal channels are generated with Python's seeded random.Random; each channel adds a Gaussian-shaped streak along the selected edge, multiplied by \(T\). The combined field is clamped to \([0,1.6]\), with \(g_m=0.40\).

**Holga 135**

The mode generates a warped edge path and a diffusing sprocket/perforation modulation. Corner positions map to their corresponding top or bottom edge. Perforation pitch, initial spread and duty ratio are scaled to the image long edge. A five-period erf-based approximation forms the blurred perforation pattern; it is multiplied by an exponential path envelope:

\[
P=M_{\mathrm{sprocket}}(d,a,seed)\exp\left[-\frac{d}{200sq}\right],
\quad g_m=0.30
\]

#### Spectral attenuation and exposure

The leak color is converted from sRGB hex to linear RGB. Red, green and blue path ratios are fixed:

\[
\rho=(1.0,0.68,0.43)
\]

The reference red path length is:

\[
\lambda_R=200sq
\]

For channel \(c\):

\[
A_c(d)=\exp\left[-\frac{d}{\lambda_R\rho_c}\right]
\]

The selected incident spectrum is multiplied by spatial profile and spectral attenuation:

\[
F_c(x,y)=P(x,y)A_c(d)C_c
\]

For First-Frame Burn only, incident color is mixed toward white near the entry edge:

\[
h(d)=\exp\left[-\frac{d}{\max(80s,1)}\right]
\]
\[
C'=C(1-0.78h)+0.78h
\]

The final linear exposure is:

\[
RGB_{\mathrm{out}}=
\operatorname{sRGB}
\left(
\operatorname{clamp}
\left[RGB_l+g_m\min(e,10)F,\ 0,\ 1\right]
\right)
\]

**PyTorch mapping:** CPU seeded torch.rand for coarse control vectors; torch.Generator; device transfers of small vectors; bicubic/linear interpolation; arange, meshgrid, sqrt, exp, cos, erf, elementwise comparisons and masks, channel stack, and linear-light exposure arithmetic. Python random.Random selects a few deterministic Light Piping parameters on CPU; it does not generate the image-sized output.

### 6.7 Color Film Emulsion Atlas — core/emulsions.py

The atlas contains exactly 50 immutable creative profiles plus a separate Neutral identity selection:

| Family | Count |
|---|---:|
| Cinema | 10 |
| Instant | 10 |
| Slide / Reversal | 10 |
| 35mm Color Negative | 20 |
| **Total creative profiles** | **50** |

Each profile stores exposure offset \(e_p\), contrast coefficient \(c_p\), saturation multiplier \(s_p\), toe coefficient \(t_p\), shoulder coefficient \(h_p\), RGB channel gains \((g_R,g_G,g_B)\), shadow tint vector \(T_s\), and highlight tint vector \(T_h\).

These are independently authored parameters. Profile names are creative reference vocabulary and do not imply measured spectral, sensitometric, chemical or scanner equivalence.

#### Profile transform

The stage converts the input to linear RGB. Optional profile exposure is:

\[
RGB_0=RGB_l2^{e_p}
\]

Let \(Y=\operatorname{clamp}(Y_{709}(RGB_0),0,1)\). The profile luminance curve is:

\[
S_c=(c_p-1)4Y(1-Y)(Y-0.5)
\]
\[
T_c=t_p3Y(1-Y)^2
\]
\[
H_c=h_p3Y^2(1-Y)
\]
\[
Y_m=\operatorname{clamp}(Y+S_c+T_c-H_c,0,1)
\]

A common luminance ratio recombines the curve with the original RGB relationship:

\[
q=\operatorname{clamp}
\left(
\frac{Y_m}{\max(Y,10^{-6})},0,4
\right)
\]
\[
RGB_1=RGB_0q
\]

The mapped luminance is recomputed:

\[
Y_1=\operatorname{clamp}(Y_{709}(RGB_1),0,1)
\]

Saturation is adjusted around this shared luminance:

\[
RGB_2=Y_1+s_p(RGB_1-Y_1)
\]

Profile channel gains are applied:

\[
RGB_3=RGB_2\odot(g_R,g_G,g_B)
\]

Shadow and highlight tints use squared luminance masks:

\[
w_s=(1-Y_1)^2,\quad w_h=Y_1^2
\]
\[
RGB_p=RGB_3+w_sT_s+w_hT_h
\]

#### Emulsion Response

The public Emulsion Response \(r\) is clamped to \([0.5,3]\), default 1.0. It scales the complete profile deviation in linear light around the exact input to the emulsion stage:

\[
RGB_{\mathrm{response}}=RGB_l+r(RGB_p-RGB_l)
\]

At exactly \(r=1.0\), the blend branch is skipped and the established profile result is preserved. Values below one interpolate toward the stage input; values above one extrapolate the complete tonal and color response before final linear-to-sRGB conversion. Neutral returns the original stage input directly and ignores the response value.

The emulsion stage does not generate grain or procedural texture. Those operations remain independent and are applied later in the pipeline.

**PyTorch mapping:** scalar profile constants are assembled into small tensors on the current processing device; luminance reductions, square, clamp, clamp_min, pow, channel broadcasting, torch.tensor gains/tints and sRGB transfer functions.

### 6.8 Sabattier — core/sabattier.py

Sabattier works from linear Rec.709 luminance and modifies the luminance curve while retaining the source RGB relationship.

The visible Solarization control \(s\in[0,1]\) maps to:

\[
\beta=2s
\]

Highlight Confinement \(c\in[0,1]\) maps to exponent:

\[
k=0.5+3.5c
\]

The solarization curve is:

\[
f(Y)=\frac{Y\,10^{-\beta Y^k}}{N}
\]

The normalization \(N\) is the larger of the curve's calculated interior peak value and endpoint value. The analytical peak candidate is:

\[
Y_p=
\operatorname{clamp}
\left(
\left[\frac{1}{\beta k\ln(10)}\right]^{1/k},
10^{-6},1
\right)
\]
\[
N=\max\left(Y_p10^{-\beta Y_p^k},10^{-\beta},10^{-8}\right)
\]

If solarization is zero, \(f(Y)=Y\).

The backend retains an optional legacy Mackie Lines parameter even though it is not currently exposed as a visible frontend control. If enabled, it computes:

\[
\sigma=2\frac{\max(H,W)}{1024},\quad
Y_b=G(\sigma)*Y,\quad
D=Y-Y_b
\]

For Mackie amount \(m\in[0,1]\):

\[
g_+=0.65m,\quad g_-=g_+/6
\]
\[
\Delta=
\begin{cases}
g_+D & D>0\\
g_-D & D\le0
\end{cases}
\]
\[
Y_o=\max(f(Y)+\Delta,0)
\]

The code creates a shared RGB scale:

\[
q=Y_o/\max(Y,10^{-6})
\]

and gamut-limits it:

\[
q\leftarrow\min\left(q,\frac{1}{\max(\max_c RGB_{l,c},10^{-6})}\right)
\]
\[
RGB_{\mathrm{out}}=\operatorname{sRGB}(RGB_lq)
\]

**PyTorch mapping:** pow, scalar tensor power, base-10 exponent mapping, clamp, Gaussian blur, where, amax, luminance ratio and broadcast RGB multiply.

### 6.9 Soft Focus — core/soft_focus.py

Soft Focus is not a uniform global blur. It mixes a linear-light Gaussian blur into the source according to an aspect-correct radial field.

The maximum blur sigma is scaled to the source long edge:

\[
\sigma=r\frac{\max(H,W)}{1024}
\]

The RGB image is converted to linear light, blurred with the shared scaled Gaussian primitive, then combined with a normalized radial field.

Coordinates span \([-1,1]\) in both directions. Radial distance normalized from center to the four corners is:

\[
d=\operatorname{clamp}
\left(
\frac{\sqrt{x^2+y^2}}{\sqrt{2}},0,1
\right)
\]

The inner crisp region is removed from the blend ramp:

\[
t=\operatorname{clamp}\left(\frac{d-0.12}{0.88},0,1\right)
\]
\[
w=S(t)^{1.35}
\]

For Amount \(a\in[0,1]\):

\[
\alpha=aw
\]
\[
RGB_o=RGB_l(1-\alpha)+G(\sigma)*RGB_l\,\alpha
\]

The exact center receives no blur; contribution rises smoothly toward the corners. Channels after RGB are reattached unchanged.

**PyTorch mapping:** linear conversion, HWC/NCHW permutations, scaled Gaussian convolution/interpolation, linspace, meshgrid, sqrt, clamp, pow, broadcast blend and torch.cat when extra channels exist.

### 6.10 Vignette — core/vignette.py

Vignette is a linear-light exposure attenuation field. Radial coordinates are aspect-correct because each axis independently spans \([-1,1]\), then corner distance is normalized by \(\sqrt{2}\):

\[
d=\operatorname{clamp}
\left(
\frac{\sqrt{x^2+y^2}}{\sqrt{2}},0,1
\right)
\]

For Feather \(f\in[0,1]\), transition begins at:

\[
d_0=0.90-0.60f
\]
\[
t=\operatorname{clamp}
\left(\frac{d-d_0}{\max(1-d_0,10^{-6})},0,1\right)
\]
\[
v=t^2(3-2t)
\]

Amount \(a\in[0,2]\) maps to corner attenuation in stops:

\[
E=1.5av,\quad M=2^{-E}
\]
\[
RGB_o=\operatorname{sRGB}(RGB_lM)
\]

At Amount 2, corner attenuation reaches 3 stops. Extra channels after RGB are concatenated unchanged.

**PyTorch mapping:** linspace, meshgrid, sqrt, clamp, square, exp2, linear conversion, channel broadcasting and concatenation.

### 6.11 Procedural Texture — core/noise_texture.py

Procedural Texture creates a deterministic 2D scalar carrier from absolute source-image coordinates. The selected kernel defines surface structure; Amount controls signed-carrier amplitude; Seed selects the deterministic field; Scale controls feature size.

The twelve exposed families are Gaussian, Value fBm, Gradient fBm, Simplex fBm, Ridged fBm, Worley Microcells, Domain Warp, Silver Halide Clumps, Emulsion Mottle, Film Scratches, Dust & Fibers and Paper Foxing.

#### Integer hash and lattice sampling

The field uses 32-bit integer hashing carried out in signed int64 tensors with explicit 32-bit masking:

\[
h=(i_x\cdot73856093)\oplus(i_y\cdot19349663)
\oplus(seed\cdot83492791)\oplus(stream\cdot2654435761)
\]

After masking with \(2^{32}-1\), the code applies integer xor-shift and multiply mixing steps. A normalized value is:

\[
U=\frac{h+0.5}{2^{32}}
\]

The integer operations and bitwise expressions execute on the same PyTorch device as the coordinate tensors. They are included in the CUDA compatibility preflight.

The quintic fade curve is:

\[
F(t)=6t^5-15t^4+10t^3
\]

Value noise hashes four lattice corners into \([-1,1]\), then interpolates first along x and then y using this fade value. Gaussian lattice values use Box–Muller:

\[
z=\sqrt{-2\ln U_1}\cos(2\pi U_2)
\]

Gradient noise chooses a deterministic angle at each lattice point, forms gradient-dot-offset values at the four corners, and interpolates with the same quintic fade. Simplex noise uses 2D skew/unskew constants, three simplex corners, a compact-support attenuation term raised to the fourth power, and deterministic gradient directions.

#### Fractal Brownian motion

The shared fBm routine accumulates octave samples with frequency doubling and amplitude halving:

\[
F_{\mathrm{fbm}}(x,y)=
\frac{\sum_{o=0}^{n-1}2^{-o}N(2^ox,2^oy)}
{\sum_{o=0}^{n-1}2^{-o}}
\]

The standard path uses five octaves. Some composite kernels select three or four octaves and independent stream offsets.

#### Worley and composite kernels

Worley Microcells searches the 3×3 neighborhood of the current lattice cell. Each candidate feature point is offset within its cell by two independently hashed coordinates. The minimum Euclidean distance to those nine candidates forms the cellular field.

| Family | Current construction |
|---|---|
| Gaussian | Gaussian lattice noise interpolated over the coordinate field, sampled at 32× internal frequency |
| Value fBm | Five-octave value noise |
| Gradient fBm | Five-octave gradient noise |
| Simplex fBm | Five-octave simplex noise |
| Ridged fBm | \(2(1-|F_{\mathrm{gradient}}|)-1\) |
| Worley Microcells | \(1-2D_{\mathrm{nearest}}\) using the nearest of nine feature points |
| Domain Warp | Two decorrelated 3-octave gradient-fBm fields displace simplex-fBm sampling coordinates |
| Silver Halide Clumps | Four-octave simplex clump density modulates a Gaussian carrier; part of the clump field is added back |
| Emulsion Mottle | Weighted broad simplex, middle value-noise and fine Gaussian fields |
| Film Scratches | Sparse hashed longitudinal columns, smooth lateral drift, irregular continuity and fine Gaussian detail |
| Dust & Fibers | Seeded sparse circular particles and oriented anisotropic fibre marks, plus a fine Gaussian carrier |
| Paper Foxing | Broad value-fBm cloud, gradient-fBm edge irregularity, sigmoid stain coverage and fine pore noise; returns a scalar field and stain mask |

Film Scratches uses a sparse column-selection hash and a Gaussian cross-section around an irregular centerline. Dust & Fibers evaluates nine neighboring cells, placing deterministic particles and fibres using hashes for presence, position, radius and orientation. Paper Foxing forms an irregular stain mask with a sigmoid over broad and edge-perturbation fields.

#### Coordinate mapping, seed and normalization

For reference dimensions \((H_r,W_r)\) and coordinate origin \((o_x,o_y)\), the source coordinate grid is:

\[
\nu=\frac{32}{\max(H_r,W_r)\max(scale,0.05)}
\]
\[
x=(arange(W)+o_x)\nu,\qquad y=(arange(H)+o_y)\nu
\]

Scale is clamped to \([0,4]\); the internal floor of 0.05 avoids division by zero at the finest setting. Default reference dimensions are the current image dimensions and origin is (0,0). Optional arguments allow preview crops to retain absolute source coordinate alignment.

For batch frame \(b\), the field seed is:

\[
seed_b=(seed+7919b)\ \&\ 0x7fffffff
\]

After field generation, the scalar field is mean-centered and normalized by RMS:

\[
N=\frac{F-\operatorname{mean}(F)}
{\max\left(\sqrt{\operatorname{mean}((F-\operatorname{mean}(F))^2)},10^{-5}\right)}
\]

The code mean-centers the field and divides by its root-mean-square magnitude. This normalizes average carrier energy between texture families and resolutions; it does not force the field to be bounded to [-1,1].

#### Tonal modulation and signed layer

The source is interpreted as sRGB and converted to linear Rec.709 luminance \(Y\). The tonal weight is:

\[
w_Y=0.42+0.58\operatorname{clamp}(4Y(1-Y),0,1)
\]

For Amount \(a\in[0,2]\):

\[
D=N(0.018a)w_Y
\]
\[
L=D[...,None]
\]

For Paper Foxing, the separate stain mask \(P_f\) also creates a warm/cool chromatic stain:

\[
d_f=0.012aP_fw_Y
\]
\[
L\leftarrow L+(d_f,\ 0.22d_f,\ -0.72d_f)
\]

The signed layer is composited through the selected texture blend mode. Additive is the default.

**PyTorch mapping:** int64 coordinate arithmetic and bitwise hash expressions, floor, to(int64), lattice indexing, sqrt, log, cos, sin, exp, sigmoid, tensor comparisons, where, minimum, maximum, mean, RMS normalization, stack, meshgrid, arange, broadcast arithmetic and shared blending.

### 6.12 Film Grain — core/grain.py

Film Grain is independent from Procedural Texture and Light Leak. It has a fixed internal seed (42) plus a deterministic batch-frame offset. UI Noise Seed and Light Leak Seed do not alter it.

#### Coordinate-hashed Gaussian field

For each source pixel coordinate \((x,y)\), channel stream \(c\) and seed \(s\), the engine builds an int64 coordinate hash and applies 32-bit avalanche mixing. Two hash values are converted to uniform samples:

\[
u_1=(h_1+0.5)/2^{32},\quad u_2=(h_2+0.5)/2^{32}
\]

A normally distributed sample is generated using Box–Muller:

\[
z=\sqrt{-2\ln(\max(u_1,10^{-7}))}\cos(2\pi u_2)
\]

The field is anchored to absolute pixel coordinates, so crop evaluation is spatially consistent when the same origin is provided.

#### Correlation and color grain

The correlation sigma is:

\[
\sigma=\max\left(0,\ grainSize\frac{\max(H_r,W_r)}{1024}0.50\right)
\]

If \(\sigma>0.35\), each noise field is softened using the shared Gaussian blur. Each independent channel field and the shared field are mean-centered and RMS-normalized.

Let \(G\) be the shared field and \(I_c\) the independent per-channel fields. Color Grain \(c_g\in[0,1]\) mixes them:

\[
N_c=(1-c_g)G+c_gI_c
\]

At zero, all RGB channels share a luminance-grain field. At one, each channel has an independent field.

#### Tonal grain amplitude

The image's linear Rec.709 luminance controls grain visibility:

\[
w_Y=0.28+0.72\operatorname{clamp}(4Y(1-Y),0,1)
\]

Amount \(a\in[0,3]\) sets the maximum carrier scale:

\[
A=0.022a,\quad L_c=N_cAw_Y
\]

The signed-grain carrier is composited through the selected Grain Blend Mode. Additive is the default. At Amount 3 the nominal carrier scale reaches 0.066 before tonal weighting and final blending/clamping.

**PyTorch mapping:** int64 coordinate hashing, integer shifts/xor/multiplication, log, sqrt, cos, optional Gaussian grouped convolution, mean/RMS reductions, stack, sRGB-to-linear luminance, signed-layer broadcasting and shared blending.

### 6.13 Shared signed-layer blend modes — core/blending.py

Procedural Texture and Film Grain use independent selectors but the same twelve compositor equations. Let source channel be \(S\in[0,1]\), and signed carrier be \(D\). For Overlay, Soft Light and Hard Light, layer variable is \(L=0.5+D\).

| Mode | Equation applied per channel |
|---|---|
| Additive | \(S+D\) |
| Multiply | \(S(1+D)\) |
| Screen | \(1-(1-S)(1-D)\) |
| Overlay | \(2SL\) if \(S\le0.5\), otherwise \(1-2(1-S)(1-L)\) |
| Soft Light | Piecewise soft-light expression using source-dependent \(d(S)\) |
| Hard Light | \(2SL\) if \(L\le0.5\), otherwise \(1-2(1-S)(1-L)\) |
| Color Dodge | \(S/\max(1-D,10^{-4})\) |
| Color Burn | \(1-(1-S)/\max(1+D,10^{-4})\) |
| Lighten | \(\max(S,S+D)\) |
| Darken | \(\min(S,S+D)\) |
| Difference | \(|S-D|\) |
| Exclusion | \(S+D-2SD\) |

Soft Light uses:

\[
d(S)=
\begin{cases}
((16S-12)S+4)S & S\le0.25\\
\sqrt{\max(S,0)} & S>0.25
\end{cases}
\]

Then:

\[
R=
\begin{cases}
S-(1-2L)S(1-S) & L\le0.5\\
S+(2L-1)(d(S)-S) & L>0.5
\end{cases}
\]

Every mode's final result is clamped to \([0,1]\). Invalid mode names normalize to Additive. The signed carrier is centered around the compositor's neutral-layer convention. These modes therefore create different nonlinear responses; they are not simply alternate names for additive grain.

**PyTorch mapping:** torch.where, torch.sqrt, torch.maximum, torch.minimum, torch.abs, clamp_min, division and multiply/add/subtract expressions.

## 7. Execution engine, batch strategy and dtype

### 7.1 Identity path

Before selecting a device, is_identity checks the stage settings. Identity requires Film Grain amount zero, Procedural Texture amount zero, all optical and tonal effect amounts at bypass, both White Balance axes exactly zero, Anti-AI and Exposure zero, and Color Film Emulsion set to Neutral.

When identity is true, the exact input tensor object is returned. No image arithmetic, float32 conversion, output allocation, CUDA probe or image-device transfer is performed.

The public node still returns a report through process_batch_with_report. The tensor-only process_batch API returns immediately without diagnostic overhead.

### 7.2 Sequential frame execution

For non-identity settings, _process_batch_on_device preallocates one output tensor with the input BHWC shape, dtype and selected output device. It loops over batch frames:

1. Take one frame view.
2. Select its first three RGB channels.
3. Move only RGB to the processing device.
4. Run process_single_image with the frame index.
5. Copy processed RGB to the output device and input dtype.
6. Copy channels after RGB directly from source to output.

Frames are not accumulated in a Python list and stacked afterward. This bounds simultaneous image working memory to one actively processed RGB frame plus the output tensor and stage temporaries.

The main pipeline is wrapped in torch.no_grad(). Retrolab does not build autograd graphs for photographic finishing.

### 7.3 Precision and device behavior

All image-engine conversion/working operations explicitly use float32 where specified. Retrolab does not expose or enable half precision, bfloat16, automatic mixed precision, global TF32 changes or a user-facing precision selector.

- CUDA-origin input normally remains on its incoming CUDA device and returns there.
- CPU-origin input may be temporarily processed on CUDA one frame at a time, then each completed frame is copied back to CPU.
- CPU-only PyTorch remains supported.
- Additional channels do not travel to the processing GPU.
- CPU preview rendering is an independent path and does not use CUDA.

The planner does not rewrite image mathematics. The exact same stage functions are used regardless of processing device.

## 8. CUDA acceleration: what runs on the GPU

This section answers which expressions are accelerated by CUDA in Retrolab.

Retrolab does not contain hand-written CUDA C++, Triton kernels, custom fused kernels, torch.compile graph capture or a bespoke GPU shader pipeline. It uses PyTorch tensor operations. When a stage receives CUDA-resident tensors, PyTorch dispatches supported operations to its CUDA backend. The same Python expression can therefore run on CPU or CUDA according to the tensor device.

The primary CUDA benefit is parallel execution of operations over the image pixel grid, together with CUDA implementations of convolution, pooling, interpolation and FFT. Exact kernel implementation and fusion behavior depend on the installed PyTorch build, GPU architecture and runtime. Retrolab does not promise that a sequence of Python tensor expressions is fused into one kernel.

### 8.1 CUDA-accelerated expression families

| Operation family | Representative PyTorch operations in Retrolab | Where used |
|---|---|---|
| Float32 elementwise arithmetic | add, subtract, multiply, divide, power, square, clamp, clamp_min, exp, exp2, log, sqrt, sin, cos, atan, asin, erf, sigmoid, abs, remainder, min/max, comparison masks, where | All tonal, optical, procedural and compositing stages |
| Reductions | sum, mean, amax, amin, tensor normalization and luminance-related reductions | Luminance, mask normalization, hash/noise normalization and kernel normalization |
| Integer tensor arithmetic | int64 multiply/add/xor/shift/and, floor-to-int64, remainder, coordinate hashing | Procedural Texture, Film Grain, Light Leak coordinate/seed fields |
| Tensor construction and indexing | arange, linspace, meshgrid, stack, cat, unbind, view, permute, broadcasting and indexing | Spatial coordinate fields, channel construction, layouts and tensor assembly |
| Convolution | torch.nn.functional.conv2d, including grouped separable convolution | Gaussian blur used by Bloom, Soft Focus, Mackie contours and correlated grain |
| Padding | torch.nn.functional.pad with replicate mode | Gaussian filtering |
| Pooling | torch.nn.functional.max_pool2d | Halation downsampling while retaining small bright sources |
| Interpolation/resampling | torch.nn.functional.interpolate with area, nearest, bicubic, linear and bilinear modes | Scaled blur, Light Leak control fields and Halation working-resolution handling |
| FFT | torch.fft.rfft2, complex multiply, torch.fft.irfft2 | Halation channel convolution |
| Device transfers | Tensor.to(device=...) | RGB frame movement for CPU-origin CUDA workloads and return copies |

The compatibility preflight explicitly exercises representative members of these families: float32 arithmetic, int64 deterministic-index arithmetic, convolution, max pooling, replicate padding, FFT, interpolation and erf. It is a tiny technical probe, not a photographic render or performance benchmark.

### 8.2 Stage-to-CUDA mapping

| Stage / engine | Main GPU work when processing tensor is CUDA |
|---|---|
| White Balance | sRGB transfer functions, channel gain construction, luminance-normalizing scalar math and RGB multiply |
| Anti-AI | luminance and hue/saturation masks, smoothstep, exponentials, channel extrema/reductions, common RGB scaling and chroma reconstruction |
| Exposure | transfer functions and per-pixel linear-light interpolation toward black or white |
| Bloom | luminance extraction, soft-knee threshold expression, channel scaling, two Gaussian blur passes at two scales and highlight addition |
| Halation | kernel tensor construction, max-pool reduction when downsampling, FFT transforms and complex spectrum multiplication, channel weighting and output resize |
| Light Leak | image-sized coordinate grids, low-frequency field interpolation after CPU-grid transfer, distance/warp/profile equations, erf-based sprocket mask, spectral attenuation and RGB exposure addition |
| Color Film Emulsion | luminance curve, per-pixel RGB scaling, saturation, channel gains, shadow/highlight tint and Emulsion Response blend |
| Sabattier | luminance curve, power/exp math, optional Gaussian high-pass blur, piecewise Mackie response and hue-ratio reconstruction |
| Soft Focus | linear conversion, Gaussian blur, spatial radial meshgrid/mask and per-pixel blend |
| Vignette | radial coordinate field, smoothstep and exp2 attenuation |
| Procedural Texture | int64 hash arithmetic, lattice sampling, fBm and cellular/particle fields, image-sized texture normalization, luminance weighting and blend compositing |
| Film Grain | int64 pixel hash, Box–Muller noise, optional Gaussian correlation, RMS normalization, luminance weighting and blend compositing |

### 8.3 What remains on CPU or is not a GPU-specific optimization

Not every instruction in the node is a CUDA operation:

- Python control flow, settings normalization, stage selection and report assembly execute on CPU.
- Small seeded CPU random grids in Light Leak are created on CPU and transferred to the selected image device for image-sized interpolation and evaluation.
- Python random.Random selects a small number of deterministic Light Piping geometry parameters on CPU.
- The node DOM frontend, browser preview interaction and HTTP request handling execute outside the PyTorch image CUDA path.
- Preview source preparation, preview rendering and Pillow PNG encoding use CPU-resident reduced data.
- An identity execution intentionally performs no GPU probing or GPU arithmetic.
- CUDA device queries, compatibility preflight orchestration and allocator telemetry are host-side control operations, even though the compatibility probe launches small CUDA operations.

### 8.4 CUDA acceleration is not custom kernel fusion

The code is written as readable PyTorch tensor mathematics. A stage can issue multiple elementwise kernels for successive expressions. PyTorch may internally select optimized library implementations for some operators, but Retrolab does not explicitly fuse the stages into a custom kernel. The GPU route accelerates supported tensor operations; it does not turn the entire 12-stage Python pipeline into one shader or one CUDA launch.

No photographic stage changes its formula because CUDA is selected. No reduced precision is introduced by the execution manager.

## 9. Automatic CPU/CUDA planning

core/execution.py is an internal service. There is no CPU/CUDA switch, Processing Device selector, VRAM slider or performance mode in the creative interface.

### 9.1 Device-preservation rules

1. Identity settings return before any hardware query.
2. CUDA-origin tensors normally remain on their input CUDA device after compatibility preflight.
3. Non-CPU, non-CUDA device types are preserved.
4. CPU-origin tensors may use CUDA only if there is meaningful work, workload size is large enough, CUDA is available and compatible, and available memory exceeds the conservative estimate plus reserve.
5. CPU is the standard route when CUDA is unavailable, the workload is small, compatibility fails or free VRAM is insufficient.

### 9.2 Workload threshold

The planner distinguishes pointwise and spatial workloads.

\[
P=H\cdot W
\]

- Pointwise corrections require at least 8,000,000 pixels before CPU-origin work is considered worthwhile to transfer.
- Spatial workloads require at least 2,500,000 pixels.

A workload is spatial when one or more of Bloom, Halation, Light Leak, Soft Focus, Vignette, Procedural Texture or Film Grain is active. A meaningful workload must contain at least one active transform.

These thresholds are transfer-economics gates, not GPU capability thresholds. They avoid transferring small CPU-origin jobs when host/device transfer and initialization can outweigh arithmetic.

### 9.3 Conservative one-frame working-set estimate

The planner estimates RGB float32 frame bytes:

\[
M_{\mathrm{RGB}}=H\cdot W\cdot3\cdot4
\]

The workspace multiplier starts at 5.5 and adds effect-specific allowances:

| Active stage | Added multiplier |
|---|---:|
| Bloom | 1.50 |
| Halation | 2.00 |
| Soft Focus | 1.25 |
| Procedural Texture | 1.25 |
| Film Grain | 0.75 |
| Light Leak | 0.50 |
| Vignette | 0.50 |
| Anti-AI | 0.50 |
| Color Film Emulsion | 0.50 |
| Sabattier / Mackie Lines | 0.50 |
| White Balance | 0.25 |
| Exposure | 0.25 |

The multiplier is capped at 14.0. A fixed 128 MiB allowance is added:

\[
M_{\mathrm{estimate}}=
M_{\mathrm{RGB}}\cdot\min(M_{\mathrm{multiplier}},14)
+128\ \mathrm{MiB}
\]

This is an admission estimate for a single frame, not a measured peak allocation and not a guarantee of exact allocator use. Actual peak depends on intermediate tensors, library workspaces and runtime state.

### 9.4 VRAM reserve and approval

For CPU-origin offload, Retrolab reserves the larger of:

\[
M_{\mathrm{reserve}}=\max(1\ \mathrm{GiB},0.10M_{\mathrm{total}})
\]

Usable free memory is:

\[
M_{\mathrm{usable}}=\max(0,M_{\mathrm{free}}-M_{\mathrm{reserve}})
\]

CUDA is approved only when the memory query succeeds and:

\[
M_{\mathrm{estimate}}\le M_{\mathrm{usable}}
\]

The reserve leaves room for ComfyUI models and other GPU users. It is not a lock on memory; concurrent allocation can still change available memory after preflight.

### 9.5 CUDA compatibility preflight

Before the first eligible CUDA image route for a runtime/device signature, a small probe validates representative operations required by Retrolab:

- float32 tensor math,
- int64 deterministic-index arithmetic,
- convolution,
- max pooling,
- replicate padding,
- FFT,
- bilinear interpolation,
- error function.

A successful result is cached using a signature containing PyTorch version, CUDA runtime, device index, device name, compute capability and total device memory. Failures are not cached so a later execution can retry a transient initialization problem. A failed preflight selects CPU before transferring the user image.

The preflight uses synthetic small tensors. It does not touch the source photograph or change global PyTorch precision, allocator settings, thread settings or ComfyUI device configuration.

### 9.6 Allocation failure and whole-batch recovery

The CUDA attempt is wrapped around the complete batch operation. If and only if the selected route is CUDA and the exception is recognized as an allocation failure (PyTorch CUDA OOM, out-of-memory markers, cuFFT allocation failure or cuDNN allocation failure):

1. The partial CUDA result is discarded.
2. The exception traceback is cleared to release references to failed worker tensors.
3. Python garbage collection and torch.cuda.empty_cache() are attempted best-effort.
4. The complete input batch is processed again on CPU.
5. The actual processing device and fallback reason are reflected in the report.

Other errors propagate. The system does not hide programming errors or non-memory CUDA failures behind a generic CPU retry.

If a CUDA-origin input requires exceptional CPU recovery, the output is CPU-resident. For a CPU-origin input temporarily offloaded to CUDA, completed frames normally return to CPU as they are written.

## 10. Execution Report

The node returns an optional-to-connect STRING socket named report. The JSON schema identifier is:

zuanfilm.retrolab.execution-report.v1

The report includes:

- package version, UTC timestamp and result status;
- input/output BHWC dimensions, megapixels, channels, batch count, dtype, device, contiguity and strides;
- internal processing dtype;
- planned device, actual device, output device, planner reason and frame strategy;
- identity bypass flag and whether fallback occurred;
- elapsed full-resolution pipeline time, average time per frame and megapixels per second;
- active processing stage names in pipeline order;
- complete RetroLabSettings snapshot;
- CUDA availability, device name, runtime, compatibility status and error details;
- preflight free/total VRAM, estimated working set, reserve and usable memory;
- before/after allocator and global-memory snapshots when available;
- Python, PyTorch, operating-system and thread-count details;
- best-effort telemetry errors.

Timing scope is the full-resolution Retrolab image pipeline. It excludes preview-cache construction, frontend drawing, upstream graph nodes and downstream graph nodes. For CUDA execution, synchronization is attempted before timing/report completion so the measured interval includes queued CUDA work.

The memory estimate is explicitly labelled as an estimate. Before/after allocator snapshots are not presented as measured peak per-node allocation. Telemetry failures are best-effort and do not invalidate an otherwise successful image output.

Retrolab returns JSON text through the workflow output only. It does not automatically save reports, create files or write external logs.

The backwards-compatible tensor-only process_batch API remains available and does not request report telemetry.

## 11. Interactive Preview Architecture

The preview system is separate from the full-resolution execution path. It is designed to update creative controls without requiring the user to queue the entire ComfyUI workflow for every slider movement.

### 11.1 Preview source creation

A normal node execution creates a random 128-bit preview token and stores compact CPU-resident source data:

1. A reduced composition proxy, with a maximum long edge of 512 pixels.
2. A native-resolution center crop up to 240 × 240 pixels, including surrounding context for texture generation.

The full-resolution source frame is not retained in the preview cache. The node returns the token through its UI payload as retrolab_preview_id.

### 11.2 Preview views

**Composition Before** is the untouched reduced source proxy.

**Composition After** applies the relevant photographic stages to the reduced proxy, including White Balance, Anti-AI, Exposure, Bloom, Halation, Light Leak, Color Film Emulsion, Sabattier, Soft Focus and Vignette. Procedural Texture and Film Grain are bypassed in the reduced composition so downscaling does not exaggerate their particle structure.

**Grain & Texture Detail** renders the center crop at source scale. Procedural Texture is applied first, then Film Grain. This view is a focused texture inspection, not a second full color-emulsion composition.

The actual full-resolution ComfyUI IMAGE output continues to apply all enabled stages in the normal pipeline order.

### 11.3 Frontend request sequence

~~~mermaid
sequenceDiagram
    participant N as Retrolab Node
    participant C as CPU Preview Cache
    participant U as DOM Frontend
    participant A as Preview API
    N->>C: Store reduced proxy and source-scale crop/context
    N-->>U: retrolab_preview_id
    U->>A: First POST with settings and original request
    A->>C: Resolve token
    A->>A: CPU render of Composition After
    A->>A: CPU render of Grain & Texture Detail
    A-->>U: Original PNG once and two processed PNGs
    U->>U: Cache original reduced PNG for token
    U->>U: Render Before/After selection
    U->>A: Later POST with changed settings
    A-->>U: Updated processed PNGs only
    U->>U: Replace After and detail images
    U->>A: DELETE token on node removal
~~~

Control changes are debounced by 180 ms. The original reduced proxy is sent once per token and cached in browser memory; later settings updates do not retransmit it. Preview rendering is CPU-based and Pillow encodes the small PNG outputs. The preview cache stores up to six entries, each expiring after 30 minutes.

The preview source is created during node execution. The first queued execution after installation is required before Live Preview and Grain & Texture Detail can be used. Queue the workflow again when the input image changes or when the cached source is unavailable/expired. Changing creative controls alone does not rerun upstream graph nodes.

### 11.4 Fail-open behavior

Preview creation, preview HTTP requests and preview rendering are fail-open. If they fail, the primary full-resolution image processing result remains available. Expired or missing preview tokens cause the frontend to request a new node execution rather than failing the ComfyUI IMAGE output.

## 12. Frontend and workflow compatibility

The frontend extension builds non-serialized DOM controls for Live Preview, Before/After comparison, emulsion selector, White Balance wheel, grouped effect controls, color pickers, localization and Grain & Texture Detail.

The visible interface writes through to the corresponding ComfyUI backend widgets. Backend widget input order is maintained append-only for later controls so saved workflows retain their established widget indices. Localized option labels map back to canonical backend values.

The language preference is stored in browser-local storage and is not a ComfyUI widget value or serialized workflow field. The frontend does not control the processing device. There are no CPU/CUDA selectors, VRAM controls or technical performance modes.

## 13. Determinism and reproducibility

For the same source data, dimensions, settings, seeds, device and compatible PyTorch/runtime environment, procedural stages are deterministic by construction:

- Light Leak uses fixed seeds for coarse geometry and Python geometry parameters, with the same profile reused across batch frames.
- Procedural Texture uses integer coordinate hashing with a seed and deterministic frame offset.
- Film Grain uses its fixed internal seed plus deterministic frame offset, with independent stream values for common and channel-specific fields.
- The emulsion atlas contains immutable scalar parameters and no random component.
- White Balance, Anti-AI, Exposure and optical/tone transforms contain no stochastic sampling.

Exact cross-device or cross-version bitwise equality is not guaranteed. Floating-point FFT, interpolation, convolution and reduction implementations can differ by device, library version or supported kernel. Determinism here means stable procedural inputs and repeatable results within a fixed execution environment, not universal bit-for-bit identity across all backends.

## 14. Performance and memory characteristics

- Full-resolution batch processing is sequential, one frame at a time.
- Output storage is preallocated once for the batch.
- Only RGB channels are transferred to a temporary CUDA processing device; additional channels are copied directly.
- CPU-origin outputs remain CPU-resident after each processed frame returns from CUDA.
- CUDA-origin inputs normally stay on their existing CUDA device.
- Bloom and Soft Focus use scaled Gaussian blur to bound large convolution kernels.
- Halation reduces its working resolution above a 1536-pixel long edge and uses FFT convolution; its kernel cache is bounded to six entries.
- Procedural Texture and Film Grain use coordinate-hashed fields rather than storing persistent random image maps.
- Preview processing uses reduced CPU data and never reserves CUDA memory for browser preview.
- No custom CUDA binaries, external CUDA extensions or additional Python packages are shipped.

The planner's working-set estimate is a safety/admission heuristic, not a benchmark or exact allocator model. Real timings depend on active stages, resolution, batch size, current VRAM occupancy, CPU/GPU transfer costs, PyTorch build, drivers and hardware.

## 15. Validation and engineering boundaries

The repository test suite covers processing engines, identity behavior, batch and channel preservation, determinism, supported parameter boundaries, node contract, preview cache validation, preview rendering and routes, and mocked frontend interaction.

The tests establish software-level behavior in their test environment. They do not replace validation inside an actual ComfyUI host, with the target PyTorch build, GPU, driver, frontend and competing GPU workloads.

For release or maintenance review:

- Validate formulas against their source functions when modifying a stage.
- Keep documented processing order synchronized with process_single_image.
- Keep CUDA planner thresholds and memory-estimate allowances synchronized with core/execution.py.
- Test CPU execution independently of CUDA.
- Test actual CUDA compatibility for all required operation families.
- Confirm an allocation failure retries the whole batch and non-allocation errors still propagate.
- Verify input/output shape, dtype and device expectations, including extra channels.
- Verify preview failure cannot interrupt IMAGE processing.
- Update this document when an algorithm, parameter mapping, tensor contract, execution rule or preview behavior changes.

## 16. Source-of-truth map

| Concern | Source |
|---|---|
| Public inputs, defaults, output sockets and settings construction | nodes/zuanfilm_retrolab.py |
| Fixed order, identity test, float32 conversion, frame loop and batch APIs | core/pipeline.py |
| Device selection, thresholds, memory estimate and compatibility preflight | core/execution.py |
| sRGB and Rec.709 functions | core/color.py |
| White Balance gain equations | core/white_balance.py |
| Anti-AI masks and tonal/chroma correction | core/anti_ai.py |
| Exposure response | core/exposure.py |
| Bloom highlight extraction and glow composition | core/bloom.py |
| Halation kernel, FFT and channel response | core/halation.py |
| Light Leak geometry, mode fields and spectral attenuation | core/light_leak.py |
| Emulsion profile data and Emulsion Response | core/emulsions.py |
| Sabattier and Mackie contour equations | core/sabattier.py |
| Soft Focus blend mask and blur | core/soft_focus.py |
| Vignette falloff | core/vignette.py |
| Procedural Texture fields and deterministic hash | core/noise_texture.py |
| Film Grain hash, correlation and tonal amplitude | core/grain.py |
| Texture/Grain blend-mode equations | core/blending.py |
| Gaussian filtering primitives | core/filters.py |
| Preview cache and routes | preview.py |
| DOM controls, localization and preview interaction | js/zuanfilm_retrolab.js |

