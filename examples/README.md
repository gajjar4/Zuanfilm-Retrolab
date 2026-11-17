# Example workflow

This folder contains a ready-to-open ComfyUI example for Zuanfilm Retrolab.

## Included files

- `workflows/Zuanfilm_Retrolab.json` — the supplied functional workflow, with the original node settings and connections preserved.
- `images/Zuanfilm-A1213345.jpg` — the input photograph referenced by the workflow's Load Image node.

## Run the example

1. Install Zuanfilm Retrolab in `ComfyUI/custom_nodes/Zuanfilm-Retrolab/` and restart ComfyUI.
2. Copy `Zuanfilm-A1213345.jpg` from this folder's `images/` directory into `ComfyUI/input/`.
3. Load `workflows/Zuanfilm_Retrolab.json` in ComfyUI.
4. Confirm that the Load Image node resolves `Zuanfilm-A1213345.jpg`, then queue the workflow.

The workflow includes ComfyUI Core's Load Image, Compare Images and Save Image (Advanced) nodes. The image output is connected to both comparison and saving. The Retrolab `report` output is available for optional diagnostics but is not required to run the example.

## Image handling

The packaged JPEG keeps its original dimensions (1200 × 1600 px), decoded pixels and embedded ICC profile (3144 bytes). Descriptive EXIF/XMP/Photoshop/comment metadata was removed from the public example copy; the image content was not resized or recompressed.
