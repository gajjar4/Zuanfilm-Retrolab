# Technical References

This document lists technical literature and platform documentation relevant to selected Retrolab implementation details. References provide background context; they do not imply that source code or a complete algorithm has been copied into the project.

- [Newson et al., Realistic Film Grain Rendering (IPOL, 2017)](https://doi.org/10.5201/ipol.2017.192) — research context for stochastic film-grain modeling. Retrolab uses a purpose-built correlated-noise implementation with coordinate-based deterministic variation; it does not claim full equivalence to the paper's model.
- [ComfyUI Frontend Extension API](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/docs/extensions/core.md) — platform documentation for frontend extension registration and DOM-based UI integration.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — tensor operations, device handling, interpolation and FFT primitives used by the image-processing engines.

The optical and tonal effects are documented as purpose-built photographic approximations with defined processing assumptions. They should not be interpreted as calibrated measurements of a specific film stock, camera body or chemical development process.


## Public film-reference vocabulary for the Color Film Emulsions Atlas

The named profiles use publicly documented film families and products as reference vocabulary. Their visual parameters are original creative interpretations rather than reproductions of manufacturer sensitometric or spectral data.

- [Kodak Motion Picture Camera Films](https://www.kodak.com/en/motion/products/camera-films/) — current motion-picture families and formats, including VISION3 50D, 200T, 250D and 500T, EKTACHROME 100D and VERITA 200D.
- [Kodak VISION3 50D Technical Data](https://www.kodak.com/content/products-brochures/Film/VISION3_50D_5203_TI-2657_Technical-Data_EN.pdf) — daylight balance, fine-grain positioning, highlight latitude and post-production flexibility.
- [Kodak Still Film — Professional](https://www.kodak.com/en/still-film/products/professional/) and [Consumer](https://www.kodak.com/en/still-film/products/consumer/) — EKTAR 100, EKTACHROME E100, GOLD 200, ULTRAMAX 400 and professional negative film descriptions.
- [Fujifilm Professional Film](https://www.fujifilm.com/us/en/business/professional-photography/film) — FUJICHROME reversal and FUJICOLOR negative film families.
- [FUJICHROME Velvia 50](https://www.fujifilm.com/us/en/business/professional-photography/film/velvia-50) — documented high saturation and vivid color.
- [FUJICHROME Provia 100F](https://www.fujifilm.com/us/en/business/professional-photography/film/provia-100f) — faithful color, medium saturation and contrast, fine grain and smooth gradation.
- [Fujifilm Professional Film Data Guide](https://asset.fujifilm.com/www/us/files/2020-03/85d928f44b0df3b2a95913e46608881d/ProfessionalFilmDataGuide.pdf) — historical color-reversal and color-negative product families, including Velvia, Provia, Astia, Sensia, Superia and PRO 400H.
- [Polaroid: What film do I use with Polaroid cameras?](https://support.polaroid.com/hc/en-us/articles/115012394347-What-film-do-I-use-with-Polaroid-cameras) — distinct SX-70 and 600 film compatibility and sensitivity context.
- [CineStill Film FAQ](https://cinestill.film/pages/frequently-asked-questions) — CineStill naming and motion-picture film lineage context.

The existence of a stock name or a published family description does not imply a calibrated match. Retrolab does not include licensed manufacturer film curves, spectral sensitivity data, chemical process models or scanner profiles.
