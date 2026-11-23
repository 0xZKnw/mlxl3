//! Bounded image input for the bridge and OpenAI-compatible API.
use crate::lfm3::{PreparedImage, patch_index};
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ImageDecoder, ImageReader, imageops::FilterType};
use serde_json::{Value, json};
use std::io::Cursor;

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_IMAGES: usize = 8;
use crate::image_budget::{MAX_IMAGE_TOKENS, MAX_PIXELS, fits_image_budget};

#[cfg(test)]
thread_local! { static PATCH_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

pub fn validate_content(message: &Value) -> Result<()> {
    if let Some(images) = message.get("images").filter(|v| !v.is_null()) {
        let images = images.as_array().context("images must be an array")?;
        ensure!(
            images.len() <= MAX_IMAGES
                && message["role"] == "user"
                && message["content"].is_string()
                && images.iter().all(|v| v
                    .as_str()
                    .is_some_and(|u| u.starts_with("data:image/") && u.contains(";base64,"))),
            "invalid message images"
        );
    }
    match &message["content"] {
        Value::String(_) => {}
        Value::Array(parts) => {
            ensure!(!parts.is_empty(), "message content parts cannot be empty");
            for part in parts {
                match part["type"].as_str() {
                    Some("text") => {
                        ensure!(part["text"].is_string(), "text part requires text");
                    }
                    Some("image_url") => {
                        ensure!(
                            message["role"] == "user",
                            "only user messages can contain images"
                        );
                        ensure!(
                            part["image_url"]["url"].as_str().is_some_and(|u| u
                                .starts_with("data:image/")
                                && u.contains(";base64,")),
                            "images require a base64 image data URL"
                        );
                    }
                    _ => bail!("unsupported message content part"),
                }
            }
        }
        _ => bail!("message requires text or image content"),
    }
    Ok(())
}

/// Same factor/min/max and rounding as LiquidAI's smart_resize.
pub fn resized_shape(height: u32, width: u32) -> Result<(u32, u32)> {
    ensure!(
        height > 0 && width > 0 && height.max(width) as f64 / height.min(width) as f64 <= 200.,
        "invalid image dimensions or aspect ratio"
    );
    let (h, w) = (f64::from(height), f64::from(width));
    let mut hh = (h / 32.).round_ties_even() * 32.;
    let mut ww = (w / 32.).round_ties_even() * 32.;
    if hh * ww > MAX_PIXELS as f64 {
        let beta = (h * w / MAX_PIXELS as f64).sqrt();
        hh = (h / beta / 32.).floor().max(1.) * 32.;
        ww = (w / beta / 32.).floor().max(1.) * 32.;
    } else if hh * ww < 65_536. {
        let beta = (65_536. / (h * w)).sqrt();
        hh = (h * beta / 32.).ceil() * 32.;
        ww = (w * beta / 32.).ceil() * 32.;
    }
    ensure!(
        hh >= 32. && ww >= 32. && hh * ww <= MAX_PIXELS as f64,
        "resized image exceeds pixel limit"
    );
    Ok((hh as u32, ww as u32))
}

pub fn file_data_url(path: &std::path::Path) -> Result<String> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "image path must be a regular file"
    );
    let mut bytes = Vec::new();
    file.take((MAX_IMAGE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= MAX_IMAGE_BYTES, "image exceeds 20 MiB");
    let mime = match image::guess_format(&bytes)? {
        image::ImageFormat::Png => "png",
        image::ImageFormat::Jpeg => "jpeg",
        image::ImageFormat::WebP => "webp",
        _ => bail!("unsupported image format"),
    };
    Ok(format!(
        "data:image/{mime};base64,{}",
        STANDARD.encode(bytes)
    ))
}

pub fn decode_image(url: &str) -> Result<PreparedImage> {
    decode_image_with_budget(url, MAX_PIXELS, MAX_IMAGE_TOKENS, None)
}

/// `decode_image`, the picture first bounded at `cap` pixels (bicubic), as d1 serves pictures.
pub fn decode_image_capped(url: &str, cap: u64) -> Result<PreparedImage> {
    decode_image_with_budget(url, MAX_PIXELS, MAX_IMAGE_TOKENS, Some(cap))
}

/// The size a picture of `width` x `height` takes once bounded at `cap` pixels,
/// as d1's `cap_pixels` computes it: unchanged within the cap, else both sides scaled.
pub fn capped_dimensions(width: u32, height: u32, cap: u64) -> (u32, u32) {
    let pixels = u64::from(width) * u64::from(height);
    if cap == 0 || pixels <= cap {
        return (width, height);
    }
    let scale = (cap as f64 / pixels as f64).sqrt();
    let side = |n: u32| ((f64::from(n) * scale) as u32).max(1);
    (side(width), side(height))
}

/// The bytes of a PNG, JPEG or WebP base64 data URL, at most 20 MiB.
fn data_url_bytes(url: &str) -> Result<Vec<u8>> {
    let (header, encoded) = url.split_once(",").context("image requires a data URL")?;
    ensure!(
        matches!(
            header,
            "data:image/png;base64" | "data:image/jpeg;base64" | "data:image/webp;base64"
        ),
        "supported image types are PNG, JPEG and WebP data URLs"
    );
    ensure!(
        encoded.len() <= MAX_IMAGE_BYTES.div_ceil(3) * 4,
        "image exceeds 20 MiB"
    );
    let bytes = STANDARD.decode(encoded).context("invalid base64 image")?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_IMAGE_BYTES,
        "invalid image byte count"
    );
    Ok(bytes)
}

/// A reader of `bytes` bounded at 32,768 pixels a side and 256 MiB of decoder allocations.
fn image_reader(bytes: Vec<u8>) -> Result<ImageReader<Cursor<Vec<u8>>>> {
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(32768);
    limits.max_image_height = Some(32768);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    Ok(reader)
}

fn decode_image_with_budget(
    url: &str,
    remaining_pixels: u64,
    remaining_tokens: u64,
    cap: Option<u64>,
) -> Result<PreparedImage> {
    let reader = image_reader(data_url_bytes(url)?)?;
    let mut decoder = reader.into_decoder().context("opening image decoder")?;
    let orientation = decoder.orientation().context("reading image orientation")?;
    let (mut width, mut height) = decoder.dimensions();
    if orientation.to_exif() >= 5 {
        std::mem::swap(&mut width, &mut height);
    }
    let capped = cap.map(|cap| capped_dimensions(width, height, cap));
    let (height, width) = match capped {
        Some((w, h)) => resized_shape(h, w)?,
        None => resized_shape(height, width)?,
    };
    ensure!(
        fits_image_budget(height, width, remaining_pixels, remaining_tokens),
        "conversation images exceed the aggregate pixel or token limit"
    );
    // Admission precedes decoding, RGB conversion, resampling and patch storage.
    let mut image = image::DynamicImage::from_decoder(decoder).context("decoding image")?;
    image.apply_orientation(orientation);
    let mut image = image.to_rgb8();
    if let Some((w, h)) = capped
        && image.dimensions() != (w, h)
    {
        image = image::imageops::resize(&image, w, h, FilterType::CatmullRom);
    }
    // The native image decoder uses antialiased Catmull-Rom bicubic resampling.
    let image = if image.dimensions() == (width, height) {
        image
    } else {
        image::imageops::resize(&image, width, height, FilterType::CatmullRom)
    };
    let gh = height / 16;
    let gw = width / 16;
    let count = usize::try_from(u64::from(gh) * u64::from(gw) * 1536)?;
    #[cfg(test)]
    PATCH_ALLOCATIONS.with(|n| n.set(n.get() + 1));
    let mut patches = Vec::new();
    patches
        .try_reserve_exact(count)
        .context("allocating image patches")?;
    patches.resize(count, 0.);
    for row in 0..gh {
        for col in 0..gw {
            let index =
                patch_index(row, col, gh, gw, 2).context("invalid image patch order")? as usize;
            let mut offset = index * 1536;
            for channel in 0..3 {
                for _temporal in 0..2 {
                    for y in 0..16 {
                        for x in 0..16 {
                            patches[offset] =
                                f32::from(image.get_pixel(col * 16 + x, row * 16 + y)[channel])
                                    / 127.5
                                    - 1.;
                            offset += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(PreparedImage {
        height: gh,
        width: gw,
        patches,
    })
}

/// Image input layout of the loaded model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    /// LFM3-VL: resized to multiples of 32 pixels, 16-pixel patches, text placeholders expanded before
    /// the chat template.
    Lfm3,
    /// Inkling 2 Mini: native resolution in 40-pixel tokens, one placeholder per image part written by
    /// the chat template and expanded afterwards (`expand_image_placeholders`).
    Inkling2,
}

/// Largest picture Inkling 2 reads at full size (`MLXL3_INKLING_IMAGE_MAX_PIXELS` overrides it): a larger
/// one is scaled down, aspect kept (1920 x 1080 is 1,296 image tokens).
pub const INKLING_MAX_PIXELS: u64 = 1920 * 1080;
/// Longest side Inkling 2 reads (the source model card's range is 40 to 4,096 pixels a side).
pub const INKLING_MAX_SIDE: u32 = 4096;

/// The Inkling picture cap: `MLXL3_INKLING_IMAGE_MAX_PIXELS` (one token's 1,600 pixels up to the
/// aggregate limit) or `INKLING_MAX_PIXELS`.
pub fn inkling_max_pixels() -> Result<u64> {
    match std::env::var("MLXL3_INKLING_IMAGE_MAX_PIXELS") {
        Ok(value) => {
            let pixels: u64 = value
                .trim()
                .parse()
                .context("MLXL3_INKLING_IMAGE_MAX_PIXELS must be a pixel count")?;
            ensure!(
                (1600..=MAX_PIXELS).contains(&pixels),
                "MLXL3_INKLING_IMAGE_MAX_PIXELS must be between 1600 and {MAX_PIXELS}"
            );
            Ok(pixels)
        }
        Err(_) => Ok(INKLING_MAX_PIXELS),
    }
}

/// Size an Inkling picture of `width` x `height` is read at: unchanged within `cap` pixels and
/// `INKLING_MAX_SIDE` a side, else both sides scaled down (at least one pixel each).
pub fn inkling_dimensions(width: u32, height: u32, cap: u64) -> Result<(u32, u32)> {
    ensure!(width > 0 && height > 0, "invalid image dimensions");
    let (mut w, mut h) = capped_dimensions(width, height, cap);
    let side = w.max(h);
    if side > INKLING_MAX_SIDE {
        let scale = f64::from(INKLING_MAX_SIDE) / f64::from(side);
        w = ((f64::from(w) * scale) as u32).clamp(1, INKLING_MAX_SIDE);
        h = ((f64::from(h) * scale) as u32).clamp(1, INKLING_MAX_SIDE);
    }
    Ok((w, h))
}

/// Token grid `(rows, columns)` of an Inkling picture: whole 40-pixel tiles, the last ones padded.
pub fn inkling_grid(width: u32, height: u32) -> (u32, u32) {
    let tile = crate::inkling2_vision::TOKEN_PIXELS;
    (height.div_ceil(tile), width.div_ceil(tile))
}

/// RGB pixels as the reference processor sees them: an image with an alpha channel is composited onto
/// its 8-pixel grey checkerboard (204 / 153), `(rgb * a + background * (255 - a) + 127) / 255`.
fn inkling_rgb(image: &image::DynamicImage) -> image::RgbImage {
    if !image.color().has_alpha() {
        return image.to_rgb8();
    }
    let rgba = image.to_rgba8();
    image::RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let pixel = rgba.get_pixel(x, y);
        let background: u32 = if (y / 8 + x / 8) % 2 == 0 { 204 } else { 153 };
        let alpha = u32::from(pixel[3]);
        image::Rgb([0, 1, 2].map(|c| {
            ((u32::from(pixel[c]) * alpha + background * (255 - alpha) + 127) / 255) as u8
        }))
    })
}

/// Encoder input of an RGB picture: `rows` x `columns` tokens of 2 x 2 patches of 20 x 20 pixels,
/// patches in the encoder's merged-block order, each patch's values by row, column, then channel,
/// `(value - 127.5) / 127.5`; pixels beyond the picture are the reference's padding (-1 on the 0-255
/// scale).
fn inkling_patches(rgb: &image::RgbImage, rows: u32, columns: u32) -> Result<Vec<f32>> {
    use crate::inkling2_vision::{MERGE, PATCH, PATCH_VALUES};
    let (height, width) = (rows * MERGE, columns * MERGE);
    let count = usize::try_from(u64::from(height) * u64::from(width))?
        .checked_mul(PATCH_VALUES)
        .context("image patch count overflow")?;
    #[cfg(test)]
    PATCH_ALLOCATIONS.with(|n| n.set(n.get() + 1));
    let mut patches = Vec::new();
    patches
        .try_reserve_exact(count)
        .context("allocating image patches")?;
    patches.resize(count, 0.);
    let pad = (-1f32 - 127.5) / 127.5;
    let (w, h) = rgb.dimensions();
    for row in 0..height {
        for column in 0..width {
            let index = crate::lfm3::patch_index(row, column, height, width, MERGE)
                .context("invalid image patch order")? as usize;
            let mut offset = index * PATCH_VALUES;
            for py in 0..PATCH {
                for px in 0..PATCH {
                    let (x, y) = (column * PATCH + px, row * PATCH + py);
                    let pixel = (x < w && y < h).then(|| rgb.get_pixel(x, y));
                    for channel in 0..3 {
                        patches[offset] = match pixel {
                            Some(p) => (f32::from(p[channel]) - 127.5) / 127.5,
                            None => pad,
                        };
                        offset += 1;
                    }
                }
            }
        }
    }
    Ok(patches)
}

/// Decodes an Inkling 2 picture within the remaining aggregate budget (padded pixels and tokens),
/// admitted from its header before any pixel is decoded.
fn decode_inkling_image(
    url: &str,
    remaining_pixels: u64,
    remaining_tokens: u64,
) -> Result<PreparedImage> {
    let reader = image_reader(data_url_bytes(url)?)?;
    let mut decoder = reader.into_decoder().context("opening image decoder")?;
    let orientation = decoder.orientation().context("reading image orientation")?;
    let (mut width, mut height) = decoder.dimensions();
    if orientation.to_exif() >= 5 {
        std::mem::swap(&mut width, &mut height);
    }
    let (w, h) = inkling_dimensions(width, height, inkling_max_pixels()?)?;
    let (rows, columns) = inkling_grid(w, h);
    let tokens = u64::from(rows) * u64::from(columns);
    let tile = u64::from(crate::inkling2_vision::TOKEN_PIXELS);
    ensure!(
        tokens * tile * tile <= remaining_pixels && tokens <= remaining_tokens,
        "conversation images exceed the aggregate pixel or token limit"
    );
    let mut image = image::DynamicImage::from_decoder(decoder).context("decoding image")?;
    image.apply_orientation(orientation);
    let rgb = inkling_rgb(&image);
    let rgb = if rgb.dimensions() == (w, h) {
        rgb
    } else {
        image::imageops::resize(&rgb, w, h, FilterType::CatmullRom)
    };
    Ok(PreparedImage {
        height: rows * crate::inkling2_vision::MERGE,
        width: columns * crate::inkling2_vision::MERGE,
        patches: inkling_patches(&rgb, rows, columns)?,
    })
}

/// Pixels an admitted image takes from the aggregate budget.
fn budget_pixels(format: ImageFormat, image: &PreparedImage) -> u64 {
    let patch = match format {
        ImageFormat::Lfm3 => 16u64,
        ImageFormat::Inkling2 => u64::from(crate::inkling2_vision::PATCH),
    };
    u64::from(image.height) * u64::from(image.width) * patch * patch
}

/// Inkling 2: the chat template writes one `<|unused_200054|>` per image part; the reference processor
/// then repeats it once per image token. Every placeholder of `rendered` must belong to an image, in
/// order. Other formats (and text-only prompts) are returned unchanged.
pub fn expand_image_placeholders(
    format: Option<ImageFormat>,
    rendered: String,
    images: &[PreparedImage],
) -> Result<String> {
    use crate::inkling2_vision::IMAGE_PLACEHOLDER;
    if format != Some(ImageFormat::Inkling2) || images.is_empty() {
        return Ok(rendered);
    }
    let pieces: Vec<&str> = rendered.split(IMAGE_PLACEHOLDER).collect();
    ensure!(
        pieces.len() == images.len() + 1,
        "the chat template wrote {} image placeholders for {} images",
        pieces.len() - 1,
        images.len()
    );
    let extra: usize = images.iter().map(PreparedImage::token_count).sum();
    let mut out = String::with_capacity(rendered.len() + extra * IMAGE_PLACEHOLDER.len());
    for (index, piece) in pieces.iter().enumerate() {
        out.push_str(piece);
        if let Some(image) = images.get(index) {
            out.push_str(&IMAGE_PLACEHOLDER.repeat(image.token_count()));
        }
    }
    Ok(out)
}

/// Expand each image at its original position before applying the chat template.
/// Remote URLs and local paths are deliberately not fetched by the HTTP server.
pub fn prepare_messages(
    messages: &[Value],
    supports_images: bool,
) -> Result<(Vec<Value>, Vec<PreparedImage>)> {
    prepare_messages_as(messages, supports_images.then_some(ImageFormat::Lfm3))
}

/// `prepare_messages` for the image layout of the loaded model (`None`: text only). LFM3 messages become
/// text with their expanded placeholders; an Inkling 2 message with images keeps its parts (text parts,
/// then `{"type": "image"}` for each picture) for the chat template. Messages without images are
/// flattened to text in both cases.
pub fn prepare_messages_as(
    messages: &[Value],
    format: Option<ImageFormat>,
) -> Result<(Vec<Value>, Vec<PreparedImage>)> {
    let mut output = Vec::new();
    let mut images = Vec::new();
    let mut total_pixels = 0u64;
    let mut total_tokens = 0u64;
    for original in messages {
        validate_content(original)?;
        let mut message = original.clone();
        if let Some(urls) = original.get("images").and_then(Value::as_array)
            && !urls.is_empty()
        {
            let mut parts = vec![json!({"type":"text","text":original["content"]})];
            parts.extend(
                urls.iter()
                    .map(|url| json!({"type":"image_url","image_url":{"url":url}})),
            );
            message["content"] = json!(parts);
        }
        let mut normalized = message.clone();
        normalized
            .as_object_mut()
            .context("message must be an object")?
            .remove("images");
        let content = match &message["content"] {
            Value::String(text) => json!(text),
            Value::Array(parts) => {
                let mut text = String::new();
                // Inkling 2: the parts of a message with pictures, kept for the chat template
                let mut kept = Vec::new();
                let mut pictures = false;
                for part in parts {
                    if part["type"] == "text" {
                        let piece = part["text"].as_str().context("missing text")?;
                        text.push_str(piece);
                        kept.push(json!({"type":"text","text":piece}));
                    } else {
                        let kind = format.context(
                            "this model does not support images; select a vision model \
                             (LFM3-VL, or Inkling 2 Mini with its vision encoder)",
                        )?;
                        ensure!(
                            images.len() < MAX_IMAGES,
                            "at most eight images per conversation are supported"
                        );
                        let url = part["image_url"]["url"]
                            .as_str()
                            .context("missing image URL")?;
                        let (pixels, tokens) =
                            (MAX_PIXELS - total_pixels, MAX_IMAGE_TOKENS - total_tokens);
                        let image = match kind {
                            ImageFormat::Lfm3 => {
                                decode_image_with_budget(url, pixels, tokens, None)?
                            }
                            ImageFormat::Inkling2 => decode_inkling_image(url, pixels, tokens)?,
                        };
                        total_pixels += budget_pixels(kind, &image);
                        total_tokens += image.token_count() as u64;
                        if kind == ImageFormat::Lfm3 {
                            text.push_str("<|vision_start|>");
                            text.push_str(&"<|image|>".repeat(image.token_count()));
                            text.push_str("<|vision_end|>");
                        }
                        kept.push(json!({"type":"image"}));
                        pictures = true;
                        images.push(image);
                    }
                }
                if pictures && format == Some(ImageFormat::Inkling2) {
                    json!(kept)
                } else {
                    json!(text)
                }
            }
            _ => unreachable!("validated above"),
        };
        normalized["content"] = content;
        output.push(normalized);
    }
    Ok((output, images))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat, RgbImage};
    fn data_url() -> String {
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::from_fn(256, 256, |x, y| {
            image::Rgb([x as u8, y as u8, 73])
        }))
        .write_to(&mut bytes, ImageFormat::Png)
        .unwrap();
        format!(
            "data:image/png;base64,{}",
            STANDARD.encode(bytes.into_inner())
        )
    }
    #[test]
    fn highly_compressed_limit_image_is_rejected_before_materialization() {
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageLuma8(image::GrayImage::new(4096, 4096))
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        assert!(bytes.get_ref().len() < MAX_IMAGE_BYTES);
        let url = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(bytes.into_inner())
        );
        PATCH_ALLOCATIONS.with(|n| n.set(0));
        // A previous admitted image reserves its pixels before this decode.
        for remaining in [0, MAX_PIXELS - 1] {
            assert!(
                decode_image_with_budget(&url, remaining, MAX_IMAGE_TOKENS, None)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("aggregate")
            );
            PATCH_ALLOCATIONS.with(|n| assert_eq!(n.get(), 0));
        }
        assert!(fits_image_budget(4096, 4096, MAX_PIXELS, MAX_IMAGE_TOKENS));
        assert!(!fits_image_budget(4096, 4096, 0, 0));
    }

    #[test]
    fn remaining_budget_rejects_before_patch_allocation() {
        let url = data_url();
        PATCH_ALLOCATIONS.with(|n| n.set(0));
        for (pixels, tokens) in [(65_535, 64), (65_536, 63), (0, 0)] {
            assert!(
                decode_image_with_budget(&url, pixels, tokens, None)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("aggregate")
            );
            PATCH_ALLOCATIONS.with(|n| assert_eq!(n.get(), 0));
        }
        let image = decode_image_with_budget(&url, 65_536, 64, None).unwrap();
        assert_eq!(image.token_count(), 64);
        PATCH_ALLOCATIONS.with(|n| assert_eq!(n.get(), 1));
        // The next image sees the remaining budget before allocating patches.
        assert!(decode_image_with_budget(&url, 0, 0, None).is_err());
        PATCH_ALLOCATIONS.with(|n| assert_eq!(n.get(), 1));
        let input = json!({"role":"user","content":"eight", "images": vec![url; 8]});
        let (_, images) = prepare_messages(&[input], true).unwrap();
        assert_eq!(images.len(), 8);
        assert_eq!(
            images.iter().map(PreparedImage::token_count).sum::<usize>(),
            512
        );
    }

    #[test]
    fn patches_preserve_channels_temporal_repeats_and_merge_order() {
        let image = decode_image(&data_url()).unwrap();
        assert_eq!(
            (
                image.height,
                image.width,
                image.token_count(),
                image.patches.len()
            ),
            (16, 16, 64, 256 * 1536)
        );
        for (patch, row, col) in [
            (0, 0, 0),
            (1, 0, 1),
            (2, 1, 0),
            (3, 1, 1),
            (4, 0, 2),
            (255, 15, 15),
        ] {
            for channel in 0..3 {
                for temporal in 0..2 {
                    for y in 0..16 {
                        for x in 0..16 {
                            let actual = image.patches
                                [patch * 1536 + channel * 512 + temporal * 256 + y * 16 + x];
                            let value = match channel {
                                0 => col * 16 + x,
                                1 => row * 16 + y,
                                _ => 73,
                            };
                            assert!((actual - (value as f32 / 127.5 - 1.)).abs() < 1e-7);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn jpeg_exif_orientation_is_applied_before_resizing() {
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 64, image::Rgb([200, 40, 10])))
            .write_to(&mut encoded, ImageFormat::Jpeg)
            .unwrap();
        let mut jpeg = encoded.into_inner();
        let exif = [
            b'E', b'x', b'i', b'f', 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 18, 1, 3, 0, 1, 0,
            0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ];
        let mut app1 = vec![255, 225, 0, (exif.len() + 2) as u8];
        app1.extend(exif);
        jpeg.splice(2..2, app1);
        let url = format!("data:image/jpeg;base64,{}", STANDARD.encode(jpeg));
        let image = decode_image(&url).unwrap();
        assert_eq!((image.height, image.width), (12, 24));
    }

    #[test]
    fn resize_boundaries_and_bad_inputs() {
        assert_eq!(resized_shape(256, 256).unwrap(), (256, 256));
        assert_eq!(resized_shape(1, 1).unwrap(), (256, 256));
        assert_eq!(resized_shape(272, 272).unwrap(), (256, 256));
        assert!(resized_shape(0, 10).is_err());
        assert!(resized_shape(1, 201).is_err());
        for h in [1, 31, 32, 255, 256, 272, 512, 8192, 32768] {
            for w in [1, 32, 128, 256, 272, 2048, 32768] {
                if let Ok((rh, rw)) = resized_shape(h, w) {
                    assert!(
                        rh > 0
                            && rw > 0
                            && rh % 32 == 0
                            && rw % 32 == 0
                            && u64::from(rh) * u64::from(rw) <= MAX_PIXELS
                    );
                }
            }
        }
        for bad in [
            "https://example.invalid/image.png",
            "data:image/png;base64,!",
            "data:image/png;base64,",
            "data:image/gif;base64,AAAA",
            "data:image/jpeg;base64,YmFk",
        ] {
            assert!(decode_image(bad).is_err());
        }
    }
    fn png_url(image: DynamicImage) -> String {
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, ImageFormat::Png).unwrap();
        format!(
            "data:image/png;base64,{}",
            STANDARD.encode(bytes.into_inner())
        )
    }

    fn inkling_image(url: &str) -> PreparedImage {
        let message =
            json!({"role":"user","content":[{"type":"image_url","image_url":{"url":url}}]});
        let (_, mut images) =
            prepare_messages_as(&[message], Some(super::ImageFormat::Inkling2)).unwrap();
        assert_eq!(images.len(), 1);
        images.remove(0)
    }

    /// Oracle: the reference processor and encoder reshapes written out directly. The picture padded to
    /// whole 40-pixel tiles with -1 (0-255 scale), normalized `(x - 127.5) / 127.5`, reshaped
    /// `(C, rows, 40, cols, 40) -> permute(1, 3, 2, 4, 0)` into one `[40, 40, C]` tile per token, then the
    /// encoder's `reshape(-1, 2, 20, 2, 20, C).transpose(2, 3)` into `[token, my, mx, py, px, C]`.
    fn reference_values(
        pixel: impl Fn(u32, u32) -> Option<[u8; 3]>,
        rows: u32,
        cols: u32,
    ) -> Vec<f32> {
        let mut out = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                for my in 0..2 {
                    for mx in 0..2 {
                        for py in 0..20 {
                            for px in 0..20 {
                                let (y, x) = (r * 40 + my * 20 + py, c * 40 + mx * 20 + px);
                                for ch in 0..3 {
                                    let raw = pixel(x, y).map_or(-1., |p| f32::from(p[ch]));
                                    out.push((raw - 127.5) / 127.5);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn inkling_patches_follow_the_reference_tiles_padding_and_order() {
        let (w, h) = (90, 50);
        let rgb = RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, (y * 3) as u8, 7]));
        let image = inkling_image(&png_url(DynamicImage::ImageRgb8(rgb)));
        // ceil(50 / 40) = 2 rows, ceil(90 / 40) = 3 columns of tokens, 2 x 2 patches each
        assert_eq!((image.height, image.width, image.token_count()), (4, 6, 6));
        let expected = reference_values(
            |x, y| (x < w && y < h).then(|| [x as u8, (y * 3) as u8, 7]),
            2,
            3,
        );
        assert_eq!(image.patches.len(), expected.len());
        assert_eq!(image.patches, expected);
        // the padding is the reference's -1 on the 0-255 scale, not 0 or -1 after normalization
        assert_eq!(*image.patches.last().unwrap(), (-1. - 127.5) / 127.5);
    }

    #[test]
    fn inkling_alpha_is_composited_on_the_reference_checkerboard() {
        let rgba = image::RgbaImage::from_fn(40, 40, |x, y| {
            image::Rgba([250, 10, 120, [0, 128, 255][((x + y) % 3) as usize]])
        });
        let image = inkling_image(&png_url(DynamicImage::ImageRgba8(rgba)));
        let expected = reference_values(
            |x, y| {
                let alpha = [0u32, 128, 255][((x + y) % 3) as usize];
                let background = if (y / 8 + x / 8) % 2 == 0 { 204 } else { 153 };
                Some(
                    [250u32, 10, 120]
                        .map(|v| ((v * alpha + background * (255 - alpha) + 127) / 255) as u8),
                )
            },
            1,
            1,
        );
        assert_eq!(image.patches, expected);
    }

    #[test]
    fn inkling_pictures_keep_native_size_within_the_caps() {
        let cap = super::INKLING_MAX_PIXELS;
        assert_eq!(
            super::inkling_dimensions(1920, 1080, cap).unwrap(),
            (1920, 1080)
        );
        assert_eq!(super::inkling_dimensions(1, 1, cap).unwrap(), (1, 1));
        let (w, h) = super::inkling_dimensions(4032, 3024, cap).unwrap();
        assert!(
            u64::from(w) * u64::from(h) <= cap && w > 1600 && h > 1200,
            "{w} x {h}"
        );
        // one side beyond the model card's 4,096 pixels is brought back to it, aspect kept
        assert_eq!(
            super::inkling_dimensions(8000, 100, MAX_PIXELS).unwrap(),
            (4096, 51)
        );
        assert!(super::inkling_dimensions(0, 10, cap).is_err());
        assert_eq!(super::inkling_grid(40, 40), (1, 1));
        assert_eq!(super::inkling_grid(41, 1), (1, 2));
        assert_eq!(super::inkling_grid(1920, 1080), (27, 48));
        // a picture over the cap is scaled before its patches are built
        let (w, h) = super::inkling_dimensions(2200, 1000, cap).unwrap();
        assert!(u64::from(w) * u64::from(h) <= cap && w < 2200 && h < 1000);
        let big = inkling_image(&png_url(DynamicImage::ImageRgb8(RgbImage::new(2200, 1000))));
        assert_eq!((big.height / 2, big.width / 2), super::inkling_grid(w, h));
    }

    #[test]
    fn inkling_messages_keep_parts_for_the_template() {
        use super::ImageFormat::Inkling2;
        let url = png_url(DynamicImage::ImageRgb8(RgbImage::new(80, 40)));
        let input = json!({"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":url}},{"type":"text","text":"after"}]});
        let (messages, images) =
            prepare_messages_as(std::slice::from_ref(&input), Some(Inkling2)).unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].token_count(), 2);
        assert_eq!(
            messages[0]["content"],
            json!([{"type":"text","text":"before"},{"type":"image"},{"type":"text","text":"after"}])
        );
        // the legacy `images` field becomes the text then the pictures
        let legacy = json!({"role":"user","content":"look","images":[url.clone(), url.clone()]});
        let (messages, images) = prepare_messages_as(&[legacy], Some(Inkling2)).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(
            messages[0]["content"],
            json!([{"type":"text","text":"look"},{"type":"image"},{"type":"image"}])
        );
        assert!(messages[0].get("images").is_none());
        // text-only messages are flattened exactly as before
        let text = json!({"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]});
        let (messages, images) =
            prepare_messages_as(std::slice::from_ref(&text), Some(Inkling2)).unwrap();
        assert!(images.is_empty());
        assert_eq!(messages, prepare_messages(&[text], false).unwrap().0);
        assert_eq!(messages[0]["content"], json!("ab"));
        assert!(prepare_messages_as(std::slice::from_ref(&input), None).is_err());
        let nine = json!({"role":"user","content":"x","images":vec![url; 9]});
        assert!(prepare_messages_as(&[nine], Some(Inkling2)).is_err());
    }

    #[test]
    fn inkling_admission_happens_before_patch_allocation() {
        let url = png_url(DynamicImage::ImageRgb8(RgbImage::new(400, 400)));
        PATCH_ALLOCATIONS.with(|n| n.set(0));
        // 100 tokens of 1,600 pixels: one pixel or one token short of the budget is refused
        for (pixels, tokens) in [(159_999, 100), (160_000, 99), (0, 0)] {
            let error = decode_inkling_image(&url, pixels, tokens)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("aggregate"), "{error}");
            PATCH_ALLOCATIONS.with(|n| assert_eq!(n.get(), 0));
        }
        let image = decode_inkling_image(&url, 160_000, 100).unwrap();
        assert_eq!(image.token_count(), 100);
        assert_eq!(
            super::budget_pixels(super::ImageFormat::Inkling2, &image),
            160_000
        );
        PATCH_ALLOCATIONS.with(|n| assert_eq!(n.get(), 1));
    }

    #[test]
    fn inkling_placeholders_expand_once_per_image_token() {
        use super::ImageFormat::{Inkling2, Lfm3};
        let p = crate::inkling2_vision::IMAGE_PLACEHOLDER;
        let image = |rows: u32, cols: u32| PreparedImage {
            height: rows * 2,
            width: cols * 2,
            patches: Vec::new(),
        };
        let images = [image(1, 2), image(3, 1)];
        let rendered =
            format!("a<|content_image|>{p}<|end_message|>b<|content_image|>{p}<|end_message|>");
        let expanded =
            expand_image_placeholders(Some(Inkling2), rendered.clone(), &images).unwrap();
        assert_eq!(
            expanded,
            format!(
                "a<|content_image|>{}<|end_message|>b<|content_image|>{}<|end_message|>",
                p.repeat(2),
                p.repeat(3)
            )
        );
        // a placeholder typed in the text, or a template that dropped one, is refused
        assert!(
            expand_image_placeholders(Some(Inkling2), format!("{rendered}{p}"), &images).is_err()
        );
        assert!(expand_image_placeholders(Some(Inkling2), rendered.clone(), &images[..1]).is_err());
        // other formats and text-only prompts are unchanged
        for (format, images) in [
            (Some(Lfm3), &images[..]),
            (None, &images[..]),
            (Some(Inkling2), &[][..]),
        ] {
            assert_eq!(
                expand_image_placeholders(format, rendered.clone(), images).unwrap(),
                rendered
            );
        }
    }

    #[test]
    fn messages_preserve_image_position_and_reject_unsupported_models() {
        let input = json!({"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":data_url()}},{"type":"text","text":"after"}]});
        assert!(prepare_messages(std::slice::from_ref(&input), false).is_err());
        let (messages, images) = prepare_messages(&[input], true).unwrap();
        assert_eq!(images.len(), 1);
        let text = messages[0]["content"].as_str().unwrap();
        assert!(
            text.starts_with("before<|vision_start|>") && text.ends_with("<|vision_end|>after")
        );
        assert_eq!(text.matches("<|image|>").count(), 64);
        for bad in [
            json!({"role":"user","content":[]}),
            json!({"role":"user","content":[{"type":"audio"}]}),
            json!({"role":"assistant","content":[{"type":"image_url","image_url":{"url":data_url()}}]}),
        ] {
            assert!(prepare_messages(&[bad], true).is_err());
        }
    }
}
