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
    decode_image_with_budget(url, MAX_PIXELS, MAX_IMAGE_TOKENS)
}

fn decode_image_with_budget(
    url: &str,
    remaining_pixels: u64,
    remaining_tokens: u64,
) -> Result<PreparedImage> {
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
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(32768);
    limits.max_image_height = Some(32768);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().context("opening image decoder")?;
    let orientation = decoder.orientation().context("reading image orientation")?;
    let (mut width, mut height) = decoder.dimensions();
    if orientation.to_exif() >= 5 {
        std::mem::swap(&mut width, &mut height);
    }
    let (height, width) = resized_shape(height, width)?;
    ensure!(
        fits_image_budget(height, width, remaining_pixels, remaining_tokens),
        "conversation images exceed the aggregate pixel or token limit"
    );
    // Admission precedes decoding, RGB conversion, resampling and patch storage.
    let mut image = image::DynamicImage::from_decoder(decoder).context("decoding image")?;
    image.apply_orientation(orientation);
    let image = image.to_rgb8();
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

/// Expand each image at its original position before applying the chat template.
/// Remote URLs and local paths are deliberately not fetched by the HTTP server.
pub fn prepare_messages(
    messages: &[Value],
    supports_images: bool,
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
        let text = match &message["content"] {
            Value::String(text) => text.clone(),
            Value::Array(parts) => {
                let mut text = String::new();
                for part in parts {
                    if part["type"] == "text" {
                        text.push_str(part["text"].as_str().context("missing text")?);
                    } else {
                        ensure!(
                            supports_images,
                            "this model does not support images; select LFM3-VL"
                        );
                        ensure!(
                            images.len() < MAX_IMAGES,
                            "at most eight images per conversation are supported"
                        );
                        let image = decode_image_with_budget(
                            part["image_url"]["url"]
                                .as_str()
                                .context("missing image URL")?,
                            MAX_PIXELS - total_pixels,
                            MAX_IMAGE_TOKENS - total_tokens,
                        )?;
                        total_pixels += u64::from(image.height) * u64::from(image.width) * 256;
                        total_tokens += image.token_count() as u64;
                        text.push_str("<|vision_start|>");
                        text.push_str(&"<|image|>".repeat(image.token_count()));
                        text.push_str("<|vision_end|>");
                        images.push(image);
                    }
                }
                text
            }
            _ => unreachable!("validated above"),
        };
        normalized["content"] = json!(text);
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
                decode_image_with_budget(&url, remaining, MAX_IMAGE_TOKENS)
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
                decode_image_with_budget(&url, pixels, tokens)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("aggregate")
            );
            PATCH_ALLOCATIONS.with(|n| assert_eq!(n.get(), 0));
        }
        let image = decode_image_with_budget(&url, 65_536, 64).unwrap();
        assert_eq!(image.token_count(), 64);
        PATCH_ALLOCATIONS.with(|n| assert_eq!(n.get(), 1));
        // The next image sees the remaining budget before allocating patches.
        assert!(decode_image_with_budget(&url, 0, 0).is_err());
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
