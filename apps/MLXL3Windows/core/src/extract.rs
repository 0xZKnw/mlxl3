//! Text extraction for chat attachments and the project document library.
//!
//! Mirrors the Mac `ChatAttachment` rules: regular files only, at most 20 MiB on disk,
//! text decoded strictly as UTF-8 (or UTF-16 with a byte-order mark), PDF text per page
//! with "[Page N]" markers. Scanned (textless) and unreadable PDFs are refused.

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::fs;
use std::path::Path;

pub const MAX_FILE_BYTES: u64 = 20 * 1024 * 1024;

/// Extensions read as text, the Mac `textExtensions` list.
pub const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml", "toml", "log",
    "ini", "cfg", "conf", "rs", "py", "swift", "js", "ts", "tsx", "jsx", "c", "h", "cpp", "hpp",
    "css", "html", "sh", "zsh", "sql", "go", "java", "kt", "rb", "tex", "ipynb", "bend", "vue",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Extracted {
    pub file_name: String,
    pub path: String,
    pub text: String,
    /// Size of the file on disk.
    pub file_bytes: u64,
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
}

pub fn is_supported(path: &Path) -> bool {
    extension(path).is_some_and(|e| e == "pdf" || TEXT_EXTENSIONS.contains(&e.as_str()))
}

/// Strict text decoding: UTF-8 (BOM optional) or UTF-16 LE/BE with a BOM. NUL bytes reject.
pub fn decode_text(bytes: &[u8]) -> Result<String> {
    let text = if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        String::from_utf8(rest.to_vec()).ok()
    } else if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        utf16(rest, u16::from_le_bytes)
    } else if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        utf16(rest, u16::from_be_bytes)
    } else {
        String::from_utf8(bytes.to_vec()).ok()
    };
    let text = text.context("the file is not valid UTF-8 or UTF-16 text")?;
    ensure!(
        !text.contains('\0'),
        "the file is not valid UTF-8 or UTF-16 text"
    );
    Ok(text)
}

fn utf16(bytes: &[u8], word: fn([u8; 2]) -> u16) -> Option<String> {
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let units: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|c| word(*c)).collect();
    String::from_utf16(&units).ok()
}

/// PDF text with a "[Page N]" line before each page that has text.
pub fn pdf_text(bytes: &[u8]) -> Result<String> {
    // pdf-extract panics on some malformed files; a bad attachment must not take the app down.
    let pages = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem_by_pages(bytes))
        .map_err(|_| anyhow::anyhow!("the PDF is unreadable or damaged"))?
        .map_err(|error| {
            let text = error.to_string();
            if text.to_ascii_lowercase().contains("encrypt") {
                anyhow::anyhow!("the PDF is locked with a password")
            } else {
                anyhow::anyhow!("the PDF is unreadable or damaged")
            }
        })?;
    let mut out = String::new();
    for (index, page) in pages.iter().enumerate() {
        let page = page.trim();
        if page.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&format!("[Page {}]\n{page}", index + 1));
    }
    ensure!(
        !out.is_empty(),
        "the PDF has no extractable text (scanned pages are not supported)"
    );
    Ok(out)
}

/// Reads one file. `max_text_bytes` bounds the extracted text (UTF-8 bytes).
pub fn extract(path: &Path, max_text_bytes: usize) -> Result<Extracted> {
    let meta =
        fs::symlink_metadata(path).with_context(|| format!("cannot read {}", path.display()))?;
    ensure!(meta.is_file(), "{} is not a regular file", path.display());
    ensure!(meta.len() <= MAX_FILE_BYTES, "the file exceeds 20 MiB");
    let Some(kind) = extension(path).filter(|_| is_supported(path)) else {
        bail!("unsupported file type: choose a text, code, Markdown, CSV, JSON or PDF file");
    };
    let bytes = fs::read(path)?;
    let text = if kind == "pdf" {
        pdf_text(&bytes)?
    } else {
        decode_text(&bytes)?
    };
    ensure!(!text.trim().is_empty(), "the file is empty");
    ensure!(
        text.len() <= max_text_bytes,
        "the extracted text exceeds {} KiB",
        max_text_bytes / 1024
    );
    Ok(Extracted {
        file_name: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        path: path.display().to_string(),
        text,
        file_bytes: meta.len(),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small PDF with one text line per page, built with lopdf.
    pub fn pdf_with_pages(pages: &[&str]) -> Vec<u8> {
        use lopdf::content::{Content, Operation};
        use lopdf::{Document, Object, Stream, dictionary};
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
        });
        let resources_id =
            doc.add_object(dictionary! { "Font" => dictionary! { "F1" => font_id } });
        let mut kids = vec![];
        for text in pages {
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new("Tf", vec!["F1".into(), 24.into()]),
                    Operation::new("Td", vec![72.into(), 700.into()]),
                    Operation::new("Tj", vec![Object::string_literal(*text)]),
                    Operation::new("ET", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page", "Parent" => pages_id, "Contents" => content_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            });
            kids.push(page_id.into());
        }
        let count = kids.len() as i64;
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages", "Kids" => kids, "Count" => count, "Resources" => resources_id,
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut out = Vec::new();
        doc.save_to(&mut out).unwrap();
        out
    }

    #[test]
    fn text_decoding_is_strict() {
        assert_eq!(decode_text(b"caf\xc3\xa9").unwrap(), "café");
        assert_eq!(decode_text(b"\xef\xbb\xbfhi").unwrap(), "hi");
        assert_eq!(decode_text(&[0xFF, 0xFE, b'h', 0, b'i', 0]).unwrap(), "hi");
        assert_eq!(decode_text(&[0xFE, 0xFF, 0, b'h', 0, b'i']).unwrap(), "hi");
        assert!(decode_text(b"\xff\xfe\x00").is_err()); // odd UTF-16 length
        assert!(decode_text(b"bad \xc3").is_err());
        assert!(decode_text(b"a\0b").is_err());
    }

    #[test]
    fn files_are_checked_before_reading() {
        let dir = tempfile::tempdir().unwrap();
        let note = dir.path().join("Note.MD");
        fs::write(&note, "# Hello").unwrap();
        let got = extract(&note, 1024).unwrap();
        assert_eq!(
            (got.file_name.as_str(), got.text.as_str(), got.file_bytes),
            ("Note.MD", "# Hello", 7)
        );
        assert!(
            extract(&note, 4)
                .unwrap_err()
                .to_string()
                .contains("exceeds")
        );
        let blank = dir.path().join("blank.txt");
        fs::write(&blank, " \n").unwrap();
        assert!(
            extract(&blank, 1024)
                .unwrap_err()
                .to_string()
                .contains("empty")
        );
        let binary = dir.path().join("a.exe");
        fs::write(&binary, "MZ").unwrap();
        assert!(
            extract(&binary, 1024)
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );
        assert!(
            extract(dir.path(), 1024)
                .unwrap_err()
                .to_string()
                .contains("not a regular file")
        );
        let big = dir.path().join("big.txt");
        fs::File::create(&big)
            .unwrap()
            .set_len(MAX_FILE_BYTES + 1)
            .unwrap();
        assert!(
            extract(&big, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("20 MiB")
        );
    }

    #[test]
    fn pdf_pages_are_marked() {
        let bytes = pdf_with_pages(&["First page", "Second page"]);
        let text = pdf_text(&bytes).unwrap();
        assert!(text.starts_with("[Page 1]\n"), "{text:?}");
        assert!(
            text.contains("First page")
                && text.contains("[Page 2]\n")
                && text.contains("Second page"),
            "{text:?}"
        );
        assert!(pdf_text(b"%PDF-1.4 garbage").is_err());
        assert!(pdf_text(b"not a pdf").is_err());
    }
}

pub const MAX_IMAGE_BYTES: u64 = 8 * 1024 * 1024;
pub const MAX_IMAGE_PIXELS: u64 = 64_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ImageInfo {
    pub mime: &'static str,
    pub width: u32,
    pub height: u32,
}

fn be16(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from(u16::from_be_bytes(
        b.get(at..at + 2)?.try_into().ok()?,
    )))
}

/// Format and dimensions of a PNG, JPEG or WebP image, from its header only.
pub fn image_info(bytes: &[u8]) -> Option<ImageInfo> {
    let info = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") && bytes.get(12..16) == Some(b"IHDR") {
        let w = u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?);
        let h = u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?);
        ImageInfo {
            mime: "image/png",
            width: w,
            height: h,
        }
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        let mut at = 2;
        loop {
            // Skip fill bytes, then read the marker.
            while bytes.get(at) == Some(&0xFF) && bytes.get(at + 1) == Some(&0xFF) {
                at += 1;
            }
            if *bytes.get(at)? != 0xFF {
                return None;
            }
            let marker = *bytes.get(at + 1)?;
            if (0xD0..=0xD9).contains(&marker) || marker == 0x01 {
                at += 2;
                continue;
            }
            let length = be16(bytes, at + 2)? as usize;
            if length < 2 {
                return None;
            }
            if matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
                let h = be16(bytes, at + 5)?;
                let w = be16(bytes, at + 7)?;
                break ImageInfo {
                    mime: "image/jpeg",
                    width: w,
                    height: h,
                };
            }
            at += 2 + length;
        }
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        let le24 = |at: usize| -> Option<u32> {
            let b = bytes.get(at..at + 3)?;
            Some(u32::from(b[0]) | u32::from(b[1]) << 8 | u32::from(b[2]) << 16)
        };
        let le16 = |at: usize| -> Option<u32> {
            Some(u32::from(u16::from_le_bytes(
                bytes.get(at..at + 2)?.try_into().ok()?,
            )))
        };
        let (w, h) = match bytes.get(12..16)? {
            b"VP8 " => (le16(26)? & 0x3FFF, le16(28)? & 0x3FFF),
            b"VP8L" => {
                let b = bytes.get(21..25)?;
                let (b0, b1, b2, b3) = (
                    u32::from(b[0]),
                    u32::from(b[1]),
                    u32::from(b[2]),
                    u32::from(b[3]),
                );
                (
                    1 + (b0 | (b1 & 0x3F) << 8),
                    1 + (b1 >> 6 | b2 << 2 | (b3 & 0x0F) << 10),
                )
            }
            b"VP8X" => (1 + le24(24)?, 1 + le24(27)?),
            _ => return None,
        };
        ImageInfo {
            mime: "image/webp",
            width: w,
            height: h,
        }
    } else {
        return None;
    };
    (info.width > 0 && info.height > 0).then_some(info)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Image {
    pub file_name: String,
    pub path: String,
    pub info: ImageInfo,
    /// Base64 of the file bytes.
    pub data: String,
    pub file_bytes: u64,
}

pub fn is_image(path: &Path) -> bool {
    extension(path).is_some_and(|e| matches!(e.as_str(), "png" | "jpg" | "jpeg" | "webp"))
}

/// Reads one PNG/JPEG/WebP attachment (at most 8 MiB and 64 MP).
pub fn read_image(path: &Path) -> Result<Image> {
    use base64::Engine;
    let meta =
        fs::symlink_metadata(path).with_context(|| format!("cannot read {}", path.display()))?;
    ensure!(meta.is_file(), "{} is not a regular file", path.display());
    ensure!(meta.len() <= MAX_IMAGE_BYTES, "the image exceeds 8 MiB");
    let bytes = fs::read(path)?;
    let info = image_info(&bytes).context("unsupported image: choose a PNG, JPEG or WebP file")?;
    ensure!(
        u64::from(info.width) * u64::from(info.height) <= MAX_IMAGE_PIXELS,
        "the image exceeds 64 megapixels"
    );
    Ok(Image {
        file_name: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        path: path.display().to_string(),
        info,
        data: base64::engine::general_purpose::STANDARD.encode(&bytes),
        file_bytes: meta.len(),
    })
}

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn image_headers_give_format_and_size() {
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        png.extend(640u32.to_be_bytes());
        png.extend(480u32.to_be_bytes());
        assert_eq!(
            image_info(&png),
            Some(ImageInfo {
                mime: "image/png",
                width: 640,
                height: 480
            })
        );
        // JPEG: SOI, APP0 (length 4), SOF0 with height 300, width 200.
        let jpeg = [
            0xFF, 0xD8, 0xFF, 0xE0, 0, 4, 0, 0, 0xFF, 0xC0, 0, 11, 8, 0x01, 0x2C, 0x00, 0xC8, 3,
        ];
        assert_eq!(
            image_info(&jpeg),
            Some(ImageInfo {
                mime: "image/jpeg",
                width: 200,
                height: 300
            })
        );
        let mut webp = b"RIFF\0\0\0\0WEBPVP8X\0\0\0\0\0\0\0\0".to_vec();
        webp.extend([0x3F, 0x01, 0x00, 0xEF, 0x00, 0x00]); // 320 x 240
        assert_eq!(
            image_info(&webp),
            Some(ImageInfo {
                mime: "image/webp",
                width: 320,
                height: 240
            })
        );
        let mut lossy = b"RIFF\0\0\0\0WEBPVP8 \0\0\0\0\0\0\0\x9d\x01\x2a".to_vec();
        lossy.extend([0x80, 0x02, 0xE0, 0x41]); // 640 (scale bits ignored) x 480 with scale 1
        assert_eq!(
            image_info(&lossy).map(|i| (i.width, i.height)),
            Some((640, 480))
        );
        let mut lossless = b"RIFF\0\0\0\0WEBPVP8L\0\0\0\0\x2f".to_vec();
        // width-1 = 99, height-1 = 49 packed as 14-bit fields.
        let bits: u32 = 99 | 49 << 14;
        lossless.extend(bits.to_le_bytes());
        assert_eq!(
            image_info(&lossless).map(|i| (i.width, i.height)),
            Some((100, 50))
        );
        assert_eq!(image_info(b"GIF89a"), None);
        assert_eq!(image_info(&png[..20]), None);
        assert_eq!(image_info(&[0xFF, 0xD8, 0xFF, 0xC0, 0, 1]), None);
        let mut zero = png.clone();
        zero[16..20].copy_from_slice(&0u32.to_be_bytes());
        assert_eq!(image_info(&zero), None);
    }

    #[test]
    fn images_are_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        png.extend(2u32.to_be_bytes());
        png.extend(3u32.to_be_bytes());
        let path = dir.path().join("a.PNG");
        fs::write(&path, &png).unwrap();
        assert!(is_image(&path));
        let image = read_image(&path).unwrap();
        assert_eq!(
            (image.info.width, image.info.height, image.file_bytes),
            (2, 3, 24)
        );
        use base64::Engine;
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(&image.data)
                .unwrap(),
            png
        );
        let mut huge = png.clone();
        huge[16..20].copy_from_slice(&10_000u32.to_be_bytes());
        huge[20..24].copy_from_slice(&10_000u32.to_be_bytes());
        fs::write(&path, &huge).unwrap();
        assert!(
            read_image(&path)
                .unwrap_err()
                .to_string()
                .contains("megapixels")
        );
        fs::write(&path, b"text").unwrap();
        assert!(
            read_image(&path)
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );
    }
}
