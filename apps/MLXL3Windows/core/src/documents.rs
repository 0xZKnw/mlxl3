//! Project document library and passage search, ported from the Mac `DocumentSearch`.
//!
//! Documents live in `<dir>/index.json` (metadata) plus one `<id>.txt` per document.
//! Search is lexical: diacritic- and case-insensitive word overlap over passages of at
//! most 1200 characters; PDF "[Page N]" lines set the page cited in an excerpt.

use crate::extract;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

pub const MAX_DOCUMENTS: usize = 128;
pub const MAX_LIBRARY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_FOLDER_ENTRIES: usize = 4000;
pub const PASSAGE_CHARS: usize = 1200;
pub const MAX_RESULTS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub project_id: String,
    pub file_name: String,
    pub original_path: String,
    /// UTF-8 bytes of the extracted text.
    pub bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Excerpt {
    pub id: String,
    pub document_id: String,
    pub file_name: String,
    pub original_path: String,
    pub page: Option<u32>,
    pub text: String,
    pub label: String,
}

impl Excerpt {
    fn new(id: String, document: &Entry, page: Option<u32>, text: String) -> Self {
        let label = match page {
            Some(page) => format!("{} · p. {page}", document.file_name),
            None => document.file_name.clone(),
        };
        Self {
            id,
            document_id: document.id.clone(),
            file_name: document.file_name.clone(),
            original_path: document.original_path.clone(),
            page,
            text,
            label,
        }
    }
}

/// Lower-cased, diacritic-free words longer than one character.
pub fn words(text: &str) -> BTreeSet<String> {
    let folded: String = text
        .nfd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(char::to_lowercase)
        .collect();
    folded
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() > 1)
        .map(str::to_owned)
        .collect()
}

pub struct Passage {
    terms: BTreeSet<String>,
    excerpt: Excerpt,
}

fn page_marker(line: &str) -> Option<u32> {
    let number: u32 = line
        .strip_prefix("[Page ")?
        .strip_suffix(']')?
        .parse()
        .ok()?;
    (number > 0).then_some(number)
}

/// Splits documents into bounded passages, in document then passage order.
pub fn index(documents: &[(Entry, String)]) -> Vec<Passage> {
    let mut passages = Vec::new();
    for (document, text) in documents {
        let is_pdf = document.file_name.to_lowercase().ends_with(".pdf");
        let mut page = None;
        let mut passage = String::new();
        let mut count = 0usize;
        let mut ordinal = 0usize;
        let mut flush = |passage: &mut String, count: &mut usize, page: Option<u32>| {
            if !passage.trim().is_empty() {
                let terms = words(passage);
                if !terms.is_empty() {
                    let id = format!("{}-{ordinal}", document.id);
                    passages.push(Passage {
                        terms,
                        excerpt: Excerpt::new(id, document, page, passage.clone()),
                    });
                }
                ordinal += 1;
            }
            passage.clear();
            *count = 0;
        };
        for line in text.split('\n') {
            if is_pdf && let Some(number) = page_marker(line) {
                flush(&mut passage, &mut count, page);
                page = Some(number);
                continue;
            }
            if count >= PASSAGE_CHARS {
                flush(&mut passage, &mut count, page);
            }
            let mut rest = line;
            loop {
                let room = PASSAGE_CHARS.saturating_sub(count).max(1);
                let split = rest.char_indices().nth(room).map_or(rest.len(), |(i, _)| i);
                let (piece, tail) = rest.split_at(split);
                passage.push_str(piece);
                count += piece.chars().count();
                rest = tail;
                if count >= PASSAGE_CHARS {
                    flush(&mut passage, &mut count, page);
                }
                if rest.is_empty() {
                    break;
                }
            }
            if count < PASSAGE_CHARS {
                passage.push('\n');
                count += 1;
            } else {
                flush(&mut passage, &mut count, page);
            }
        }
        flush(&mut passage, &mut count, page);
    }
    passages
}

/// Best passages by shared-term count; ties keep index order. At most `min(limit, 8)`.
pub fn search(query: &str, index: &[Passage], limit: usize) -> Vec<Excerpt> {
    let terms = words(query);
    let capacity = limit.min(MAX_RESULTS);
    if terms.is_empty() || capacity == 0 {
        return vec![];
    }
    let mut best: Vec<(usize, &Excerpt)> = Vec::new();
    for passage in index {
        let score = passage.terms.intersection(&terms).count();
        if score == 0 {
            continue;
        }
        let position = best
            .iter()
            .position(|(s, _)| *s < score)
            .unwrap_or(best.len());
        if position < capacity {
            best.insert(position, (score, &passage.excerpt));
            best.truncate(capacity);
        }
    }
    best.into_iter().map(|(_, e)| e.clone()).collect()
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

#[derive(Default, Serialize)]
pub struct AddReport {
    pub added: Vec<Entry>,
    /// `(file name, reason)` for every file that was skipped.
    pub skipped: Vec<(String, String)>,
}

/// The on-disk library with an in-memory passage index per project.
pub struct Library {
    dir: PathBuf,
    entries: Vec<Entry>,
    cache: HashMap<String, Vec<Passage>>,
}

impl Library {
    pub fn open(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let index = dir.join("index.json");
        let entries: Vec<Entry> = match fs::read(&index) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("the document index is damaged")?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => vec![],
            Err(error) => return Err(error.into()),
        };
        // Drop entries whose id is unsafe or whose text went missing.
        let entries = entries
            .into_iter()
            .filter(|e| {
                valid_id(&e.id)
                    && valid_id(&e.project_id)
                    && dir.join(format!("{}.txt", e.id)).is_file()
            })
            .collect();
        Ok(Self {
            dir: dir.to_owned(),
            entries,
            cache: HashMap::new(),
        })
    }

    pub fn list(&self, project_id: &str) -> Vec<Entry> {
        self.entries
            .iter()
            .filter(|e| e.project_id == project_id)
            .cloned()
            .collect()
    }

    fn total_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    fn save(&self) -> Result<()> {
        let tmp = self.dir.join("index.json.tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(&self.entries)?)?;
        fs::rename(&tmp, self.dir.join("index.json"))?;
        Ok(())
    }

    fn new_id(&self, path: &Path) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut salt = 0u32;
        loop {
            let digest = Sha256::digest(format!(
                "{nanos}|{salt}|{}|{}",
                path.display(),
                self.entries.len()
            ));
            let hex: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
            let id = format!(
                "{}-{}-{}-{}-{}",
                &hex[..8],
                &hex[8..12],
                &hex[12..16],
                &hex[16..20],
                &hex[20..]
            );
            if !self.entries.iter().any(|e| e.id == id) {
                return id;
            }
            salt += 1;
        }
    }

    pub fn add_file(&mut self, project_id: &str, path: &Path) -> Result<Entry> {
        ensure!(valid_id(project_id), "invalid project");
        ensure!(
            self.entries.len() < MAX_DOCUMENTS,
            "the library is full ({MAX_DOCUMENTS} documents)"
        );
        let room = MAX_LIBRARY_BYTES.saturating_sub(self.total_bytes());
        ensure!(room > 0, "the library is full (16 MiB of text)");
        let extracted = extract::extract(path, room).map_err(|e| {
            if e.to_string().contains("exceeds") && !e.to_string().contains("20 MiB") {
                anyhow::anyhow!("not enough room left in the library (16 MiB of text)")
            } else {
                e
            }
        })?;
        let entry = Entry {
            id: self.new_id(path),
            project_id: project_id.to_owned(),
            file_name: extracted.file_name,
            original_path: extracted.path,
            bytes: extracted.text.len(),
        };
        fs::write(self.dir.join(format!("{}.txt", entry.id)), &extracted.text)?;
        self.entries.push(entry.clone());
        if let Err(error) = self.save() {
            self.entries.pop();
            let _ = fs::remove_file(self.dir.join(format!("{}.txt", entry.id)));
            return Err(error);
        }
        self.cache.remove(project_id);
        Ok(entry)
    }

    /// Adds files, walking folders for supported files (bounded, symlinks not followed).
    pub fn add_paths(&mut self, project_id: &str, paths: &[PathBuf]) -> AddReport {
        let mut report = AddReport::default();
        let mut files = Vec::new();
        let mut seen = 0usize;
        for path in paths {
            if path.is_dir() {
                walk(path, &mut files, &mut seen);
            } else {
                files.push(path.clone());
            }
        }
        if seen > MAX_FOLDER_ENTRIES {
            report.skipped.push((
                String::new(),
                format!("folder walk stopped after {MAX_FOLDER_ENTRIES} entries"),
            ));
        }
        for file in files {
            let name = file
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match self.add_file(project_id, &file) {
                Ok(entry) => report.added.push(entry),
                Err(error) => report.skipped.push((name, error.to_string())),
            }
        }
        report
    }

    pub fn remove(&mut self, id: &str) -> Result<()> {
        let Some(position) = self.entries.iter().position(|e| e.id == id) else {
            bail!("no such document");
        };
        let entry = self.entries.remove(position);
        self.save()?;
        let _ = fs::remove_file(self.dir.join(format!("{}.txt", entry.id)));
        self.cache.remove(&entry.project_id);
        Ok(())
    }

    pub fn remove_project(&mut self, project_id: &str) -> Result<usize> {
        let ids: Vec<String> = self.list(project_id).into_iter().map(|e| e.id).collect();
        for id in &ids {
            self.remove(id)?;
        }
        Ok(ids.len())
    }

    pub fn search(&mut self, project_id: &str, query: &str, limit: usize) -> Result<Vec<Excerpt>> {
        if words(query).is_empty() || limit == 0 {
            return Ok(vec![]);
        }
        if !self.cache.contains_key(project_id) {
            let mut documents = Vec::new();
            for entry in self.list(project_id) {
                let text = fs::read_to_string(self.dir.join(format!("{}.txt", entry.id)))?;
                documents.push((entry, text));
            }
            self.cache.insert(project_id.to_owned(), index(&documents));
        }
        Ok(search(query, &self.cache[project_id], limit))
    }
}

fn walk(dir: &Path, files: &mut Vec<PathBuf>, seen: &mut usize) {
    let Ok(read) = fs::read_dir(dir) else { return };
    let mut children: Vec<_> = read.flatten().map(|e| e.path()).collect();
    children.sort();
    for child in children {
        *seen += 1;
        if *seen > MAX_FOLDER_ENTRIES {
            return;
        }
        let Ok(meta) = fs::symlink_metadata(&child) else {
            continue;
        };
        let hidden = child
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'));
        if hidden {
            continue;
        }
        if meta.is_dir() {
            walk(&child, files, seen);
        } else if meta.is_file() && extract::is_supported(&child) {
            files.push(child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, name: &str) -> Entry {
        Entry {
            id: id.into(),
            project_id: "p".into(),
            file_name: name.into(),
            original_path: format!("/x/{name}"),
            bytes: 0,
        }
    }

    #[test]
    fn words_fold_case_and_diacritics() {
        let got: Vec<_> = words("Élan, CAFÉ a 42 x-ray ü").into_iter().collect();
        assert_eq!(got, ["42", "cafe", "elan", "ray"]);
        assert!(words("a b c !").is_empty());
    }

    #[test]
    fn pdf_pages_label_excerpts_and_ties_keep_order() {
        let docs = vec![
            (
                entry("d1", "Guide.PDF"),
                "[Page 1]\nalpha beta\n[Page 3]\nalpha gamma".to_owned(),
            ),
            (
                entry("d2", "notes.md"),
                "[Page 9]\nalpha beta gamma".to_owned(),
            ),
        ];
        let idx = index(&docs);
        let hits = search("Alpha", &idx, 5);
        let ids: Vec<_> = hits.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["d1-0", "d1-1", "d2-0"]);
        assert_eq!(hits[0].label, "Guide.PDF · p. 1");
        assert_eq!(hits[1].page, Some(3));
        // Not a PDF: the marker stays text, no page.
        assert_eq!((hits[2].page, hits[2].label.as_str()), (None, "notes.md"));
        let best = search("gamma beta", &idx, 1);
        assert_eq!(best[0].id, "d2-0");
        assert!(search("zzz", &idx, 5).is_empty());
        assert!(search("alpha", &idx, 0).is_empty());
    }

    #[test]
    fn passages_are_bounded_even_without_breaks() {
        let long = "é".repeat(PASSAGE_CHARS * 2 + 10);
        let text = format!("{long}\nshort tail");
        let idx = index(&[(entry("d", "a.txt"), text)]);
        assert!(idx.len() >= 3);
        for p in &idx {
            assert!(
                p.excerpt.text.chars().count() <= PASSAGE_CHARS + 1,
                "{}",
                p.excerpt.text.chars().count()
            );
        }
        // Every character of the source is kept, in order.
        let joined: String = idx
            .iter()
            .map(|p| p.excerpt.text.replace('\n', ""))
            .collect();
        assert_eq!(joined, format!("{long}short tail"));
        let many: Vec<_> = (0..20)
            .map(|i| (entry(&format!("d{i}"), "a.txt"), "word".to_owned()))
            .collect();
        assert_eq!(search("word", &index(&many), 99).len(), MAX_RESULTS);
    }

    #[test]
    fn library_persists_and_enforces_limits() {
        let dir = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();
        fs::create_dir(src.path().join("sub")).unwrap();
        fs::write(src.path().join("a.md"), "Rust ownership rules").unwrap();
        fs::write(src.path().join("sub/b.txt"), "Borrow checker notes").unwrap();
        fs::write(src.path().join("sub/c.bin"), "skip").unwrap();
        fs::write(src.path().join(".hidden.txt"), "skip").unwrap();
        fs::write(
            src.path().join("guide.pdf"),
            extract::tests::pdf_with_pages(&["ownership on page one"]),
        )
        .unwrap();
        let mut lib = Library::open(dir.path()).unwrap();
        let report = lib.add_paths(
            "p1",
            &[src.path().to_owned(), src.path().join("missing.txt")],
        );
        let names: Vec<_> = report.added.iter().map(|e| e.file_name.as_str()).collect();
        assert_eq!(names, ["a.md", "guide.pdf", "b.txt"]);
        assert_eq!(report.skipped.len(), 1);
        assert!(lib.add_file("../evil", &src.path().join("a.md")).is_err());
        let hits = lib.search("p1", "ownership", 5).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].label, "guide.pdf · p. 1");
        assert!(lib.search("p2", "ownership", 5).unwrap().is_empty());

        let reopened = Library::open(dir.path()).unwrap();
        assert_eq!(reopened.list("p1").len(), 3);
        let mut lib = reopened;
        let first = lib.list("p1")[0].id.clone();
        lib.remove(&first).unwrap();
        assert!(lib.remove(&first).is_err());
        assert!(!dir.path().join(format!("{first}.txt")).exists());
        assert_eq!(
            lib.search("p1", "rust", 5).unwrap().len(),
            0,
            "cache was invalidated"
        );
        assert_eq!(lib.remove_project("p1").unwrap(), 2);
        assert!(Library::open(dir.path()).unwrap().list("p1").is_empty());

        // Count limit.
        let mut lib = Library::open(dir.path()).unwrap();
        lib.entries = (0..MAX_DOCUMENTS)
            .map(|i| Entry {
                bytes: 1,
                ..entry(&format!("x{i}"), "a.txt")
            })
            .collect();
        assert!(
            lib.add_file("p", &src.path().join("a.md"))
                .unwrap_err()
                .to_string()
                .contains("full")
        );
        // Byte limit.
        lib.entries = vec![Entry {
            bytes: MAX_LIBRARY_BYTES - 4,
            ..entry("y", "a.txt")
        }];
        assert!(
            lib.add_file("p", &src.path().join("a.md"))
                .unwrap_err()
                .to_string()
                .contains("room")
        );
        // A damaged index is reported rather than silently emptied.
        fs::write(dir.path().join("index.json"), "{").unwrap();
        assert!(Library::open(dir.path()).is_err());
    }
}
