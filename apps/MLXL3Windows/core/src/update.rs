//! Automatic updates from GitHub releases, mirroring the Mac `UpdateManager`.
//!
//! A release qualifies when it is published (not a draft), its tag is `vX.Y.Z` (stable,
//! not marked prerelease) or, on the beta channel, `vX.Y.Z-beta.N` (marked prerelease),
//! it is newer than the running build and it carries exactly
//! `MLXL3-Desktop-<tag>-Windows-x64-setup.exe` with a GitHub `sha256:` digest and a
//! download URL under `https://github.com/0xZKnw/mlxl3/releases/download/<tag>/`.
//! Ordering matches the Mac: `2.1.0-beta.2 < 2.1.0-beta.10 < 2.1.0`.
//! The installer is downloaded in the background, verified, then run by the user
//! with Tauri's NSIS flags `/P /UPDATE /R` (passive, keep data, restart the app).

use crate::hub::{Http, sha256_file};
use crate::runtime::Command;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

pub const REPOSITORY: &str = "0xZKnw/mlxl3";
pub const RELEASES_URL: &str = "https://api.github.com/repos/0xZKnw/mlxl3/releases?per_page=30";
pub const MAX_INSTALLER_BYTES: u64 = 1_500_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u64, pub u64, pub u64);

impl Version {
    /// `1.2.3` or `v1.2.3`; nothing else (no prerelease suffix, no leading zeros beyond `0`).
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.strip_prefix('v').unwrap_or(text);
        let mut parts = text.split('.');
        let mut next = || {
            let part = parts.next()?;
            let valid = !part.is_empty()
                && part.len() <= 9
                && part.bytes().all(|b| b.is_ascii_digit())
                && (part == "0" || !part.starts_with('0'));
            valid.then(|| part.parse().ok()).flatten()
        };
        let version = Self(next()?, next()?, next()?);
        parts.next().is_none().then_some(version)
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.0, self.1, self.2)
    }
}

/// A Desktop release tag: `vX.Y.Z` or `vX.Y.Z-beta.N` (N ≥ 1, no leading zero).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReleaseId {
    pub version: Version,
    pub beta: Option<u32>,
}

impl ReleaseId {
    pub fn parse(text: &str) -> Option<Self> {
        let (base, beta) = match text.split_once("-beta.") {
            Some((base, number)) => {
                let valid = !number.is_empty()
                    && number.len() <= 6
                    && number.bytes().all(|b| b.is_ascii_digit())
                    && !number.starts_with('0');
                (base, Some(valid.then(|| number.parse().ok()).flatten()?))
            }
            None => (text, None),
        };
        Some(Self {
            version: Version::parse(base)?,
            beta,
        })
    }

    pub fn tag(&self) -> String {
        format!("v{self}")
    }
}

impl std::fmt::Display for ReleaseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.beta {
            Some(n) => write!(f, "{}-beta.{n}", self.version),
            None => write!(f, "{}", self.version),
        }
    }
}

impl Ord for ReleaseId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // A stable release sorts after every beta of the same version.
        self.version
            .cmp(&other.version)
            .then_with(|| match (self.beta, other.beta) {
                (Some(a), Some(b)) => a.cmp(&b),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            })
    }
}

impl PartialOrd for ReleaseId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    #[default]
    Stable,
    Beta,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    #[serde(default)]
    size: u64,
    browser_download_url: String,
    #[serde(default)]
    digest: Option<String>,
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    assets: Vec<Asset>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Update {
    pub tag: String,
    pub version: String,
    pub asset: String,
    pub url: String,
    pub size: u64,
    pub sha256: String,
    pub notes: String,
    pub published_at: Option<String>,
}

pub fn asset_name(release: ReleaseId) -> String {
    format!("MLXL3-Desktop-{}-Windows-x64-setup.exe", release.tag())
}

/// Same rule as the Mac `trustedURL`: https, github.com, the exact release path, no query.
pub fn trusted_url(url: &str, tag: &str, asset: &str) -> bool {
    url == format!("https://github.com/{REPOSITORY}/releases/download/{tag}/{asset}")
}

fn digest(value: Option<&str>) -> Option<String> {
    let hex = value?.strip_prefix("sha256:")?;
    (hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| hex.to_ascii_lowercase())
}

/// The newest qualifying Windows update strictly newer than `current` on `channel`, if any.
/// `current` is the running build's release tag (`v2.1.0`, `2.1.0-beta.3`); an unparsable
/// value counts as `0.0.0`.
pub fn select(releases_json: &[u8], current: &str, channel: Channel) -> Result<Option<Update>> {
    let releases: Vec<Release> =
        serde_json::from_slice(releases_json).context("invalid GitHub releases response")?;
    let current = ReleaseId::parse(current).unwrap_or(ReleaseId {
        version: Version(0, 0, 0),
        beta: None,
    });
    let mut best: Option<(ReleaseId, Update)> = None;
    for release in releases {
        if release.draft || !release.tag_name.starts_with('v') {
            continue;
        }
        let Some(id) = ReleaseId::parse(&release.tag_name) else {
            continue;
        };
        // Stable tags must not be prereleases; betas must be, and only on the beta channel.
        let allowed = match id.beta {
            None => !release.prerelease,
            Some(_) => release.prerelease && channel == Channel::Beta,
        };
        if !allowed || id <= current || best.as_ref().is_some_and(|(b, _)| *b >= id) {
            continue;
        }
        let name = asset_name(id);
        let mut matching = release.assets.iter().filter(|a| a.name == name);
        let (Some(asset), None) = (matching.next(), matching.next()) else {
            continue;
        };
        let Some(sha256) = digest(asset.digest.as_deref()) else {
            continue;
        };
        if asset.size == 0
            || asset.size > MAX_INSTALLER_BYTES
            || !trusted_url(&asset.browser_download_url, &release.tag_name, &name)
        {
            continue;
        }
        best = Some((
            id,
            Update {
                tag: release.tag_name.clone(),
                version: id.to_string(),
                asset: name,
                url: asset.browser_download_url.clone(),
                size: asset.size,
                sha256,
                notes: release
                    .body
                    .unwrap_or_default()
                    .chars()
                    .take(20_000)
                    .collect(),
                published_at: release.published_at,
            },
        ));
    }
    Ok(best.map(|(_, update)| update))
}

pub fn check(http: &dyn Http, current: &str, channel: Channel) -> Result<Option<Update>> {
    select(&http.get(RELEASES_URL, 8 * 1024 * 1024)?, current, channel)
}

fn verified(path: &Path, update: &Update) -> Result<bool> {
    Ok(
        path.metadata().is_ok_and(|m| m.len() == update.size)
            && sha256_file(path)? == update.sha256,
    )
}

/// Download (resuming) and verify the installer into `dir`; returns its path.
pub fn download(
    http: &dyn Http,
    dir: &Path,
    update: &Update,
    cancelled: &AtomicBool,
    report: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf> {
    let id = ReleaseId::parse(&update.tag).context("invalid update tag")?;
    ensure!(
        update.tag == id.tag()
            && update.asset == asset_name(id)
            && trusted_url(&update.url, &update.tag, &update.asset)
            && update.size > 0
            && update.size <= MAX_INSTALLER_BYTES
            && update.sha256.len() == 64,
        "untrusted update"
    );
    fs::create_dir_all(dir)?;
    let target = dir.join(&update.asset);
    if verified(&target, update)? {
        report(update.size, update.size);
        return Ok(target);
    }
    fs::remove_file(&target).ok();
    let partial = dir.join(format!("{}.part", update.asset));
    if partial.metadata().is_ok_and(|m| m.len() > update.size) {
        fs::remove_file(&partial)?;
    }
    http.fetch(&update.url, &partial, cancelled, &mut || {
        report(
            partial.metadata().map_or(0, |m| m.len()).min(update.size),
            update.size,
        )
    })?;
    if !verified(&partial, update)? {
        fs::remove_file(&partial).ok();
        bail!("the downloaded installer does not match the release checksum");
    }
    fs::rename(&partial, &target)?;
    // Older installers are no longer needed.
    for entry in fs::read_dir(dir)?.flatten() {
        if entry.path() != target {
            fs::remove_file(entry.path()).ok();
        }
    }
    report(update.size, update.size);
    Ok(target)
}

/// Passive install that keeps user data and relaunches the app (Tauri NSIS template).
pub fn install_command(installer: &Path) -> Command {
    Command {
        label: "Install update".into(),
        program: installer.to_string_lossy().into_owned(),
        args: vec!["/P".into(), "/UPDATE".into(), "/R".into()],
        env: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::tests::FakeHttp;
    use sha2::{Digest, Sha256};

    fn hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn release(tag: &str, asset: &str, digest: &str, prerelease: bool) -> String {
        format!(
            r#"{{"tag_name":"{tag}","draft":false,"prerelease":{prerelease},"body":"notes {tag}","assets":[
              {{"name":"MLXL3-Desktop-{tag}-b9-Apple-Silicon.dmg","size":10,"browser_download_url":"https://github.com/0xZKnw/mlxl3/releases/download/{tag}/x.dmg","digest":"sha256:{d}"}},
              {{"name":"{asset}","size":12,"browser_download_url":"https://github.com/0xZKnw/mlxl3/releases/download/{tag}/{asset}","digest":"{digest}"}}]}}"#,
            d = "a".repeat(64)
        )
    }

    #[test]
    fn versions_parse_strictly() {
        assert_eq!(Version::parse("v2.10.0"), Some(Version(2, 10, 0)));
        assert_eq!(Version::parse("0.0.0"), Some(Version(0, 0, 0)));
        assert!(Version(2, 10, 0) > Version(2, 9, 9));
        for bad in [
            "",
            "v1",
            "1.2",
            "1.2.3.4",
            "1.2.3-beta",
            "01.2.3",
            "1.x.3",
            "v1.2.3 ",
            "1.2.9999999999",
        ] {
            assert_eq!(Version::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn selects_the_newest_trusted_windows_installer() {
        let sha = format!("sha256:{}", "b".repeat(64));
        let json = format!(
            "[{},{},{},{},{},{}]",
            release(
                "v2.2.0",
                "MLXL3-Desktop-v2.2.0-Windows-x64-setup.exe",
                &sha,
                true
            ),
            release(
                "v2.1.1",
                "MLXL3-Desktop-v2.1.1-Windows-x64-setup.exe",
                "sha256:short",
                false
            ),
            release(
                "v2.1.0",
                "MLXL3-Desktop-v2.1.0-Windows-x64-setup.exe",
                &sha,
                false
            ),
            release(
                "v2.0.9",
                "MLXL3-Desktop-v2.0.9-Windows-x64-setup.exe",
                &sha,
                false
            ),
            release(
                "engine-v9.0.0",
                "MLXL3-Desktop-v9.0.0-Windows-x64-setup.exe",
                &sha,
                false
            ),
            release(
                "v3.0.0",
                "MLXL3-Desktop-v3.0.0-Windows-x64-setup.exe.zip",
                &sha,
                false
            ),
        );
        let update = select(json.as_bytes(), "2.0.3", Channel::Stable)
            .unwrap()
            .unwrap();
        assert_eq!(update.tag, "v2.1.0");
        assert_eq!(update.asset, "MLXL3-Desktop-v2.1.0-Windows-x64-setup.exe");
        assert_eq!(update.sha256, "b".repeat(64));
        assert_eq!(update.notes, "notes v2.1.0");
        assert_eq!(
            select(json.as_bytes(), "2.1.0", Channel::Stable).unwrap(),
            None,
            "not newer"
        );
        assert!(
            select(json.as_bytes(), "0.0.0", Channel::Stable)
                .unwrap()
                .is_some(),
            "dev builds update to a release"
        );
        assert!(select(b"{}", "1.0.0", Channel::Stable).is_err());
    }

    #[test]
    fn release_ids_order_like_the_mac() {
        let id = |t: &str| ReleaseId::parse(t).unwrap();
        assert_eq!(id("v2.1.0-beta.3").to_string(), "2.1.0-beta.3");
        assert_eq!(id("2.1.0").tag(), "v2.1.0");
        assert!(id("v2.1.0-beta.2") < id("v2.1.0-beta.10"));
        assert!(id("v2.1.0-beta.10") < id("v2.1.0"));
        assert!(id("v2.1.0") < id("v2.1.1-beta.1"));
        assert!(id("v2.0.9") < id("v2.1.0-beta.1"));
        for bad in [
            "v2.1.0-beta.",
            "v2.1.0-beta.0",
            "v2.1.0-beta.01",
            "v2.1.0-beta.x",
            "v2.1.0-rc.1",
            "v2.1.0-beta.1-beta.2",
        ] {
            assert_eq!(ReleaseId::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn beta_channel_includes_marked_prereleases_only() {
        let sha = format!("sha256:{}", "d".repeat(64));
        let exe = |tag: &str| format!("MLXL3-Desktop-{tag}-Windows-x64-setup.exe");
        let json = format!(
            "[{},{},{},{}]",
            release("v2.2.0-beta.2", &exe("v2.2.0-beta.2"), &sha, true),
            release("v2.2.0-beta.3", &exe("v2.2.0-beta.3"), &sha, false), // beta tag not marked prerelease
            release("v2.1.0", &exe("v2.1.0"), &sha, false),
            release("v2.1.1", &exe("v2.1.1"), &sha, true), // stable tag marked prerelease
        );
        let stable = select(json.as_bytes(), "v2.0.0", Channel::Stable)
            .unwrap()
            .unwrap();
        assert_eq!(stable.tag, "v2.1.0");
        let beta = select(json.as_bytes(), "v2.0.0", Channel::Beta)
            .unwrap()
            .unwrap();
        assert_eq!(
            (beta.tag.as_str(), beta.version.as_str()),
            ("v2.2.0-beta.2", "2.2.0-beta.2")
        );
        assert_eq!(beta.asset, exe("v2.2.0-beta.2"));
        // A beta build moves to the final release of the same version, never back.
        let json = format!(
            "[{},{}]",
            release("v2.2.0", &exe("v2.2.0"), &sha, false),
            release("v2.2.0-beta.1", &exe("v2.2.0-beta.1"), &sha, true),
        );
        let next = select(json.as_bytes(), "v2.2.0-beta.2", Channel::Stable)
            .unwrap()
            .unwrap();
        assert_eq!(next.tag, "v2.2.0");
        assert_eq!(
            select(json.as_bytes(), "v2.2.0", Channel::Beta).unwrap(),
            None
        );
    }

    #[test]
    fn rejects_untrusted_urls_and_duplicate_assets() {
        assert!(trusted_url(
            "https://github.com/0xZKnw/mlxl3/releases/download/v1.0.0/a.exe",
            "v1.0.0",
            "a.exe"
        ));
        for bad in [
            "http://github.com/0xZKnw/mlxl3/releases/download/v1.0.0/a.exe",
            "https://github.com.evil/0xZKnw/mlxl3/releases/download/v1.0.0/a.exe",
            "https://github.com/other/mlxl3/releases/download/v1.0.0/a.exe",
            "https://github.com/0xZKnw/mlxl3/releases/download/v1.0.0/a.exe?x=1",
            "https://github.com/0xZKnw/mlxl3/releases/download/v0.9.0/a.exe",
        ] {
            assert!(!trusted_url(bad, "v1.0.0", "a.exe"), "{bad}");
        }
        let sha = format!("sha256:{}", "c".repeat(64));
        let name = "MLXL3-Desktop-v1.1.0-Windows-x64-setup.exe";
        let evil = format!(
            r#"[{{"tag_name":"v1.1.0","assets":[{{"name":"{name}","size":5,"browser_download_url":"https://evil.test/{name}","digest":"{sha}"}}]}}]"#
        );
        assert_eq!(
            select(evil.as_bytes(), "1.0.0", Channel::Beta).unwrap(),
            None
        );
        let duplicate = format!(
            r#"[{{"tag_name":"v1.1.0","assets":[{a},{a}]}}]"#,
            a = format!(
                r#"{{"name":"{name}","size":5,"browser_download_url":"https://github.com/0xZKnw/mlxl3/releases/download/v1.1.0/{name}","digest":"{sha}"}}"#
            )
        );
        assert_eq!(
            select(duplicate.as_bytes(), "1.0.0", Channel::Beta).unwrap(),
            None
        );
    }

    #[test]
    fn download_verifies_the_checksum() {
        let body = b"installer-v1";
        let name = asset_name(ReleaseId::parse("v1.1.0").unwrap());
        let update = Update {
            tag: "v1.1.0".into(),
            version: "1.1.0".into(),
            url: format!("https://github.com/0xZKnw/mlxl3/releases/download/v1.1.0/{name}"),
            asset: name,
            size: body.len() as u64,
            sha256: hex(body),
            notes: String::new(),
            published_at: None,
        };
        let http = FakeHttp::default();
        http.route(&update.url, &body[..]);
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path()
                .join("MLXL3-Desktop-v1.0.0-Windows-x64-setup.exe"),
            b"old",
        )
        .unwrap();
        let mut last = (0, 0);
        let path = download(
            &http,
            dir.path(),
            &update,
            &AtomicBool::new(false),
            &mut |d, t| last = (d, t),
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(last, (12, 12));
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            1,
            "old installers are removed"
        );
        // Already downloaded: no new request.
        let before = http.requests.lock().unwrap().len();
        download(
            &http,
            dir.path(),
            &update,
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(http.requests.lock().unwrap().len(), before);

        // Tampered body: rejected and removed.
        fs::remove_file(&path).unwrap();
        http.route(&update.url, &b"installer-v2"[..]);
        let error = download(
            &http,
            dir.path(),
            &update,
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(error.to_string().contains("checksum"));
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);

        // Untrusted metadata never reaches the network.
        let forged = Update {
            url: "https://evil.test/x.exe".into(),
            ..update.clone()
        };
        assert!(
            download(
                &http,
                dir.path(),
                &forged,
                &AtomicBool::new(false),
                &mut |_, _| {}
            )
            .is_err()
        );
        let install = install_command(&path);
        assert_eq!(install.args, ["/P", "/UPDATE", "/R"]);
    }
}
