//! Windows paths as seen from a WSL2 distribution, and WSL command output.

use anyhow::{Result, bail, ensure};

/// Translate an absolute Windows path for use inside `distro`.
///
/// * `C:\Users\me\m` → `/mnt/c/Users/me/m` (drvfs mount, the WSL default)
/// * `\\wsl.localhost\<distro>\home\me\m` or `\\wsl$\<distro>\...` → `/home/me/m`
///
/// Other UNC paths, relative paths, `..` components and characters that drvfs
/// cannot represent are rejected rather than guessed.
pub fn to_wsl_path(windows: &str, distro: &str) -> Result<String> {
    ensure!(!windows.is_empty(), "path is empty");
    ensure!(
        !windows.contains('\0') && !windows.contains('\n'),
        "path contains a control character"
    );
    let mut normalized = windows.replace('/', "\\");
    // Verbatim forms: \\?\UNC\host\share → \\host\share, \\?\C:\x → C:\x
    if let Some(rest) = normalized.strip_prefix("\\\\?\\UNC\\") {
        normalized = format!("\\\\{rest}");
    } else if let Some(rest) = normalized.strip_prefix("\\\\?\\") {
        normalized = rest.to_owned();
    }
    if let Some(rest) = normalized.strip_prefix("\\\\") {
        let mut parts = rest.split('\\');
        let host = parts.next().unwrap_or_default();
        ensure!(
            host.eq_ignore_ascii_case("wsl.localhost") || host.eq_ignore_ascii_case("wsl$"),
            "network paths are not supported; copy the model to a local drive"
        );
        let name = parts.next().unwrap_or_default();
        ensure!(
            name.eq_ignore_ascii_case(distro),
            "the path belongs to WSL distribution {name:?}, not {distro:?}"
        );
        return join("", parts);
    }
    let bytes = normalized.as_bytes();
    if bytes.len() < 3 || !bytes[0].is_ascii_alphabetic() || bytes[1] != b':' || bytes[2] != b'\\' {
        bail!("expected an absolute path such as C:\\Models\\name");
    }
    let drive = (bytes[0] as char).to_ascii_lowercase();
    join(&format!("/mnt/{drive}"), normalized[3..].split('\\'))
}

fn join<'a>(prefix: &str, parts: impl Iterator<Item = &'a str>) -> Result<String> {
    let mut out = prefix.to_owned();
    for part in parts.filter(|p| !p.is_empty() && *p != ".") {
        ensure!(part != "..", "paths with '..' are not supported");
        ensure!(
            !part.contains([':', '*', '?', '"', '<', '>', '|']),
            "invalid character in {part:?}"
        );
        out.push('/');
        out.push_str(part);
    }
    if out.is_empty() {
        out.push('/');
    }
    Ok(out)
}

/// Decode `wsl.exe --list --quiet` output (UTF-16LE, optional BOM, CRLF, NUL padding).
pub fn parse_distro_list(raw: &[u8]) -> Vec<String> {
    let text = if raw.len() >= 2
        && raw.len().is_multiple_of(2)
        && raw.iter().skip(1).step_by(2).any(|b| *b == 0)
    {
        let units: Vec<u16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        String::from_utf16_lossy(&units)
    } else {
        String::from_utf8_lossy(raw).into_owned()
    };
    text.trim_start_matches('\u{feff}')
        .lines()
        .map(|l| l.trim_matches(|c: char| c.is_whitespace() || c == '\0'))
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Distribution names are passed as a separate argv element to `wsl.exe -d`;
/// still refuse anything that is not a plain name.
pub fn validate_distro(name: &str) -> Result<()> {
    ensure!(
        name.len() <= 64
            && name.starts_with(|c: char| c.is_ascii_alphanumeric())
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
        "invalid WSL distribution name {name:?}"
    );
    Ok(())
}

/// `$HOME` reported by the distribution, used to place the runtime on ext4.
pub fn parse_home(raw: &[u8]) -> Result<String> {
    let home = String::from_utf8_lossy(raw).trim().to_owned();
    ensure!(
        home.starts_with('/')
            && !home.contains(['\n', '\0', '\\'])
            && !home.split('/').any(|p| p == ".."),
        "unexpected WSL home directory {home:?}"
    );
    Ok(home.trim_end_matches('/').to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drive_paths() {
        assert_eq!(
            to_wsl_path(r"C:\Users\Me\Models\Q", "Ubuntu").unwrap(),
            "/mnt/c/Users/Me/Models/Q"
        );
        assert_eq!(to_wsl_path("d:/m/./x/", "Ubuntu").unwrap(), "/mnt/d/m/x");
        assert_eq!(
            to_wsl_path(r"\\?\E:\long\path", "Ubuntu").unwrap(),
            "/mnt/e/long/path"
        );
        assert_eq!(to_wsl_path(r"Z:\", "Ubuntu").unwrap(), "/mnt/z");
        assert_eq!(
            to_wsl_path(r"C:\Modèles\é", "U").unwrap(),
            "/mnt/c/Modèles/é"
        );
    }

    #[test]
    fn wsl_share_paths() {
        assert_eq!(
            to_wsl_path(r"\\wsl.localhost\Ubuntu\home\me\m", "Ubuntu").unwrap(),
            "/home/me/m"
        );
        assert_eq!(to_wsl_path(r"\\wsl$\ubuntu\opt", "Ubuntu").unwrap(), "/opt");
        assert_eq!(
            to_wsl_path(r"\\?\UNC\wsl.localhost\Ubuntu\srv", "Ubuntu").unwrap(),
            "/srv"
        );
        assert_eq!(
            to_wsl_path(r"\\wsl.localhost\Ubuntu", "Ubuntu").unwrap(),
            "/"
        );
    }

    #[test]
    fn rejected_paths() {
        for path in [
            "",
            "relative\\x",
            "C:relative",
            "\\rooted",
            r"\\server\share\x",
            r"\\wsl.localhost\Debian\home",
            r"C:\a\..\b",
            r"C:\a:b",
            r"C:\a*",
            "C:\\a\nb",
            "1:\\x",
        ] {
            assert!(to_wsl_path(path, "Ubuntu").is_err(), "{path:?}");
        }
    }

    #[test]
    fn distro_list_utf16_and_utf8() {
        let mut raw = vec![0xff, 0xfe];
        for unit in "Ubuntu-24.04\r\n\r\ndocker-desktop\r\n".encode_utf16() {
            raw.extend(unit.to_le_bytes());
        }
        assert_eq!(parse_distro_list(&raw), ["Ubuntu-24.04", "docker-desktop"]);
        assert_eq!(parse_distro_list(b"Ubuntu\nArch\n"), ["Ubuntu", "Arch"]);
        assert!(parse_distro_list(b"").is_empty());
    }

    #[test]
    fn distro_and_home_validation() {
        assert!(validate_distro("Ubuntu-24.04").is_ok());
        for bad in ["", "a b", "x;rm", "-d", ".hidden", &"a".repeat(65)] {
            assert!(validate_distro(bad).is_err(), "{bad:?}");
        }
        assert_eq!(parse_home(b"/home/me/\n").unwrap(), "/home/me");
        assert!(parse_home(b"relative").is_err());
        assert!(parse_home(b"/home/../etc").is_err());
    }
}
