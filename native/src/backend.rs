//! Shared admission and private, byte-verified executable cache for hybrid builds.
pub fn native_flash(model_type: &[u8]) -> bool {
    model_type == b"qwen4_exp"
}
pub fn valid_helper_size(size: usize) -> bool {
    size > 0 && size <= 256 * 1024 * 1024
}
pub fn valid_helper_key(key: &[u8]) -> bool {
    key.len() == 64
        && key
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}

#[cfg(not(kani))]
mod cache {
    use super::{valid_helper_key, valid_helper_size};
    use anyhow::{Context, Result, ensure};
    use std::{
        fs::{self, File},
        io::{Read, Write},
        os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
        path::{Path, PathBuf},
    };

    pub fn private_directory(path: &Path) -> Result<()> {
        if let Err(error) = fs::symlink_metadata(path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                return Err(error.into());
            }
            fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(path)?;
        }
        let meta = fs::symlink_metadata(path)?;
        ensure!(
            meta.is_dir()
                && !meta.file_type().is_symlink()
                && meta.uid() == unsafe { libc::geteuid() }
                && meta.mode() & 0o077 == 0,
            "unsafe hybrid helper cache directory: {}",
            path.display()
        );
        Ok(())
    }

    fn verified(path: &Path, expected: &[u8]) -> Result<bool> {
        let mut file = match File::options()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error).context("cannot open hybrid helper safely"),
        };
        let meta = file.metadata()?;
        ensure!(
            meta.is_file()
                && meta.nlink() == 1
                && meta.len() == expected.len() as u64
                && meta.uid() == unsafe { libc::geteuid() }
                && meta.mode() & 0o077 == 0
                && meta.mode() & 0o100 != 0,
            "invalid cached hybrid helper"
        );
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take(expected.len() as u64 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes == expected,
            "cached hybrid helper differs from its embedded executable"
        );
        Ok(true)
    }

    pub fn install_helper(root: &Path, key: &str, bytes: &[u8]) -> Result<PathBuf> {
        ensure!(
            valid_helper_key(key.as_bytes()) && valid_helper_size(bytes.len()),
            "invalid hybrid helper identity or size"
        );
        private_directory(root)?;
        let path = root.join(key);
        if verified(&path, bytes)? {
            return Ok(path);
        }
        let mut temp = tempfile::NamedTempFile::new_in(root)?;
        temp.write_all(bytes)?;
        temp.as_file()
            .set_permissions(fs::Permissions::from_mode(0o500))?;
        temp.as_file().sync_all()?;
        match temp.persist_noclobber(&path) {
            Ok(_) => {}
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.error).context("cannot publish hybrid helper"),
        }
        ensure!(
            verified(&path, bytes)?,
            "hybrid helper disappeared after installation"
        );
        Ok(path)
    }
}
#[cfg(not(kani))]
pub use cache::{install_helper, private_directory};

#[cfg(all(test, not(kani)))]
mod tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{PermissionsExt, symlink},
        sync::Arc,
    };
    const KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    #[test]
    fn only_the_actual_flash_type_is_native() {
        assert!(native_flash(b"qwen4_exp"));
        for name in [
            "",
            "QWEN4_EXP",
            "qwen4_exp/",
            "lfm3",
            "lfm3_vl",
            "qwen3_5_moe",
            "my-qwen4_exp",
        ] {
            assert!(!native_flash(name.as_bytes()));
        }
        for i in 0..9 {
            for value in 0..=255u8 {
                let mut name = *b"qwen4_exp";
                name[i] = value;
                assert_eq!(native_flash(&name), name == *b"qwen4_exp");
            }
        }
    }
    #[test]
    fn cache_is_atomic_reusable_and_rejects_corruption_links_and_permissions() -> anyhow::Result<()>
    {
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("private");
        let data = Arc::new(b"a disposable executable fixture".to_vec());
        let handles = (0..4)
            .map(|_| {
                let root = root.clone();
                let data = data.clone();
                std::thread::spawn(move || install_helper(&root, KEY, &data))
            })
            .collect::<Vec<_>>();
        for h in handles {
            assert_eq!(h.join().unwrap()?, root.join(KEY));
        }
        assert_eq!(fs::read_dir(&root)?.count(), 1);
        fs::set_permissions(root.join(KEY), fs::Permissions::from_mode(0o700))?;
        fs::write(root.join(KEY), b"corrupted")?;
        assert!(install_helper(&root, KEY, &data).is_err());
        fs::remove_file(root.join(KEY))?;
        symlink(dir.path().join("outside"), root.join(KEY))?;
        assert!(install_helper(&root, KEY, &data).is_err());
        assert!(!dir.path().join("outside").exists());
        let linked = dir.path().join("linked");
        symlink(&root, &linked)?;
        assert!(install_helper(&linked, KEY, &data).is_err());
        fs::set_permissions(&root, fs::Permissions::from_mode(0o777))?;
        assert!(install_helper(&root, KEY, &data).is_err());
        assert!(install_helper(&root, "../escape", &data).is_err());
        assert!(install_helper(&root, KEY, &[]).is_err());
        Ok(())
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(17)]
fn hybrid_routes_only_exact_flash_identifiers() {
    let data: [u8; 16] = kani::any();
    let length: u8 = kani::any();
    if usize::from(length) <= data.len() {
        let native = native_flash(&data[..usize::from(length)]);
        let expected = length == 9 && data[..9] == *b"qwen4_exp";
        assert_eq!(native, expected);
        kani::cover!(native);
        kani::cover!(length == 7 && &data[..7] == b"lfm3_vl" && !native);
    }
    let size: usize = kani::any();
    assert_eq!(valid_helper_size(size), size > 0 && size <= 268435456);
}
