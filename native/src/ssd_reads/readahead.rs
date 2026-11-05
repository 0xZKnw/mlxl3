//! Dedicated expert read descriptions: never change the inspected checkpoint FD.
use super::*;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::Path,
};

pub(super) struct ExpertFiles {
    files: RefCell<BTreeMap<[u64; 2], File>>,
    #[cfg(test)]
    selected_records: Cell<usize>,
}
fn identity(file: &File) -> Result<[u64; 2]> {
    let m = file.metadata()?;
    Ok([m.dev(), m.ino()])
}
fn disable_readahead(fd: std::os::fd::RawFd) -> Result<()> {
    // This setter affects an open-file description; the FD must be independently opened.
    ensure!(
        unsafe { libc::fcntl(fd, libc::F_RDAHEAD, 0) } == 0,
        "disabling expert read-ahead failed: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}
impl ExpertFiles {
    pub(super) fn from_env() -> Result<Option<Self>> {
        let value = std::env::var("MLXL3_NATIVE_EXPERT_READAHEAD_OFF");
        let enabled = match value {
            Ok(value) => common_queue_flag(Some(&value)),
            Err(std::env::VarError::NotPresent) => common_queue_flag(None),
            Err(error) => return Err(error.into()),
        }
        .context("MLXL3_NATIVE_EXPERT_READAHEAD_OFF must be 0 or 1")?;
        Ok(enabled.then(Self::new))
    }
    fn new() -> Self {
        Self {
            files: RefCell::new(BTreeMap::new()),
            #[cfg(test)]
            selected_records: Cell::new(0),
        }
    }
    pub(super) fn source(&self, path: &Path, inspected: &File) -> Result<File> {
        let key = identity(inspected)?;
        let mut files = self.files.borrow_mut();
        if let std::collections::btree_map::Entry::Vacant(entry) = files.entry(key) {
            let file = File::open(path).with_context(|| {
                format!("opening dedicated expert descriptor {}", path.display())
            })?;
            ensure!(
                readahead_identity_matches(key, identity(&file)?),
                "expert file identity changed after inspection"
            );
            disable_readahead(file.as_raw_fd())?;
            entry.insert(file);
        }
        let file = files
            .get(&key)
            .context("missing dedicated expert descriptor")?
            .try_clone()?;
        #[cfg(test)]
        self.selected_records.set(self.selected_records.get() + 1);
        Ok(file)
    }
    #[cfg(test)]
    pub(super) fn counts(&self) -> [usize; 2] {
        [self.files.borrow().len(), self.selected_records.get()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};

    #[test]
    #[ignore = "Darwin F_RDAHEAD actual descriptor/byte/error oracle; run alone"]
    fn physical_ar20_dedicated_readahead_preserves_inode_bytes_and_original_description()
    -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("experts");
        let data = (0..EXPERT_BYTES as usize + 19)
            .map(|i| ((i * 37 + i / 127) % 251) as u8)
            .collect::<Vec<_>>();
        std::fs::write(&path, &data)?;
        let mut original = File::options().read(true).write(true).open(&path)?;
        original.seek(SeekFrom::Start(3))?;
        let files = ExpertFiles::new();
        let mut fd = files.source(&path, &original)?;
        fd.seek(SeekFrom::Start(17))?;
        ensure!(
            original.stream_position()? == 3,
            "dedicated FD shares the original open description"
        );
        let mut actual = vec![0; EXPERT_BYTES as usize];
        fd.read_exact_at(&mut actual, 3)?;
        ensure!(actual == data[3..3 + actual.len()] && actual.iter().any(|&x| x != 0));
        std::fs::rename(&path, dir.path().join("old"))?;
        std::fs::write(&path, vec![0; data.len()])?;
        let cached = files.source(&path, &original)?;
        cached.read_exact_at(&mut actual, 3)?;
        ensure!(actual == data[3..3 + actual.len()]);
        ensure!(
            ExpertFiles::new().source(&path, &original).is_err(),
            "replaced inode accepted"
        );
        original.set_len(7)?;
        ensure!(cached.read_exact_at(&mut actual, 3).is_err());
        ensure!(disable_readahead(-1).is_err());
        ensure!(files.counts() == [1, 2]);
        eprintln!(
            "AR20_EXPERT_READAHEAD {{\"parity\":true,\"bytes\":1248012,\"configured_fds\":1,\"selected_records\":2,\"independent_description\":true,\"path_replacement_rejected\":true,\"eof_and_badfd_rejected\":true}}"
        );
        Ok(())
    }
}
