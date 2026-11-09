//! A durable intent precedes publication. Neither a completion marker nor a
//! failed registry write is evidence that the published directory can be removed.
use super::*;

const RECEIPT: &str = ".mlxl3-install.json";

#[derive(Serialize, Deserialize)]
pub(super) struct InstallTransaction {
    attempt: String,
    registry: PathBuf,
    request: PendingDownload,
}

fn same_request(a: &PendingDownload, b: &PendingDownload) -> bool {
    a.id == b.id
        && a.repo == b.repo
        && a.commit == b.commit
        && a.folder == b.folder
        && a.name == b.name
        && a.destination == b.destination
        && a.size_bytes == b.size_bytes
}

fn committed_identity(receipt_matches: bool, path_matches: bool, attempt_matches: bool) -> bool {
    receipt_matches && path_matches && attempt_matches
}

impl InstallTransaction {
    pub(super) fn prepare(
        stage: &Path,
        assembled: &Path,
        request: &PendingDownload,
        registry: &Path,
    ) -> Result<Self> {
        // The random temporary name distinguishes reinstallations of the same revision.
        let nonce = tempfile::Builder::new().rand_bytes(32).tempfile_in(stage)?;
        let transaction = Self {
            attempt: nonce
                .path()
                .file_name()
                .context("missing attempt identity")?
                .to_string_lossy()
                .into_owned(),
            registry: std::path::absolute(registry)?,
            request: request.clone(),
        };
        write_json(&assembled.join(RECEIPT), &transaction)?;
        // Payloads may be hard links or copies. Sync both before directory publication.
        for entry in fs::read_dir(assembled)? {
            File::open(entry?.path())?.sync_all()?;
        }
        File::open(assembled)?.sync_all()?;
        write_json(&stage.join("transaction.json"), &transaction)?;
        File::open(stage.parent().context("stage has no parent")?)?.sync_all()?;
        File::open(
            stage
                .parent()
                .and_then(Path::parent)
                .context("stage has no root")?,
        )?
        .sync_all()?;
        Ok(transaction)
    }

    fn load(stage: &Path, request: &PendingDownload) -> Result<Option<Self>> {
        let file = match File::open(stage.join("transaction.json")) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let transaction: Self = serde_json::from_reader(file)?;
        ensure!(
            same_request(&transaction.request, request),
            "download transaction identity changed"
        );
        let root = stage
            .parent()
            .and_then(Path::parent)
            .context("invalid stage root")?;
        ensure!(
            Path::new(&request.destination).parent() == Some(root),
            "transaction escaped managed storage"
        );
        Ok(Some(transaction))
    }

    fn receipt_matches(&self) -> bool {
        File::open(Path::new(&self.request.destination).join(RECEIPT))
            .ok()
            .and_then(|file| serde_json::from_reader::<_, Self>(file).ok())
            .is_some_and(|receipt| {
                receipt.attempt == self.attempt
                    && receipt.registry == self.registry
                    && same_request(&receipt.request, &self.request)
            })
    }

    fn registered(&self) -> Result<Option<crate::registry::ModelEntry>> {
        let models = crate::registry::load(&self.registry)?;
        Ok(models
            .get(&self.request.name)
            .filter(|entry| {
                committed_identity(
                    self.receipt_matches(),
                    entry.path == self.request.destination,
                    entry.installation_id.as_deref() == Some(self.attempt.as_str()),
                )
            })
            .cloned())
    }

    pub(super) fn register(
        &self,
        registry: &Path,
        request: &PendingDownload,
    ) -> Result<crate::registry::ModelEntry> {
        ensure!(
            std::path::absolute(registry)? == self.registry,
            "download registry changed"
        );
        ensure!(
            same_request(&self.request, request) && self.receipt_matches(),
            "published installation identity changed"
        );
        // A successful retry must also make a prior ambiguous directory sync durable.
        File::open(
            Path::new(&request.destination)
                .parent()
                .context("destination has no parent")?,
        )?
        .sync_all()?;
        if let Some(entry) = self.registered()? {
            File::open(self.registry.parent().context("registry has no parent")?)?.sync_all()?;
            return Ok(entry);
        }
        crate::registry::register_managed(
            registry,
            &request.name,
            Path::new(&request.destination),
            &self.attempt,
        )
    }
}

pub(super) fn completed_download(stage: &Path, request: &PendingDownload) -> bool {
    // Read failures and legacy markers fail open for recovery, never hide pending work.
    InstallTransaction::load(stage, request)
        .ok()
        .flatten()
        .and_then(|transaction| transaction.registered().ok().flatten())
        .is_some()
}

pub(super) fn recover_publication(
    stage: &Path,
    request: &PendingDownload,
    registry: &Path,
) -> Result<Option<crate::registry::ModelEntry>> {
    let Some(transaction) = InstallTransaction::load(stage, request)? else {
        return Ok(None);
    };
    if !Path::new(&request.destination).try_exists()? {
        return Ok(None);
    }
    // A foreign or corrupt destination must never be overwritten or rolled back.
    ensure!(
        transaction.receipt_matches(),
        "published installation identity changed"
    );
    File::open(stage)?.sync_all()?;
    transaction.register(registry, request).map(Some)
}

#[cfg(test)]
#[path = "install_transaction_tests.rs"]
mod tests;

#[cfg(kani)]
mod verification {
    #[kani::proof]
    fn completion_requires_all_three_identities() {
        let receipt: bool = kani::any();
        let path: bool = kani::any();
        let attempt: bool = kani::any();
        let committed = super::committed_identity(receipt, path, attempt);
        assert_eq!(committed, receipt && path && attempt);
        kani::cover!(committed);
        kani::cover!(receipt && path && !attempt && !committed);
    }
}
