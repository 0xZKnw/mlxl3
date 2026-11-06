//! Owned immutable cache snapshot: read leases retain the same owners as A11's cache.
use crate::{device_hot_policy as policy, flash_support::EXPERT_BYTES};
use anyhow::{Context, Result, ensure};
use metal::{Buffer, MTLStorageMode, foreign_types::ForeignType};
use std::{cell::Cell, rc::Rc};

pub(super) struct RecordOwner {
    backing: Buffer,
    readers: Cell<usize>,
}

pub(super) enum RecordBacking {
    Plain(Buffer),
    Tracked(Rc<RecordOwner>),
}

impl RecordBacking {
    pub(super) fn new(backing: Buffer, tracked: bool) -> Self {
        if tracked {
            Self::Tracked(Rc::new(RecordOwner {
                backing,
                readers: Cell::new(0),
            }))
        } else {
            Self::Plain(backing)
        }
    }
    pub(super) fn raw(&self) -> &Buffer {
        match self {
            Self::Plain(buffer) => buffer,
            Self::Tracked(owner) => &owner.backing,
        }
    }
    pub(super) fn matches(&self, backing: &Buffer) -> bool {
        self.raw().as_ptr() == backing.as_ptr()
    }
    /// Old GPU producers must already have been drained and projection views dropped.
    pub(super) fn into_reusable(self) -> Option<Buffer> {
        match self {
            Self::Plain(backing) => Some(backing),
            Self::Tracked(owner) => (owner.readers.get() == 0).then(|| owner.backing.clone()),
        }
    }
}

pub(super) struct RecordReadOwner(Rc<RecordOwner>);
impl RecordReadOwner {
    fn acquire(owner: &Rc<RecordOwner>) -> Result<Self> {
        let next = policy::readers_after(owner.readers.get(), true)
            .context("record read lease overflow")?;
        owner.readers.set(next);
        Ok(Self(owner.clone()))
    }
}
impl Drop for RecordReadOwner {
    fn drop(&mut self) {
        self.0.readers.set(
            policy::readers_after(self.0.readers.get(), false).expect("balanced record read lease"),
        );
    }
}

pub(super) struct ResidentSnapshot {
    map: [u32; policy::EXPERTS],
    identities: Vec<Option<(u32, u64)>>,
    records: Vec<Option<Buffer>>,
    _read_owners: Vec<RecordReadOwner>,
}
impl ResidentSnapshot {
    pub(super) fn from_cache(
        identities: Vec<Option<(u32, u64)>>,
        owners: &[Option<RecordBacking>],
    ) -> Result<Self> {
        ensure!(
            policy::supported_slots(identities.len()) && owners.len() == identities.len(),
            "device Hot requires 53 or 64 matching slots"
        );
        let mut map = [u32::MAX; policy::EXPERTS];
        let mut records = Vec::with_capacity(identities.len());
        let mut read_owners = Vec::new();
        for (slot, identity) in identities.iter().enumerate() {
            if let Some((expert, _)) = identity {
                let entry = map
                    .get_mut(*expert as usize)
                    .context("ready expert outside vocabulary")?;
                ensure!(*entry == u32::MAX, "duplicate ready expert");
                let owner = owners[slot]
                    .as_ref()
                    .context("ready slot lacks record owner")?;
                let RecordBacking::Tracked(owner) = owner else {
                    anyhow::bail!("ready record lacks typed ownership")
                };
                ensure!(
                    owner.backing.length() == EXPERT_BYTES
                        && owner.backing.storage_mode() == MTLStorageMode::Shared,
                    "invalid ready record backing"
                );
                ensure!(
                    !records
                        .iter()
                        .flatten()
                        .any(|old: &Buffer| old.as_ptr() == owner.backing.as_ptr()),
                    "aliased ready slots"
                );
                read_owners.push(RecordReadOwner::acquire(owner)?);
                records.push(Some(owner.backing.clone()));
                *entry = slot as u32;
            } else {
                records.push(None);
            }
        }
        Ok(Self {
            map,
            identities,
            records,
            _read_owners: read_owners,
        })
    }
    pub(super) fn map(&self) -> &[u32; policy::EXPERTS] {
        &self.map
    }
    pub(super) fn records(&self) -> &[Option<Buffer>] {
        &self.records
    }
    pub(super) fn validate_routes(
        &self,
        current: &[Option<(u32, u64)>],
        ids: &[u32],
    ) -> Result<Vec<Option<usize>>> {
        ensure!(
            current == self.identities,
            "ready cache changed before route publication"
        );
        ensure!(
            ids.len() == policy::ROUTES,
            "invalid device Hot route count"
        );
        let mut seen = [false; policy::EXPERTS];
        ids.iter()
            .map(|&id| {
                let used = seen
                    .get_mut(id as usize)
                    .context("GPU route ID outside vocabulary")?;
                ensure!(!*used, "duplicate GPU route ID");
                *used = true;
                let slot = self.map[id as usize];
                if slot == u32::MAX {
                    return Ok(None);
                }
                let slot = slot as usize;
                let identity = self.identities[slot].context("mapped slot not ready")?;
                ensure!(
                    policy::identity_matches(id, identity.1, current[slot]),
                    "stale device Hot generation"
                );
                Ok(Some(slot))
            })
            .collect()
    }
    pub(super) fn retired_bytes(&self, misses: impl IntoIterator<Item = usize>) -> Result<u64> {
        let mut bits = 0u64;
        let mut count = 0;
        for slot in misses {
            let bit = policy::retirement_bit(slot, self.identities.len())
                .context("retired slot outside snapshot")?;
            if self.identities[slot].is_some() && bits & bit == 0 {
                bits |= bit;
                count += 1;
            }
        }
        policy::retirement_bytes(count).context("device Hot retired-record budget exceeded")
    }
    #[cfg(test)]
    pub(super) fn for_test(
        map: [u32; policy::EXPERTS],
        records: Vec<Option<Buffer>>,
    ) -> Result<Self> {
        ensure!(
            policy::supported_slots(records.len()),
            "invalid test record count"
        );
        let mut identities = vec![None; records.len()];
        for (expert, &slot) in map.iter().enumerate() {
            if slot == u32::MAX {
                continue;
            }
            let identity = identities
                .get_mut(slot as usize)
                .context("test map slot outside table")?;
            ensure!(identity.is_none(), "multiple test experts share a slot");
            *identity = Some((expert as u32, 1));
        }
        let owners = records
            .into_iter()
            .map(|record| record.map(|b| RecordBacking::new(b, true)))
            .collect::<Vec<_>>();
        let snapshot = Self::from_cache(identities, &owners)?;
        ensure!(snapshot.map == map, "test map differs from ready snapshot");
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        checkpoint,
        flash_support::{DemandPlan, ExpertCache},
        ssd_reads::{ReadPayload, ReadService},
    };
    use std::{cell::RefCell, io::Write};

    #[test]
    #[ignore = "physical Shared cache ownership and real fresh SSD destination; no model"]
    fn physical_ar34_ready_snapshot_preserves_owners_and_rejects_stale_generation() -> Result<()> {
        let device = metal::Device::system_default().context("Metal device required")?;
        let old = crate::gpu::upload(&device, &vec![0x31u8; EXPERT_BYTES as usize])?;
        let mut owners = (0..policy::SLOTS).map(|_| None).collect::<Vec<_>>();
        owners[52] = Some(RecordBacking::new(old.clone(), true));
        let RecordBacking::Tracked(owner) = owners[52].as_ref().unwrap() else {
            unreachable!()
        };
        let observer = owner.clone();
        let mut current = vec![None; policy::SLOTS];
        current[52] = Some((7, 99));
        let snapshot = ResidentSnapshot::from_cache(current.clone(), &owners)?;
        ensure!(observer.readers.get() == 1 && snapshot.map()[7] == 52);
        let ids = [7, 10, 11, 12, 13, 14, 15, 16, 17, 18];
        let slots = snapshot.validate_routes(&current, &ids)?;
        ensure!(slots[0] == Some(52) && slots[1..].iter().all(Option::is_none));
        let mut stale = current.clone();
        stale[52] = Some((7, 100));
        ensure!(snapshot.validate_routes(&stale, &ids).is_err());
        stale[52] = Some((8, 99));
        ensure!(snapshot.validate_routes(&stale, &ids).is_err());
        let mut duplicate = ids;
        duplicate[9] = 7;
        ensure!(snapshot.validate_routes(&current, &duplicate).is_err());
        ensure!(snapshot.retired_bytes([52, 52, 0])? == EXPERT_BYTES);
        ensure!(snapshot.retired_bytes([53]).is_err());
        ensure!(owners[52].take().unwrap().into_reusable().is_none());
        ensure!(
            observer.readers.get() == 1,
            "retiring cache released a read lease"
        );

        // The actual foreground API has no reusable-owner parameter. Keep the old
        // typed lease alive while a replacement is read into a distinct allocation.
        let mut file = tempfile::NamedTempFile::new()?;
        let header = serde_json::to_vec(&serde_json::json!({"record": {
            "dtype":"U8", "shape":[EXPERT_BYTES], "data_offsets":[0,EXPERT_BYTES]
        }}))?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        let expected = vec![0xa7u8; EXPERT_BYTES as usize];
        file.write_all(&expected)?;
        file.flush()?;
        let record = checkpoint::read_header(file.path())?
            .remove("record")
            .unwrap();
        let cache = Rc::new(RefCell::new(ExpertCache::new(policy::SLOTS)?));
        let lease = DemandPlan::reserve(&cache, &[10])?.reads[0];
        let service = ReadService::new(2)?;
        let mut batch = service.submit_shared_fresh([(lease, record)], &device)?;
        let mut ready = batch.ready(true)?;
        ensure!(ready.len() == 1 && ready[0].0 == lease);
        let ReadPayload::Shared(fresh) = ready.pop().unwrap().1 else {
            anyhow::bail!("Shared destination required")
        };
        ensure!(fresh.as_ptr() != old.as_ptr());
        let bytes = unsafe {
            std::slice::from_raw_parts(fresh.contents().cast::<u8>(), EXPERT_BYTES as usize)
        };
        ensure!(bytes == expected);
        let retained = unsafe {
            std::slice::from_raw_parts(old.contents().cast::<u8>(), EXPERT_BYTES as usize)
        };
        ensure!(retained.iter().all(|&byte| byte == 0x31));
        drop(batch);
        ensure!(service.in_flight() == 0);
        drop(snapshot);
        ensure!(observer.readers.get() == 0);
        let reusable = RecordBacking::Tracked(observer)
            .into_reusable()
            .context("released backing must become reusable")?;
        ensure!(reusable.as_ptr() == old.as_ptr());
        eprintln!(
            "AR34_DEVICE_HOT_CACHE {{\"bytes\":{},\"retained_owner\":true,\"fresh_destination\":true,\"stale_generation_rejected\":true,\"remap_rejected\":true,\"released_reusable\":true,\"parity\":true}}",
            EXPERT_BYTES
        );
        Ok(())
    }
}
