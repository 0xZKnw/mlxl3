use super::{FloatType, MatrixView, Metal};
use crate::hc_workspace_policy::{Key, State};
use anyhow::{Result, ensure};
use metal::{
    CommandBuffer, CommandBufferRef, MTLCommandBufferStatus, MTLResourceOptions as Storage,
};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
pub(super) struct Views {
    pub normalized: MatrixView,
    pub down: MatrixView,
    pub active: MatrixView,
    pub up: Option<MatrixView>,
}

#[derive(Default)]
pub(super) struct Cache {
    state: State,
    views: Option<Views>,
}

pub(super) struct Lease {
    cache: Rc<RefCell<Cache>>,
    command: CommandBuffer,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let completed = self.command.status() == MTLCommandBufferStatus::Completed;
        let mut cache = self.cache.borrow_mut();
        cache.state.release(completed);
        // A dropped NotEnqueued command may still be committed by its caller.
        if !completed {
            cache.views = None;
        }
    }
}

impl Cache {
    pub(super) fn acquire(
        slot: &Rc<RefCell<Self>>,
        gpu: &Metal,
        command: &CommandBufferRef,
        dtype: FloatType,
        with_up: bool,
    ) -> Result<Option<(Lease, Views)>> {
        let key = Key {
            fp32: dtype == FloatType::F32,
            up: with_up,
        };
        {
            let mut cache = slot.borrow_mut();
            let previous = cache.state.key();
            if !cache.state.acquire(key) {
                return Ok(None);
            }
            if previous != Some(key) {
                cache.views = None;
            }
        }
        let lease = Lease {
            cache: Rc::clone(slot),
            command: command.to_owned(),
        };
        if slot.borrow().views.is_none() {
            #[cfg(test)]
            let _scope = crate::flash_profile::span("native.hc.workspace.reserve");
            let output = |n| MatrixView::output(gpu, dtype, 1, n, Storage::StorageModePrivate);
            let views = Views {
                normalized: output(10240)?,
                down: output(320)?,
                active: output(320)?,
                up: with_up.then(|| output(10240)).transpose()?,
            };
            slot.borrow_mut().views = Some(views);
        }
        let views = slot
            .borrow()
            .views
            .as_ref()
            .expect("initialized HC workspace")
            .clone();
        #[cfg(test)]
        gpu.hc_workspace_uses.set(gpu.hc_workspace_uses.get() + 1);
        Ok(Some((lease, views)))
    }

    pub(super) fn clear(&mut self) -> Result<()> {
        ensure!(self.state.clear(), "HC workspace still in flight");
        self.views = None;
        Ok(())
    }
    pub(super) fn record_memory(
        &self,
        backings: &mut super::super::memory::Backings,
    ) -> Result<()> {
        if let Some(v) = &self.views {
            for view in [&v.normalized, &v.down, &v.active]
                .into_iter()
                .chain(v.up.as_ref())
            {
                backings.record("hc_workspace", &view.buffer)?;
            }
        }
        Ok(())
    }
}
