//! The optional set owns only fixed-weight handles; it never follows later allocations.
use super::{FlashNext, Metal};
use anyhow::Result;
use metal::{Buffer, CommandQueue, foreign_types::ForeignType};
use objc::{
    msg_send,
    rc::StrongPtr,
    runtime::{BOOL, Class, Object, YES},
    sel, sel_impl,
};

pub(super) struct FixedResidency {
    set: StrongPtr,
    queue: CommandQueue,
    _buffers: Vec<Buffer>,
    #[cfg(test)]
    bytes: u64,
}

#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macros use legacy cargo-clippy cfg"
)]
impl FixedResidency {
    fn new(gpu: &Metal, buffers: Vec<Buffer>, bytes: u64, limit: u64) -> Option<Self> {
        let descriptor_class = Class::get("MTLResidencySetDescriptor")?;
        let available = unsafe {
            let device: BOOL = msg_send![gpu.device.as_ref(), respondsToSelector: sel!(newResidencySetWithDescriptor:error:)];
            let queue: BOOL =
                msg_send![gpu.queue.as_ref(), respondsToSelector: sel!(addResidencySet:)];
            let remove: BOOL =
                msg_send![gpu.queue.as_ref(), respondsToSelector: sel!(removeResidencySet:)];
            device == YES && queue == YES && remove == YES
        };
        if !crate::fixed_residency_policy::eligible(
            gpu.fixed_residency_enabled.get(),
            available,
            bytes,
            limit,
        ) {
            return None;
        }
        // Both `new` methods transfer +1 ownership; StrongPtr releases it on every return path.
        objc::rc::autoreleasepool(|| unsafe {
            let descriptor: *mut Object = msg_send![descriptor_class, new];
            if descriptor.is_null() {
                return None;
            }
            let descriptor = StrongPtr::new(descriptor);
            let _: () = msg_send![*descriptor, setInitialCapacity: buffers.len()];
            let mut error: *mut Object = std::ptr::null_mut();
            let set: *mut Object = msg_send![gpu.device.as_ref(), newResidencySetWithDescriptor: *descriptor error: &mut error];
            if set.is_null() {
                return None;
            }
            let set = StrongPtr::new(set);
            for buffer in &buffers {
                let _: () = msg_send![*set, addAllocation: buffer.as_ptr()];
            }
            let _: () = msg_send![*set, commit];
            let _: () = msg_send![*set, requestResidency];
            let _: () = msg_send![gpu.queue.as_ref(), addResidencySet: *set];
            Some(Self {
                set,
                queue: gpu.queue.to_owned(),
                _buffers: buffers,
                #[cfg(test)]
                bytes,
            })
        })
    }
    #[cfg(test)]
    fn diagnostics(&self) -> [u64; 2] {
        let count: usize = unsafe { msg_send![*self.set, allocationCount] };
        [count as u64, self.bytes]
    }
}

#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macros use legacy cargo-clippy cfg"
)]
impl Drop for FixedResidency {
    fn drop(&mut self) {
        // Queue/weights remain owned until all detach/end messages and set release finish.
        unsafe {
            let _: () = msg_send![self.queue.as_ref(), removeResidencySet: *self.set];
            let _: () = msg_send![*self.set, endResidency];
        }
    }
}

impl FlashNext<'_> {
    pub(super) fn ensure_fixed_residency(&mut self) -> Result<()> {
        if self.fixed_residency_checked || !self.gpu.fixed_residency_enabled.get() {
            return Ok(());
        }
        let mut backings = super::memory::Backings::collect_fixed();
        self.head.record_memory(&mut backings)?;
        self.mixer.record_memory(&mut backings)?;
        for layer in &self.layers {
            layer.record_memory(&mut backings)?;
        }
        let (buffers, bytes) = backings.take_fixed()?;
        let recommended = self.gpu.device.recommended_max_working_set_size();
        let limit = self
            .request_limit
            .map_or(recommended, |n| n.min(recommended));
        self.fixed_residency = FixedResidency::new(self.gpu, buffers, bytes, limit);
        self.fixed_residency_checked = true;
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn fixed_residency_diagnostics(&self) -> [u64; 2] {
        self.fixed_residency
            .as_ref()
            .map_or([0, 0], FixedResidency::diagnostics)
    }
}
