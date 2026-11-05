//! One HC workspace is unavailable until its enclosing command completes.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    pub fp32: bool,
    pub up: bool,
}

impl Key {
    pub(crate) fn bytes(self) -> u64 {
        (10240 + 320 + 320 + if self.up { 10240 } else { 0 }) * if self.fp32 { 4 } else { 2 }
    }
    pub(crate) fn reserve(self, count: u64) -> Option<u64> {
        self.bytes().checked_mul(count)
    }
}

#[derive(Default, Debug)]
pub(crate) struct State {
    key: Option<Key>,
    busy: bool,
}

impl State {
    pub(crate) fn key(&self) -> Option<Key> {
        self.key
    }
    pub(crate) fn acquire(&mut self, key: Key) -> bool {
        if self.busy || self.key.is_some_and(|old| old.fp32 != key.fp32) {
            return false;
        }
        self.busy = true;
        self.key = Some(key);
        true
    }
    pub(crate) fn release(&mut self, completed: bool) {
        self.busy = false;
        if !completed {
            self.key = None;
        }
    }
    pub(crate) fn clear(&mut self) -> bool {
        if self.busy {
            return false;
        }
        self.key = None;
        true
    }
}

pub(crate) fn flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, false).map(|on| on && qualified_m5)
}

#[cfg(any(test, kani))]
#[path = "hc_workspace_policy_tests.rs"]
mod tests;
