pub const EXPERT_BYTES: u64 = 1_248_012;
pub(crate) fn direct_destination_valid(bytes: usize, length: u64, shared: bool) -> bool {
    bytes > 0 && bytes <= EXPERT_BYTES as usize && length == bytes as u64 && shared
}
#[test]
fn rejects_non_shared_owned_destination() { assert!(!direct_destination_valid(17, 17, false)); }
