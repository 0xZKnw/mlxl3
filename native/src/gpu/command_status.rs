//! Failure diagnostics only: never wait or inspect NSError on the success path.
#![allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macros test legacy cargo-clippy cfg"
)]
use anyhow::{Result, bail};
use metal::{CommandBufferRef, MTLCommandBufferStatus};
use objc::{msg_send, sel, sel_impl};

pub(super) fn completed(command: &CommandBufferRef, operation: &str) -> Result<()> {
    let status = command.status();
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    // MTLCommandBuffer.error is a nullable NSError. Copy NSString UTF-8 bytes
    // while the command retains the error; no borrowed pointer escapes.
    let detail = unsafe {
        let error: *mut objc::runtime::Object = msg_send![command, error];
        if error.is_null() {
            "NSError unavailable".to_owned()
        } else {
            let domain: *mut objc::runtime::Object = msg_send![error, domain];
            let code: isize = msg_send![error, code];
            let description: *mut objc::runtime::Object = msg_send![error, localizedDescription];
            format!(
                "domain={}, code={code}, description={}",
                string(domain),
                string(description)
            )
        }
    };
    bail!("{operation}: Metal command failed ({status:?}); {detail}")
}

// SAFETY: callers supply a live nullable NSString returned by NSError.
unsafe fn string(value: *mut objc::runtime::Object) -> String {
    if value.is_null() {
        return "<unavailable>".to_owned();
    }
    let utf8: *const std::ffi::c_char = unsafe { msg_send![value, UTF8String] };
    if utf8.is_null() {
        "<unavailable>".to_owned()
    } else {
        unsafe { std::ffi::CStr::from_ptr(utf8) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a physical Apple GPU; success and nil-error diagnostics, not GPU fault injection"]
    fn physical_command_status_success_and_missing_error() -> Result<()> {
        let device =
            metal::Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
        let queue = device.new_command_queue();
        let command = queue.new_command_buffer();
        let error = completed(command, "diagnostic fixture layer=7")
            .unwrap_err()
            .to_string();
        assert!(error.contains("layer=7") && error.contains("NSError unavailable"));
        command.commit();
        command.wait_until_completed();
        completed(command, "diagnostic fixture")
    }
}
