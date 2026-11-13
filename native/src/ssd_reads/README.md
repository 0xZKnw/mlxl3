# SSD reads

[cancellable_io.rs](cancellable_io.rs), [readahead.rs](readahead.rs), and [window.rs](window.rs) supplement [ssd_reads.rs](../ssd_reads.rs). Test files cover cancellation, errors, queues, reuse, and split reads. Related C/C++ contracts are in [native/ffi/](../../ffi/README.md).
