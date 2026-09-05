//! Process-neutral C ABI definitions required by both the Bun binary and the
//! Nimbus shared embedder.

#![allow(non_snake_case, clippy::missing_safety_doc)]

// Force-link `bun_platform` so its C exports reach every Rust static-library
// root that owns this bridge.
use bun_platform as _;

/// Panic entry point for native callers. This keeps shared-embedder failures
/// on the same crash-report path as the Bun binary.
#[unsafe(no_mangle)]
extern "C" fn Bun__panic(msg: *const u8, len: usize) -> ! {
    let bytes = if msg.is_null() {
        &b""[..]
    } else {
        // SAFETY: The native caller guarantees that a non-null pointer is
        // readable for `len` bytes for the duration of this call.
        unsafe { core::slice::from_raw_parts(msg, len) }
    };
    bun_core::output::panic(format_args!("{}", bstr::BStr::new(bytes)));
}

/// Out-of-memory entry point for native callers that cannot propagate an
/// allocation failure.
#[unsafe(no_mangle)]
extern "C" fn Bun__outOfMemory() -> ! {
    bun_core::out_of_memory()
}
