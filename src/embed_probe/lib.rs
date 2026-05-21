//! Non-CLI Bun/JSC embed proof root.
//!
//! This crate deliberately avoids `bun_bin`: no process `main`, no global
//! allocator override, no crash/signal/stdio setup, no CLI dispatch, and no
//! process exit. The native build graph links this archive with Bun's normal
//! C++/WebKit/JSC objects through the opt-in `check-bun-embed-probe` target.

use bun_jsc::virtual_machine::{InitOptions, VirtualMachine};

// Force-link the shared C ABI bridge that still owns process-neutral symbols
// required by Bun's native object graph.
use bun_link_bridge as _;

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_construct_and_destroy_vm() -> i32 {
    bun_core::output::init_test();
    bun_runtime::allocators::register_safety_vtables();
    bun_jsc::initialize(false);

    // Touch the high-tier runtime hooks so this staticlib root owns
    // `__BUN_RUNTIME_HOOKS` without depending on Bun's process-owned CLI root.
    bun_runtime::jsc_hooks::embedder_touch_runtime_state();

    let opts = InitOptions {
        is_main_thread: false,
        ..Default::default()
    };

    match VirtualMachine::init(opts) {
        Ok(vm) => {
            // SAFETY: `VirtualMachine::init` returned a fresh VM pointer owned
            // by this probe. No other Rust reference exists, and this path
            // immediately tears it down before returning to the C driver.
            unsafe { (&mut *vm).destroy() };
            0
        }
        Err(_) => 1,
    }
}
