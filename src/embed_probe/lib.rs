//! Non-CLI Bun/JSC embed proof root.
//!
//! This crate deliberately avoids `bun_bin`: no process `main`, no global
//! allocator override, no crash/signal/stdio setup, no CLI dispatch, and no
//! process exit. The native build graph links this archive with Bun's normal
//! C++/WebKit/JSC objects through the opt-in `check-bun-embed-probe` target.

use core::sync::atomic::{AtomicI32, Ordering};

use bun_jsc::virtual_machine::{InitOptions, VirtualMachine};
use bun_jsc::{CallFrame, JSFunction, JSGlobalObject, JSValue, JsResult};

// Force-link the shared C ABI bridge that still owns process-neutral symbols
// required by Bun's native object graph.
use bun_link_bridge as _;

static HOST_CALL_COUNT: AtomicI32 = AtomicI32::new(0);
static HOST_CALL_PAYLOAD: AtomicI32 = AtomicI32::new(0);
static HOST_CALL_RETURNED: AtomicI32 = AtomicI32::new(0);

unsafe extern "C" {
    fn Bun__REPL__evaluate(
        global_object: *const JSGlobalObject,
        source_ptr: *const u8,
        source_len: usize,
        filename_ptr: *const u8,
        filename_len: usize,
        exception: *mut JSValue,
    ) -> JSValue;
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_construct_and_destroy_vm() -> i32 {
    construct_vm_and_run(|_| 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_sync_host_call() -> i32 {
    construct_vm_and_run(run_sync_host_call_probe)
}

fn construct_vm_and_run(run: impl FnOnce(&mut VirtualMachine) -> i32) -> i32 {
    bun_core::output::init_test();
    bun_runtime::allocators::register_safety_vtables();
    bun_jsc::initialize(false);

    // Touch the high-tier runtime hooks so this staticlib root owns
    // `__BUN_RUNTIME_HOOKS` without depending on Bun's process-owned CLI root.
    let _ = bun_runtime::jsc_hooks::runtime_state();

    let opts = InitOptions {
        is_main_thread: false,
        ..Default::default()
    };

    match VirtualMachine::init(opts) {
        Ok(vm) => {
            // SAFETY: `VirtualMachine::init` returned a fresh VM pointer for
            // this probe invocation. The closure runs before teardown and does
            // not store the mutable reference beyond this stack frame.
            let status = run(unsafe { &mut *vm });
            // SAFETY: `VirtualMachine::init` returned a fresh VM pointer owned
            // by this probe. No other Rust reference exists, and this path
            // immediately tears it down before returning to the C driver.
            unsafe { (&mut *vm).destroy() };
            status
        }
        Err(_) => 1,
    }
}

fn run_sync_host_call_probe(vm: &mut VirtualMachine) -> i32 {
    HOST_CALL_COUNT.store(0, Ordering::SeqCst);
    HOST_CALL_PAYLOAD.store(0, Ordering::SeqCst);
    HOST_CALL_RETURNED.store(0, Ordering::SeqCst);

    let global = vm.global();
    let _lock = vm.jsc_vm().get_api_lock();

    global.to_js_value().put(
        global,
        b"__nimbusHostCall",
        JSFunction::create(
            global,
            "__nimbusHostCall",
            __jsc_host_nimbus_bun_embed_sync_host_call,
            1,
            Default::default(),
        ),
    );

    let mut exception = JSValue::ZERO;
    let source = b"globalThis.__nimbusHostCall(41)";
    let filename = b"nimbus-bun-embed-probe-sync-host-call.js";
    // SAFETY: `global` is the live VM global; source and filename byte slices
    // are valid for the duration of this synchronous program evaluation; and
    // `exception` is a unique writable out-parameter.
    let result = unsafe {
        Bun__REPL__evaluate(
            core::ptr::from_ref(global),
            source.as_ptr(),
            source.len(),
            filename.as_ptr(),
            filename.len(),
            &mut exception,
        )
    };

    if !exception.is_empty() || global.has_exception() {
        return 2;
    }
    if !result.is_number() || result.as_number() as i32 != 42 {
        return 6;
    }

    if HOST_CALL_COUNT.load(Ordering::SeqCst) != 1 {
        return 3;
    }
    if HOST_CALL_PAYLOAD.load(Ordering::SeqCst) != 41 {
        return 4;
    }
    if HOST_CALL_RETURNED.load(Ordering::SeqCst) != 42 {
        return 5;
    }

    0
}

#[bun_jsc::host_fn]
pub fn nimbus_bun_embed_sync_host_call(
    _global: &JSGlobalObject,
    frame: &CallFrame,
) -> JsResult<JSValue> {
    let payload = frame.argument(0);
    let payload = if payload.is_number() {
        payload.as_number() as i32
    } else {
        -1
    };
    let returned = payload + 1;

    HOST_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
    HOST_CALL_PAYLOAD.store(payload, Ordering::SeqCst);
    HOST_CALL_RETURNED.store(returned, Ordering::SeqCst);

    Ok(JSValue::js_number_from_int32(returned))
}
