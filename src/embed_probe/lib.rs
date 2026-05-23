//! Non-CLI Bun/JSC embed proof root.
//!
//! This crate deliberately avoids `bun_bin`: no process `main`, no global
//! allocator override, no crash/signal/stdio setup, no CLI dispatch, and no
//! process exit. The native build graph links this archive with Bun's normal
//! C++/WebKit/JSC objects through the opt-in `check-bun-embed-probe` target.

use core::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, Ordering};
use std::{sync::Arc, thread};

use bun_jsc::virtual_machine::{InitOptions, VirtualMachine};
use bun_jsc::{
    AnyPromise, CallFrame, GlobalRef, JSFunction, JSGlobalObject, JSPromise, JSValue, JsResult,
    PromiseStatus, VM,
};

// Force-link the shared C ABI bridge that still owns process-neutral symbols
// required by Bun's native object graph.
use bun_link_bridge as _;

static HOST_CALL_COUNT: AtomicI32 = AtomicI32::new(0);
static HOST_CALL_PAYLOAD: AtomicI32 = AtomicI32::new(0);
static HOST_CALL_RETURNED: AtomicI32 = AtomicI32::new(0);
static ASYNC_HOST_CALL_COUNT: AtomicI32 = AtomicI32::new(0);
static ASYNC_TASK_RUN_COUNT: AtomicI32 = AtomicI32::new(0);
static ASYNC_HOST_CALL_PAYLOAD: AtomicI32 = AtomicI32::new(0);
static ASYNC_TASK_RETURNED: AtomicI32 = AtomicI32::new(0);
static ASYNC_PROMISE: AtomicPtr<JSPromise> = AtomicPtr::new(core::ptr::null_mut());
static SAFETY_VTABLES_REGISTERED: AtomicBool = AtomicBool::new(false);
static SPIN_ENTERED_ACK: AtomicBool = AtomicBool::new(false);

const GENERATED_NIMBUS_PROGRAM_BUNDLE: &[u8] = include_bytes!("nimbus_generated_program_bundle.js");

unsafe extern "C" {
    safe fn JSC__VM__getAPILock(vm: &VM);
    safe fn JSC__VM__releaseAPILock(vm: &VM);

    fn Bun__REPL__evaluate(
        global_object: *const JSGlobalObject,
        source_ptr: *const u8,
        source_len: usize,
        filename_ptr: *const u8,
        filename_len: usize,
        exception: *mut JSValue,
    ) -> JSValue;

    fn Bun__embedderApplyNativePermissionDenyProfileForTesting(global_object: *mut JSGlobalObject);
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_construct_and_destroy_vm() -> i32 {
    construct_vm_and_run(|_| 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_sync_host_call() -> i32 {
    construct_vm_and_run(run_sync_host_call_probe)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_async_host_call() -> i32 {
    construct_vm_and_run(run_async_host_call_probe)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_program_bundle_host_calls() -> i32 {
    construct_vm_and_run(run_program_bundle_host_call_probe)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_timeout_and_cancel() -> i32 {
    construct_vm_and_run(run_timeout_and_cancel_probe)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_permission_surface_inventory() -> i32 {
    construct_vm_and_run(run_permission_surface_inventory_probe)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_memory_behavior() -> i32 {
    construct_vm_and_run(run_memory_behavior_probe)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_package_module_policy() -> i32 {
    construct_vm_and_run(run_package_module_policy_probe)
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_lifecycle_reuse_stress() -> i32 {
    for _ in 0..LIFECYCLE_FRESH_VM_ITERATIONS {
        let status = construct_vm_and_run(run_program_bundle_host_call_probe);
        if status != 0 {
            return status;
        }
    }
    construct_vm_and_run(run_lifecycle_reuse_stress_probe)
}

fn construct_vm_and_run(run: impl FnOnce(&mut VirtualMachine) -> i32) -> i32 {
    bun_core::output::init_test();
    if !SAFETY_VTABLES_REGISTERED.swap(true, Ordering::SeqCst) {
        bun_runtime::allocators::register_safety_vtables();
    }
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

struct EmbedderResolutionDenyGuard;

impl EmbedderResolutionDenyGuard {
    fn new() -> Self {
        bun_jsc::ModuleLoader::set_embedder_deny_all_module_resolution_for_testing(true);
        Self
    }
}

impl Drop for EmbedderResolutionDenyGuard {
    fn drop(&mut self) {
        bun_jsc::ModuleLoader::set_embedder_deny_all_module_resolution_for_testing(false);
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

    let source = b"globalThis.__nimbusHostCall(41)";
    let filename = b"nimbus-bun-embed-probe-sync-host-call.js";
    let result = match evaluate_program(global, source, filename, 2) {
        Ok(result) => result,
        Err(status) => return status,
    };
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

fn run_async_host_call_probe(vm: &mut VirtualMachine) -> i32 {
    ASYNC_HOST_CALL_COUNT.store(0, Ordering::SeqCst);
    ASYNC_TASK_RUN_COUNT.store(0, Ordering::SeqCst);
    ASYNC_HOST_CALL_PAYLOAD.store(0, Ordering::SeqCst);
    ASYNC_TASK_RETURNED.store(0, Ordering::SeqCst);
    ASYNC_PROMISE.store(core::ptr::null_mut(), Ordering::SeqCst);

    vm.event_loop_mut().ensure_waker();

    let global = vm.global();
    {
        let _lock = vm.jsc_vm().get_api_lock();

        global.to_js_value().put(
            global,
            b"__nimbusAsyncHostCall",
            JSFunction::create(
                global,
                "__nimbusAsyncHostCall",
                __jsc_host_nimbus_bun_embed_async_host_call,
                1,
                Default::default(),
            ),
        );

        let source = br#"
globalThis.__nimbusAsyncObserved = -1;
globalThis.__nimbusAsyncHostCall(41).then((value) => {
  globalThis.__nimbusAsyncObserved = value;
});
"#;
        let filename = b"nimbus-bun-embed-probe-async-host-call.js";
        if let Err(status) = evaluate_program(global, source, filename, 10) {
            return status;
        }
    }

    if ASYNC_HOST_CALL_COUNT.load(Ordering::SeqCst) != 1 {
        return 11;
    }
    if ASYNC_HOST_CALL_PAYLOAD.load(Ordering::SeqCst) != 41 {
        return 12;
    }
    if ASYNC_TASK_RUN_COUNT.load(Ordering::SeqCst) != 0 {
        return 13;
    }

    let promise = ASYNC_PROMISE.load(Ordering::SeqCst);
    if promise.is_null() {
        return 14;
    }

    {
        let _lock = ProbeApiLock::new(vm.jsc_vm());
        vm.wait_for_promise(AnyPromise::Normal(promise));

        if ASYNC_TASK_RUN_COUNT.load(Ordering::SeqCst) != 1 {
            return 15;
        }
        if ASYNC_TASK_RETURNED.load(Ordering::SeqCst) != 42 {
            return 16;
        }

        let promise = JSPromise::opaque_mut(promise);
        if promise.status() != PromiseStatus::Fulfilled {
            return 17;
        }
        let promise_result = promise.result(vm.jsc_vm());
        if !promise_result.is_number() || promise_result.as_number() as i32 != 42 {
            return 18;
        }

        let observed = match evaluate_program(
            global,
            b"globalThis.__nimbusAsyncObserved",
            b"nimbus-bun-embed-probe-async-observed.js",
            19,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !observed.is_number() || observed.as_number() as i32 != 42 {
            return 20;
        }
    }

    0
}

fn run_program_bundle_host_call_probe(vm: &mut VirtualMachine) -> i32 {
    HOST_CALL_COUNT.store(0, Ordering::SeqCst);
    HOST_CALL_PAYLOAD.store(0, Ordering::SeqCst);
    HOST_CALL_RETURNED.store(0, Ordering::SeqCst);
    ASYNC_HOST_CALL_COUNT.store(0, Ordering::SeqCst);
    ASYNC_TASK_RUN_COUNT.store(0, Ordering::SeqCst);
    ASYNC_HOST_CALL_PAYLOAD.store(0, Ordering::SeqCst);
    ASYNC_TASK_RETURNED.store(0, Ordering::SeqCst);
    ASYNC_PROMISE.store(core::ptr::null_mut(), Ordering::SeqCst);

    vm.event_loop_mut().ensure_waker();

    let global = vm.global();
    let async_invocation_promise = {
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
        global.to_js_value().put(
            global,
            b"__nimbusAsyncHostCall",
            JSFunction::create(
                global,
                "__nimbusAsyncHostCall",
                __jsc_host_nimbus_bun_embed_async_host_call,
                1,
                Default::default(),
            ),
        );

        let context_source = br#"
globalThis.__nimbusGeneratedProgramState = {
  dbObserved: -1,
  scheduleDelay: -1,
  scheduleName: "",
  scheduleVisibility: "",
  scheduleKind: "",
  scheduleBody: "",
  scheduleHostResult: -1,
};
globalThis.__nimbusCreateContext = () => ({
  db: {
    insert: async (_table, document) => {
      const observed = await globalThis.__nimbusAsyncHostCall(
        document && document.body === "hello" ? 41 : -1,
      );
      globalThis.__nimbusGeneratedProgramState.dbObserved = observed;
      return document && document.body === "hello" ? "message-id" : "scheduled-id";
    },
  },
  scheduler: {
    runAfter: async (delayMs, mutationRef, args) => {
      const state = globalThis.__nimbusGeneratedProgramState;
      state.scheduleDelay = delayMs;
      state.scheduleName = mutationRef.name;
      state.scheduleVisibility = mutationRef.visibility;
      state.scheduleKind = mutationRef.kind;
      state.scheduleBody = args.body;
      state.scheduleHostResult = globalThis.__nimbusHostCall(41);
      return "job-id";
    },
  },
});
1
"#;
        let context_loaded = match evaluate_program(
            global,
            context_source,
            b"nimbus-bun-embed-probe-generated-program-context.js",
            21,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !context_loaded.is_number() || context_loaded.as_number() as i32 != 1 {
            return 22;
        }

        if let Err(status) = evaluate_program(
            global,
            GENERATED_NIMBUS_PROGRAM_BUNDLE,
            b"nimbus-bun-embed-probe-generated-program-bundle.js",
            23,
        ) {
            return status;
        }

        let async_result = match evaluate_program(
            global,
            br#"
globalThis.__nimbusGeneratedProgramPromise = globalThis.__nimbusInvoke({
  kind: "mutation",
  function_name: "messages:sendAndSchedule",
  args: { body: "hello" },
}).then((response) => {
  globalThis.__nimbusGeneratedProgramResponse = response;
  return response.status === "ok" && response.value === "message-id" ? 42 : -1;
});
globalThis.__nimbusGeneratedProgramPromise
"#,
            b"nimbus-bun-embed-probe-generated-program-invoke.js",
            24,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        match async_result.as_promise() {
            Some(promise) => promise,
            None => return 25,
        }
    };

    if ASYNC_HOST_CALL_COUNT.load(Ordering::SeqCst) != 1 {
        return 26;
    }
    if ASYNC_HOST_CALL_PAYLOAD.load(Ordering::SeqCst) != 41 {
        return 28;
    }
    if ASYNC_TASK_RUN_COUNT.load(Ordering::SeqCst) != 0 {
        return 29;
    }
    if HOST_CALL_COUNT.load(Ordering::SeqCst) != 0 {
        return 30;
    }

    let host_promise = ASYNC_PROMISE.load(Ordering::SeqCst);
    if host_promise.is_null() {
        return 31;
    }

    {
        let _lock = ProbeApiLock::new(vm.jsc_vm());
        vm.wait_for_promise(AnyPromise::Normal(async_invocation_promise));

        if ASYNC_TASK_RUN_COUNT.load(Ordering::SeqCst) != 1 {
            return 32;
        }
        if ASYNC_TASK_RETURNED.load(Ordering::SeqCst) != 42 {
            return 33;
        }
        if HOST_CALL_COUNT.load(Ordering::SeqCst) != 1 {
            return 34;
        }
        if HOST_CALL_PAYLOAD.load(Ordering::SeqCst) != 41 {
            return 35;
        }
        if HOST_CALL_RETURNED.load(Ordering::SeqCst) != 42 {
            return 36;
        }

        let host_promise = JSPromise::opaque_mut(host_promise);
        if host_promise.status() != PromiseStatus::Fulfilled {
            return 37;
        }
        let invocation_promise = JSPromise::opaque_mut(async_invocation_promise);
        if invocation_promise.status() != PromiseStatus::Fulfilled {
            return 38;
        }
        let invocation_result = invocation_promise.result(vm.jsc_vm());
        if !invocation_result.is_number() || invocation_result.as_number() as i32 != 42 {
            return 39;
        }

        let state_check = match evaluate_program(
            global,
            br#"
(() => {
  const state = globalThis.__nimbusGeneratedProgramState;
  const response = globalThis.__nimbusGeneratedProgramResponse;
  return state.dbObserved === 42
    && state.scheduleDelay === 1000
    && state.scheduleName === "messages:sendInternal"
    && state.scheduleVisibility === "internal"
    && state.scheduleKind === "mutation"
    && state.scheduleBody === "hello later"
    && state.scheduleHostResult === 42
    && response.status === "ok"
    && response.value === "message-id"
      ? 42
      : -1;
})()
"#,
            b"nimbus-bun-embed-probe-generated-program-state.js",
            40,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !state_check.is_number() || state_check.as_number() as i32 != 42 {
            return 41;
        }
    }

    0
}

fn run_timeout_and_cancel_probe(vm: &mut VirtualMachine) -> i32 {
    vm.event_loop_mut().ensure_waker();

    let global = vm.global();
    {
        let _lock = vm.jsc_vm().get_api_lock();
        // Cross-thread `notify_need_termination` expects JSC's termination
        // exception to have been materialized by the owning thread first.
        global.request_termination();
        global.clear_termination_exception();
        vm.jsc_vm().clear_has_termination_request();

        let context_loaded = match evaluate_program(
            global,
            b"globalThis.__nimbusCreateContext = () => ({}); 1",
            b"nimbus-bun-embed-probe-timeout-context.js",
            62,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !context_loaded.is_number() || context_loaded.as_number() as i32 != 1 {
            return 63;
        }

        if let Err(status) = evaluate_program(
            global,
            GENERATED_NIMBUS_PROGRAM_BUNDLE,
            b"nimbus-bun-embed-probe-timeout-program-bundle.js",
            42,
        ) {
            return status;
        }
    }

    if let Err(status) = evaluate_generated_spin_with_deadline_timeout(vm, global) {
        return status;
    }
    if let Err(status) = evaluate_recovery_script(vm, global, 50, 51) {
        return status;
    }

    if let Err(status) = evaluate_generated_spin_with_external_cancel(vm, global) {
        return status;
    }
    if let Err(status) = evaluate_recovery_script(vm, global, 60, 61) {
        return status;
    }

    0
}

const PERMISSION_ABSENT_BY_DEFAULT: i32 = 1;
const PERMISSION_DENIED_BY_DEFAULT: i32 = 2;
const PERMISSION_POLICY_HOOK_AVAILABLE: i32 = 3;
const PERMISSION_POLICY_HOOK_MISSING: i32 = 4;
const PERMISSION_UNSAFE_BYPASS: i32 = 5;

struct PermissionSurfaceProbe {
    name: &'static str,
    source: &'static [u8],
}

const PERMISSION_SURFACE_PROBES: &[PermissionSurfaceProbe] = &[
    PermissionSurfaceProbe {
        name: "Bun global",
        source: br#"
typeof globalThis.Bun === "undefined"
  ? 1
  : (globalThis.Bun.__nimbusNativePermissionProfile === "deny" ? 3 : 5)
"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.file",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.file)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.write",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.write)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.spawn",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.spawn)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.spawnSync",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.spawnSync)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.serve",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.serve)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.listen",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.listen)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.connect",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.connect)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.plugin",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.plugin)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.FFI",
        source: br#"globalThis.__nimbusPermissionProbeProfileObject(globalThis.Bun?.FFI)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.dlopen",
        source: br#"typeof globalThis.Bun?.dlopen === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.FFI.dlopen",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.FFI?.dlopen)"#,
    },
    PermissionSurfaceProbe {
        name: "Bun.env",
        source: br#"typeof globalThis.Bun?.env === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "process",
        source: br#"
typeof globalThis.process === "undefined"
  ? 1
  : (globalThis.process.__nimbusNativePermissionProfile === "deny" ? 3 : 5)
"#,
    },
    PermissionSurfaceProbe {
        name: "process.env",
        source: br#"typeof globalThis.process?.env === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "Node builtin modules via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "node:fs via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "fs via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "node:child_process via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "node:worker_threads via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "node:net via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "node:dgram via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "node:ffi via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "native addon via require",
        source: br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
    },
    PermissionSurfaceProbe {
        name: "fetch",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.fetch)"#,
    },
    PermissionSurfaceProbe {
        name: "WebSocket",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.WebSocket)"#,
    },
    PermissionSurfaceProbe {
        name: "setTimeout",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.setTimeout)"#,
    },
    PermissionSurfaceProbe {
        name: "Worker",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.Worker)"#,
    },
    PermissionSurfaceProbe {
        name: "new Function",
        source: br#"globalThis.__nimbusPermissionProbeDynamicCode(() => new Function("return 1"))"#,
    },
    PermissionSurfaceProbe {
        name: "Function constructor escape",
        source: br#"globalThis.__nimbusPermissionProbeDynamicCode(() => (() => {}).constructor("return 1"))"#,
    },
    PermissionSurfaceProbe {
        name: "eval",
        source: br#"globalThis.__nimbusPermissionProbeDynamicCode(() => globalThis.eval("1"))"#,
    },
    PermissionSurfaceProbe {
        name: "dynamic import syntax",
        source: br#"globalThis.__nimbusPermissionProbeDynamicCode(() => new Function("return import('node:fs')"))"#,
    },
    PermissionSurfaceProbe {
        name: "Nimbus host hooks and generated wrapper",
        source: br#"
typeof globalThis.__nimbusHostCall === "function"
  && typeof globalThis.__nimbusAsyncHostCall === "function"
  && typeof globalThis.__nimbusInvoke === "function"
    ? 3
    : 4
"#,
    },
];

fn run_permission_surface_inventory_probe(vm: &mut VirtualMachine) -> i32 {
    vm.event_loop_mut().ensure_waker();

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
    global.to_js_value().put(
        global,
        b"__nimbusAsyncHostCall",
        JSFunction::create(
            global,
            "__nimbusAsyncHostCall",
            __jsc_host_nimbus_bun_embed_async_host_call,
            1,
            Default::default(),
        ),
    );

    let context_loaded = match evaluate_program(
        global,
        br#"
globalThis.__nimbusCreateContext = () => ({});
1
"#,
        b"nimbus-bun-embed-probe-permission-context.js",
        100,
    ) {
        Ok(result) => result,
        Err(status) => return status,
    };
    if !context_loaded.is_number() || context_loaded.as_number() as i32 != 1 {
        return 101;
    }

    if let Err(status) = evaluate_program(
        global,
        GENERATED_NIMBUS_PROGRAM_BUNDLE,
        b"nimbus-bun-embed-probe-permission-generated-program-bundle.js",
        102,
    ) {
        return status;
    }

    // SAFETY: this proof VM is single-threaded under the JSC API lock. The
    // native helper only mutates the current global object's embedder profile.
    unsafe {
        Bun__embedderApplyNativePermissionDenyProfileForTesting(
            global as *const JSGlobalObject as *mut JSGlobalObject,
        );
    }

    let helper_loaded = match evaluate_program(
        global,
        br#"
globalThis.__nimbusPermissionDenied = (error) => {
  const message = String(error && error.message || error);
  return message.includes("Bun embedder denied native capability")
    || message.includes("Code generation from strings disallowed by Bun embedder profile");
};
globalThis.__nimbusPermissionProbeFunction = (value) => {
  if (typeof value === "undefined") return 1;
  if (typeof value !== "function") return 4;
  if (value.__nimbusDeniedNativeCapability !== true) return 5;
  try {
    value();
    return 4;
  } catch (error) {
    return globalThis.__nimbusPermissionDenied(error) ? 2 : 4;
  }
};
globalThis.__nimbusPermissionProbeProfileObject = (value) => {
  if (typeof value === "undefined") return 1;
  return value && value.__nimbusNativePermissionProfile === "deny" ? 3 : 5;
};
globalThis.__nimbusPermissionProbeDynamicCode = (callback) => {
  try {
    callback();
    return 5;
  } catch (error) {
    return globalThis.__nimbusPermissionDenied(error) ? 2 : 4;
  }
};
1
"#,
        b"nimbus-bun-embed-probe-permission-lockdown-helpers.js",
        103,
    ) {
        Ok(result) => result,
        Err(status) => return status,
    };
    if !helper_loaded.is_number() || helper_loaded.as_number() as i32 != 1 {
        return 104;
    }

    eprintln!("nimbus bun embed permission surface inventory:");
    for (index, probe) in PERMISSION_SURFACE_PROBES.iter().enumerate() {
        let exception_status = 110 + (index as i32 * 2);
        let mismatch_status = exception_status + 1;
        let result = match evaluate_program(
            global,
            probe.source,
            b"nimbus-bun-embed-probe-permission-surface.js",
            exception_status,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !result.is_number() {
            return mismatch_status;
        }

        let classification = result.as_number() as i32;
        if !is_known_permission_classification(classification) {
            return mismatch_status;
        }
        if classification == PERMISSION_POLICY_HOOK_MISSING
            || classification == PERMISSION_UNSAFE_BYPASS
        {
            return mismatch_status;
        }

        eprintln!(
            "  {}: {}",
            probe.name,
            permission_classification_name(classification)
        );
    }

    0
}

const MEMORY_BEHAVIOR_INVOCATION_COUNT: i32 = 16;

fn run_memory_behavior_probe(vm: &mut VirtualMachine) -> i32 {
    HOST_CALL_COUNT.store(0, Ordering::SeqCst);
    HOST_CALL_PAYLOAD.store(0, Ordering::SeqCst);
    HOST_CALL_RETURNED.store(0, Ordering::SeqCst);
    ASYNC_HOST_CALL_COUNT.store(0, Ordering::SeqCst);
    ASYNC_TASK_RUN_COUNT.store(0, Ordering::SeqCst);
    ASYNC_HOST_CALL_PAYLOAD.store(0, Ordering::SeqCst);
    ASYNC_TASK_RETURNED.store(0, Ordering::SeqCst);
    ASYNC_PROMISE.store(core::ptr::null_mut(), Ordering::SeqCst);

    vm.event_loop_mut().ensure_waker();

    let global = vm.global();
    {
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
        global.to_js_value().put(
            global,
            b"__nimbusAsyncHostCall",
            JSFunction::create(
                global,
                "__nimbusAsyncHostCall",
                __jsc_host_nimbus_bun_embed_async_host_call,
                1,
                Default::default(),
            ),
        );

        let context_loaded = match evaluate_program(
            global,
            br#"
globalThis.__nimbusMemoryProbeState = {
  dbObserved: -1,
  insertCount: 0,
  scheduleCount: 0,
  scheduleHostResult: -1,
};
globalThis.__nimbusCreateContext = () => ({
  db: {
    insert: async (_table, document) => {
      const observed = await globalThis.__nimbusAsyncHostCall(41);
      const state = globalThis.__nimbusMemoryProbeState;
      state.dbObserved = observed;
      state.insertCount += 1;
      return document && document.body
        ? `message-id-${state.insertCount}`
        : "message-id-missing-body";
    },
  },
  scheduler: {
    runAfter: async () => {
      const state = globalThis.__nimbusMemoryProbeState;
      state.scheduleCount += 1;
      state.scheduleHostResult = globalThis.__nimbusHostCall(41);
      return `job-id-${state.scheduleCount}`;
    },
  },
});
1
"#,
            b"nimbus-bun-embed-probe-memory-context.js",
            180,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !context_loaded.is_number() || context_loaded.as_number() as i32 != 1 {
            return 181;
        }

        if let Err(status) = evaluate_program(
            global,
            GENERATED_NIMBUS_PROGRAM_BUNDLE,
            b"nimbus-bun-embed-probe-memory-generated-program-bundle.js",
            182,
        ) {
            return status;
        }
    }

    let heap_after_setup_gc = vm.garbage_collect(true);
    let heap_before_load = vm.jsc_vm().heap_size();

    let invocation_promise = {
        let _lock = vm.jsc_vm().get_api_lock();
        let result = match evaluate_program(
            global,
            br#"
globalThis.__nimbusMemoryRetained = [];
globalThis.__nimbusMemoryProbe = async () => {
  for (let i = 0; i < 16; i += 1) {
    const payload = "x".repeat(64 * 1024) + ":" + i;
    const cells = Array.from({ length: 2048 }, (_value, j) => ({
      i,
      j,
      payload,
      marker: `cell-${i}-${j}`,
    }));
    globalThis.__nimbusMemoryRetained.push({ payload, cells });
    const response = await globalThis.__nimbusInvoke({
      kind: "mutation",
      function_name: "messages:sendAndSchedule",
      args: { body: payload.slice(0, 64) },
    });
    if (response.status !== "ok") {
      return -1;
    }
  }
  return globalThis.__nimbusMemoryRetained.length;
};
globalThis.__nimbusMemoryProbe()
"#,
            b"nimbus-bun-embed-probe-memory-invocations.js",
            183,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        match result.as_promise() {
            Some(promise) => promise,
            None => return 184,
        }
    };

    {
        let _lock = ProbeApiLock::new(vm.jsc_vm());
        vm.wait_for_promise(AnyPromise::Normal(invocation_promise));

        let promise = JSPromise::opaque_mut(invocation_promise);
        if promise.status() != PromiseStatus::Fulfilled {
            return 185;
        }
        let result = promise.result(vm.jsc_vm());
        if !result.is_number() || result.as_number() as i32 != MEMORY_BEHAVIOR_INVOCATION_COUNT {
            return 186;
        }
    }

    if ASYNC_HOST_CALL_COUNT.load(Ordering::SeqCst) != MEMORY_BEHAVIOR_INVOCATION_COUNT {
        return 187;
    }
    if ASYNC_TASK_RUN_COUNT.load(Ordering::SeqCst) != MEMORY_BEHAVIOR_INVOCATION_COUNT {
        return 188;
    }
    if HOST_CALL_COUNT.load(Ordering::SeqCst) != MEMORY_BEHAVIOR_INVOCATION_COUNT {
        return 189;
    }

    let heap_after_load = vm.jsc_vm().heap_size();
    let heap_retained_after_gc = vm.garbage_collect(true);

    {
        let _lock = vm.jsc_vm().get_api_lock();
        let released = match evaluate_program(
            global,
            br#"
globalThis.__nimbusMemoryRetained = null;
globalThis.__nimbusMemoryProbe = undefined;
1
"#,
            b"nimbus-bun-embed-probe-memory-release.js",
            190,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !released.is_number() || released.as_number() as i32 != 1 {
            return 191;
        }
    }

    let heap_after_release_gc = vm.garbage_collect(true);
    let heap_after_shrink = {
        let _lock = vm.jsc_vm().get_api_lock();
        vm.jsc_vm().shrink_footprint();
        vm.jsc_vm().heap_size()
    };

    if heap_after_load <= heap_before_load {
        return 192;
    }

    eprintln!("nimbus bun embed memory behavior:");
    eprintln!("  invocation_count: {MEMORY_BEHAVIOR_INVOCATION_COUNT}");
    eprintln!("  heap_after_setup_gc_bytes: {heap_after_setup_gc}");
    eprintln!("  heap_before_load_bytes: {heap_before_load}");
    eprintln!("  heap_after_load_bytes: {heap_after_load}");
    eprintln!("  heap_retained_after_gc_bytes: {heap_retained_after_gc}");
    eprintln!("  heap_after_release_gc_bytes: {heap_after_release_gc}");
    eprintln!("  heap_after_shrink_bytes: {heap_after_shrink}");
    eprintln!(
        "  observed_load_growth_bytes: {}",
        heap_after_load.saturating_sub(heap_before_load)
    );
    eprintln!(
        "  observed_gc_retained_growth_bytes: {}",
        heap_retained_after_gc.saturating_sub(heap_after_setup_gc)
    );
    eprintln!(
        "  observed_release_drop_bytes: {}",
        heap_retained_after_gc.saturating_sub(heap_after_release_gc)
    );
    eprintln!("  hard_heap_limit: not_observed");
    eprintln!("  pressure_signal: vm.heap_size_and_sync_gc");
    eprintln!("  safe_first_policy: fresh_vm_or_discard_on_pressure");

    0
}

fn run_package_module_policy_probe(vm: &mut VirtualMachine) -> i32 {
    vm.event_loop_mut().ensure_waker();

    let global = vm.global();
    {
        let _lock = ProbeApiLock::new(vm.jsc_vm());

        let context_loaded = match evaluate_program(
            global,
            br#"
globalThis.__nimbusCreateContext = () => ({});
1
"#,
            b"nimbus-bun-embed-probe-module-context.js",
            200,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !context_loaded.is_number() || context_loaded.as_number() as i32 != 1 {
            return 201;
        }

        if let Err(status) = evaluate_program(
            global,
            GENERATED_NIMBUS_PROGRAM_BUNDLE,
            b"nimbus-bun-embed-probe-module-generated-program-bundle.js",
            202,
        ) {
            return status;
        }

        let wrapper_loaded = match evaluate_program(
            global,
            br#"typeof globalThis.__nimbusInvoke === "function" ? 1 : 0"#,
            b"nimbus-bun-embed-probe-module-wrapper-loaded.js",
            203,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !wrapper_loaded.is_number() || wrapper_loaded.as_number() as i32 != 1 {
            return 204;
        }

        let static_esm_rejected = evaluate_program(
            global,
            br#"import { readFile } from "node:fs"; 1"#,
            b"nimbus-bun-embed-probe-static-esm-program.js",
            205,
        )
        .is_err();
        if !static_esm_rejected {
            return 206;
        }
    }

    let _resolution_deny_guard = EmbedderResolutionDenyGuard::new();

    let dynamic_import_status = match evaluate_dynamic_import_node_fs(vm, global) {
        Ok(status) => status,
        Err(status) => return status,
    };
    let dynamic_package_status = match evaluate_dynamic_import_status(
        vm,
        global,
        br#"
import("left-pad").then(
  () => 7,
  (error) => String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 6
)
"#,
        b"nimbus-bun-embed-probe-dynamic-import-package.js",
        230,
        231,
        232,
    ) {
        Ok(status) => status,
        Err(status) => return status,
    };
    let plugin_virtual_module_status = match evaluate_dynamic_import_status(
        vm,
        global,
        br#"
import("nimbus-plugin:probe").then(
  () => 7,
  (error) => String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 6
)
"#,
        b"nimbus-bun-embed-probe-dynamic-import-plugin-virtual.js",
        233,
        234,
        235,
    ) {
        Ok(status) => status,
        Err(status) => return status,
    };

    let (
        require_status,
        bun_resolve_status,
        bun_resolve_sync_status,
        native_addon_resolve_sync_status,
        generated_node_builtin_status,
        generated_external_package_status,
    ) = {
        let require_status = {
            let _lock = ProbeApiLock::new(vm.jsc_vm());
            match evaluate_number(
                global,
                br#"typeof globalThis.require === "undefined" ? 1 : 5"#,
                b"nimbus-bun-embed-probe-require-status.js",
                212,
                213,
            ) {
                Ok(status) => status,
                Err(status) => return status,
            }
        };

        let bun_resolve_status = match evaluate_number_or_promise(
            vm,
            global,
            br#"
typeof globalThis.Bun === "undefined"
  ? 1
  : (typeof globalThis.Bun.resolve !== "function"
      ? 1
      : globalThis.Bun.resolve("node:fs", "/tmp/nimbus-bun-embed-probe/source.js").then(
          () => 5,
          (error) => String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 4
        ))
"#,
            b"nimbus-bun-embed-probe-bun-resolve-status.js",
            214,
            215,
            216,
        ) {
            Ok(status) => status,
            Err(status) => return status,
        };

        let bun_resolve_sync_status = {
            let _lock = ProbeApiLock::new(vm.jsc_vm());
            match evaluate_number(
                global,
                br#"
(() => {
  if (typeof globalThis.Bun === "undefined" || typeof globalThis.Bun.resolveSync !== "function") {
    return 1;
  }
  try {
    globalThis.Bun.resolveSync("node:fs", "/tmp/nimbus-bun-embed-probe/source.js");
    return 5;
  } catch (error) {
    return String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 4;
  }
})()
"#,
                b"nimbus-bun-embed-probe-bun-resolve-sync-status.js",
                217,
                218,
            ) {
                Ok(status) => status,
                Err(status) => return status,
            }
        };

        let native_addon_resolve_sync_status = {
            let _lock = ProbeApiLock::new(vm.jsc_vm());
            match evaluate_number(
                global,
                br#"
(() => {
  if (typeof globalThis.Bun === "undefined" || typeof globalThis.Bun.resolveSync !== "function") {
    return 1;
  }
  try {
    globalThis.Bun.resolveSync("./nimbus-native-addon.node", "/tmp/nimbus-bun-embed-probe/source.js");
    return 5;
  } catch (error) {
    return String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 4;
  }
})()
"#,
                b"nimbus-bun-embed-probe-native-addon-resolve-sync-status.js",
                236,
                237,
            ) {
                Ok(status) => status,
                Err(status) => return status,
            }
        };

        let generated_node_builtin_status = {
            let _lock = ProbeApiLock::new(vm.jsc_vm());
            match evaluate_number(
                global,
                br#"
(() => {
  globalThis.__nimbusNodeBuiltinModules = new Map();
  try {
    nodeBuiltinModule("node:fs");
    return 5;
  } catch (error) {
    return String(error && error.message || error).includes("missing generated Node.js builtin binding")
      ? 2
      : 4;
  }
})()
"#,
                b"nimbus-bun-embed-probe-generated-node-builtin-status.js",
                219,
                220,
            ) {
                Ok(status) => status,
                Err(status) => return status,
            }
        };

        let generated_external_package_status = {
            let _lock = ProbeApiLock::new(vm.jsc_vm());
            match evaluate_number(
                global,
                br#"
(() => {
  globalThis.__nimbusNodeExternalPackages = new Map();
  try {
    nodeExternalPackage("left-pad");
    return 5;
  } catch (error) {
    return String(error && error.message || error).includes("missing generated Node.js external package binding")
      ? 2
      : 4;
  }
})()
"#,
                b"nimbus-bun-embed-probe-generated-external-package-status.js",
                221,
                222,
            ) {
                Ok(status) => status,
                Err(status) => return status,
            }
        };

        (
            require_status,
            bun_resolve_status,
            bun_resolve_sync_status,
            native_addon_resolve_sync_status,
            generated_node_builtin_status,
            generated_external_package_status,
        )
    };

    if require_status != 1 {
        return 240;
    }
    if dynamic_import_status != 8 {
        return 241;
    }
    if dynamic_package_status != 8 {
        return 242;
    }
    if plugin_virtual_module_status != 8 {
        return 243;
    }
    if bun_resolve_status != 8 || bun_resolve_sync_status != 8 {
        return 244;
    }
    if native_addon_resolve_sync_status != 8 {
        return 245;
    }
    if generated_node_builtin_status != 2 || generated_external_package_status != 2 {
        return 246;
    }

    eprintln!("nimbus bun embed package/module policy:");
    eprintln!("  artifact_shape: self_contained_program_wrapper");
    eprintln!("  evaluation_format: program_via_Bun__REPL__evaluate");
    eprintln!("  static_esm_import_in_program: rejected");
    eprintln!(
        "  dynamic_import_node_fs: {}",
        module_policy_status_name(dynamic_import_status)
    );
    eprintln!(
        "  dynamic_import_package_root: {}",
        module_policy_status_name(dynamic_package_status)
    );
    eprintln!(
        "  plugin_virtual_module_import: {}",
        module_policy_status_name(plugin_virtual_module_status)
    );
    eprintln!("  require: {}", module_policy_status_name(require_status));
    eprintln!(
        "  Bun.resolve: {}",
        module_policy_status_name(bun_resolve_status)
    );
    eprintln!(
        "  Bun.resolveSync: {}",
        module_policy_status_name(bun_resolve_sync_status)
    );
    eprintln!(
        "  native_addon_resolveSync: {}",
        module_policy_status_name(native_addon_resolve_sync_status)
    );
    eprintln!(
        "  generated_node_builtin_empty_map: {}",
        module_policy_status_name(generated_node_builtin_status)
    );
    eprintln!(
        "  generated_external_package_empty_map: {}",
        module_policy_status_name(generated_external_package_status)
    );
    eprintln!("  selected_next_lane: program_wrapper");
    eprintln!("  resolver_policy_hook: native_embedder_deny_all");
    eprintln!("  required_resolver_api: nimbus_owned_bun_package_resolver");

    0
}

fn evaluate_dynamic_import_node_fs(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
) -> Result<i32, i32> {
    evaluate_dynamic_import_status(
        vm,
        global,
        br#"
import("node:fs").then(
  () => 7,
  (error) => String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 6
)
"#,
        b"nimbus-bun-embed-probe-dynamic-import-node-fs.js",
        207,
        208,
        209,
    )
}

fn evaluate_dynamic_import_status(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
    source: &[u8],
    filename: &[u8],
    exception_status: i32,
    promise_rejected_status: i32,
    mismatch_status: i32,
) -> Result<i32, i32> {
    evaluate_number_or_promise(
        vm,
        global,
        source,
        filename,
        exception_status,
        promise_rejected_status,
        mismatch_status,
    )
}

fn evaluate_number_or_promise(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
    source: &[u8],
    filename: &[u8],
    exception_status: i32,
    promise_rejected_status: i32,
    mismatch_status: i32,
) -> Result<i32, i32> {
    let _lock = ProbeApiLock::new(vm.jsc_vm());
    let mut result = evaluate_program(global, source, filename, exception_status)?;
    if let Some(promise) = result.as_promise() {
        vm.wait_for_promise(AnyPromise::Normal(promise));

        let promise = JSPromise::opaque_mut(promise);
        if promise.status() != PromiseStatus::Fulfilled {
            return Err(promise_rejected_status);
        }
        result = promise.result(vm.jsc_vm());
    }
    if !result.is_number() {
        return Err(mismatch_status);
    }
    Ok(result.as_number() as i32)
}

fn evaluate_number(
    global: &JSGlobalObject,
    source: &[u8],
    filename: &[u8],
    exception_status: i32,
    mismatch_status: i32,
) -> Result<i32, i32> {
    let result = evaluate_program(global, source, filename, exception_status)?;
    if !result.is_number() {
        return Err(mismatch_status);
    }
    Ok(result.as_number() as i32)
}

fn module_policy_status_name(status: i32) -> &'static str {
    match status {
        1 => "absent_by_default",
        2 => "denied_by_generated_wrapper",
        4 => "policy_hook_missing",
        5 => "unsafe_bypass",
        6 => "rejected_by_bun_loader",
        7 => "unsafe_import_fulfilled",
        8 => "denied_by_resolver_policy",
        _ => "unknown",
    }
}

const LIFECYCLE_FRESH_VM_ITERATIONS: usize = 4;
const LIFECYCLE_RETAINED_INVOCATIONS: i32 = 8;
const LIFECYCLE_CANCEL_ITERATIONS: usize = 3;
const CANCELLATION_PROOF_MAX_ACK_SPINS: usize = 1_000_000;

fn run_lifecycle_reuse_stress_probe(vm: &mut VirtualMachine) -> i32 {
    HOST_CALL_COUNT.store(0, Ordering::SeqCst);
    HOST_CALL_PAYLOAD.store(0, Ordering::SeqCst);
    HOST_CALL_RETURNED.store(0, Ordering::SeqCst);
    ASYNC_HOST_CALL_COUNT.store(0, Ordering::SeqCst);
    ASYNC_TASK_RUN_COUNT.store(0, Ordering::SeqCst);
    ASYNC_HOST_CALL_PAYLOAD.store(0, Ordering::SeqCst);
    ASYNC_TASK_RETURNED.store(0, Ordering::SeqCst);
    ASYNC_PROMISE.store(core::ptr::null_mut(), Ordering::SeqCst);

    vm.event_loop_mut().ensure_waker();

    let global = vm.global();
    {
        let _lock = ProbeApiLock::new(vm.jsc_vm());
        global.request_termination();
        global.clear_termination_exception();
        vm.jsc_vm().clear_has_termination_request();

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
        global.to_js_value().put(
            global,
            b"__nimbusAsyncHostCall",
            JSFunction::create(
                global,
                "__nimbusAsyncHostCall",
                __jsc_host_nimbus_bun_embed_async_host_call,
                1,
                Default::default(),
            ),
        );

        let context_loaded = match evaluate_program(
            global,
            br#"
globalThis.__nimbusLifecycleProbeState = {
  dbObserved: -1,
  insertCount: 0,
  scheduleCount: 0,
  scheduleHostResult: -1,
  lastBody: "",
};
globalThis.__nimbusCreateContext = () => ({
  db: {
    insert: async (_table, document) => {
      const observed = await globalThis.__nimbusAsyncHostCall(41);
      const state = globalThis.__nimbusLifecycleProbeState;
      state.dbObserved = observed;
      state.insertCount += 1;
      state.lastBody = document && document.body || "";
      return `message-id-${state.insertCount}`;
    },
  },
  scheduler: {
    runAfter: async () => {
      const state = globalThis.__nimbusLifecycleProbeState;
      state.scheduleCount += 1;
      state.scheduleHostResult = globalThis.__nimbusHostCall(41);
      return `job-id-${state.scheduleCount}`;
    },
  },
});
1
"#,
            b"nimbus-bun-embed-probe-lifecycle-context.js",
            230,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };
        if !context_loaded.is_number() || context_loaded.as_number() as i32 != 1 {
            return 231;
        }

        if let Err(status) = evaluate_program(
            global,
            GENERATED_NIMBUS_PROGRAM_BUNDLE,
            b"nimbus-bun-embed-probe-lifecycle-generated-program-bundle.js",
            232,
        ) {
            return status;
        }
    }

    for iteration in 0..LIFECYCLE_RETAINED_INVOCATIONS {
        if let Err(status) = invoke_lifecycle_generated_mutation(
            vm,
            global,
            &format!("reuse-{iteration}"),
            iteration + 1,
            233,
        ) {
            return status;
        }
    }

    if ASYNC_HOST_CALL_COUNT.load(Ordering::SeqCst) != LIFECYCLE_RETAINED_INVOCATIONS {
        return 236;
    }
    if ASYNC_TASK_RUN_COUNT.load(Ordering::SeqCst) != LIFECYCLE_RETAINED_INVOCATIONS {
        return 237;
    }
    if HOST_CALL_COUNT.load(Ordering::SeqCst) != LIFECYCLE_RETAINED_INVOCATIONS {
        return 238;
    }

    for _ in 0..LIFECYCLE_CANCEL_ITERATIONS {
        if let Err(status) = evaluate_generated_spin_with_external_cancel(vm, global) {
            return status;
        }
        if let Err(status) = evaluate_recovery_script(vm, global, 239, 240) {
            return status;
        }
    }

    if let Err(status) = invoke_lifecycle_generated_mutation(
        vm,
        global,
        "post-cancel",
        LIFECYCLE_RETAINED_INVOCATIONS + 1,
        241,
    ) {
        return status;
    }

    let expected_total = LIFECYCLE_RETAINED_INVOCATIONS + 1;
    if ASYNC_HOST_CALL_COUNT.load(Ordering::SeqCst) != expected_total {
        return 244;
    }
    if ASYNC_TASK_RUN_COUNT.load(Ordering::SeqCst) != expected_total {
        return 245;
    }
    if HOST_CALL_COUNT.load(Ordering::SeqCst) != expected_total {
        return 246;
    }

    let state_check_source = format!(
        r#"
(() => {{
  const state = globalThis.__nimbusLifecycleProbeState;
  return state.insertCount === {expected_total}
    && state.scheduleCount === {expected_total}
    && state.dbObserved === 42
    && state.scheduleHostResult === 42
    && state.lastBody === "post-cancel"
      ? 42
      : -1;
}})()
"#
    );
    let state_check = {
        let _lock = ProbeApiLock::new(vm.jsc_vm());
        match evaluate_program(
            global,
            state_check_source.as_bytes(),
            b"nimbus-bun-embed-probe-lifecycle-state.js",
            242,
        ) {
            Ok(result) => result,
            Err(status) => return status,
        }
    };
    if !state_check.is_number() || state_check.as_number() as i32 != 42 {
        return 243;
    }

    eprintln!("nimbus bun embed lifecycle reuse stress:");
    eprintln!(
        "  fresh_vm_create_invoke_destroy_iterations: {}",
        LIFECYCLE_FRESH_VM_ITERATIONS
    );
    eprintln!("  retained_vm_invocations_before_cancel: {LIFECYCLE_RETAINED_INVOCATIONS}");
    eprintln!(
        "  external_cancel_recovery_iterations: {}",
        LIFECYCLE_CANCEL_ITERATIONS
    );
    eprintln!("  external_cancel_trigger: spin_entered_ack");
    eprintln!("  cancellation_timing_policy: state_ack_not_sleep");
    eprintln!("  retained_vm_post_cancel_invocation: ok");
    eprintln!("  retained_vm_reuse: trusted_generated_wrapper_ok");
    eprintln!("  product_first_policy: fresh_vm_or_discard_until_containment");

    0
}

fn invoke_lifecycle_generated_mutation(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
    body: &str,
    expected_message_id: i32,
    status_base: i32,
) -> Result<(), i32> {
    let body = format!("{body:?}");
    let source = format!(
        r#"
globalThis.__nimbusInvoke({{
  kind: "mutation",
  function_name: "messages:sendAndSchedule",
  args: {{ body: {body} }},
}}).then((response) => {{
  return response.status === "ok" && response.value === "message-id-{expected_message_id}"
    ? 42
    : -1;
}})
"#
    );

    let _lock = ProbeApiLock::new(vm.jsc_vm());
    let result = evaluate_program(
        global,
        source.as_bytes(),
        b"nimbus-bun-embed-probe-lifecycle-invoke.js",
        status_base,
    )?;
    let Some(promise) = result.as_promise() else {
        return Err(status_base + 1);
    };
    vm.wait_for_promise(AnyPromise::Normal(promise));

    let promise = JSPromise::opaque_mut(promise);
    if promise.status() != PromiseStatus::Fulfilled {
        return Err(status_base + 2);
    }
    let result = promise.result(vm.jsc_vm());
    if !result.is_number() || result.as_number() as i32 != 42 {
        return Err(status_base + 3);
    }

    Ok(())
}

fn is_known_permission_classification(classification: i32) -> bool {
    matches!(
        classification,
        PERMISSION_ABSENT_BY_DEFAULT
            | PERMISSION_DENIED_BY_DEFAULT
            | PERMISSION_POLICY_HOOK_AVAILABLE
            | PERMISSION_POLICY_HOOK_MISSING
            | PERMISSION_UNSAFE_BYPASS
    )
}

fn permission_classification_name(classification: i32) -> &'static str {
    match classification {
        PERMISSION_ABSENT_BY_DEFAULT => "absent_by_default",
        PERMISSION_DENIED_BY_DEFAULT => "denied_by_default",
        PERMISSION_POLICY_HOOK_AVAILABLE => "policy_hook_available",
        PERMISSION_POLICY_HOOK_MISSING => "policy_hook_missing",
        PERMISSION_UNSAFE_BYPASS => "unsafe_bypass",
        _ => "unknown",
    }
}

fn evaluate_generated_spin_with_deadline_timeout(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
) -> Result<(), i32> {
    let completed = Arc::new(AtomicBool::new(false));
    let deadline_fired = Arc::new(AtomicBool::new(false));
    let deadline_ack_observed = Arc::new(AtomicBool::new(false));
    let jsc_vm_ptr = core::ptr::from_ref(vm.jsc_vm()) as usize;
    let completed_for_thread = Arc::clone(&completed);
    let deadline_for_thread = Arc::clone(&deadline_fired);
    let deadline_ack_for_thread = Arc::clone(&deadline_ack_observed);
    SPIN_ENTERED_ACK.store(false, Ordering::SeqCst);
    let deadline = thread::spawn(move || {
        bun_core::StackCheck::configure_thread();
        for _ in 0..CANCELLATION_PROOF_MAX_ACK_SPINS {
            if completed_for_thread.load(Ordering::SeqCst) {
                return;
            }
            if SPIN_ENTERED_ACK.load(Ordering::SeqCst) {
                deadline_ack_for_thread.store(true, Ordering::SeqCst);
                break;
            }
            thread::yield_now();
        }
        if !completed_for_thread.load(Ordering::SeqCst) {
            deadline_for_thread.store(true, Ordering::SeqCst);
            // SAFETY: the proof joins this thread before the VM can be torn down.
            let jsc_vm = unsafe { &*(jsc_vm_ptr as *const VM) };
            jsc_vm.notify_need_termination();
        }
    });

    let spin_evaluation = {
        let _lock = vm.jsc_vm().get_api_lock();
        evaluate_generated_spin(
            vm.jsc_vm(),
            global,
            b"nimbus-bun-embed-probe-timeout-spin.js",
            43,
            44,
        )
    };
    let result = spin_evaluation.and_then(|evaluation| {
        if let SpinEvaluation::Promise(promise) = evaluation {
            let _lock = ProbeApiLock::new(vm.jsc_vm());
            vm.wait_for_promise(AnyPromise::Normal(promise));
            if !vm.jsc_vm().has_termination_request() && !global.has_exception() {
                return Err(45);
            }
        }
        Ok(())
    });
    completed.store(true, Ordering::SeqCst);
    if deadline.join().is_err() {
        return Err(46);
    }
    global.clear_termination_exception();

    result?;
    if !deadline_ack_observed.load(Ordering::SeqCst) {
        return Err(47);
    }
    if !deadline_fired.load(Ordering::SeqCst) {
        return Err(48);
    }
    if vm.jsc_vm().has_execution_time_limit() {
        return Err(49);
    }
    if vm.jsc_vm().has_termination_request() {
        return Err(64);
    }
    Ok(())
}

fn evaluate_generated_spin_with_external_cancel(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
) -> Result<(), i32> {
    let completed = Arc::new(AtomicBool::new(false));
    let cancel_fired = Arc::new(AtomicBool::new(false));
    let cancel_ack_observed = Arc::new(AtomicBool::new(false));
    let jsc_vm_ptr = core::ptr::from_ref(vm.jsc_vm()) as usize;
    let completed_for_thread = Arc::clone(&completed);
    let cancel_for_thread = Arc::clone(&cancel_fired);
    let cancel_ack_for_thread = Arc::clone(&cancel_ack_observed);
    SPIN_ENTERED_ACK.store(false, Ordering::SeqCst);
    let canceller = thread::spawn(move || {
        bun_core::StackCheck::configure_thread();
        for _ in 0..CANCELLATION_PROOF_MAX_ACK_SPINS {
            if completed_for_thread.load(Ordering::SeqCst) {
                return;
            }
            if SPIN_ENTERED_ACK.load(Ordering::SeqCst) {
                cancel_ack_for_thread.store(true, Ordering::SeqCst);
                break;
            }
            thread::yield_now();
        }
        if !completed_for_thread.load(Ordering::SeqCst) {
            cancel_for_thread.store(true, Ordering::SeqCst);
            // SAFETY: the proof joins this thread before the VM can be torn down.
            let jsc_vm = unsafe { &*(jsc_vm_ptr as *const VM) };
            jsc_vm.notify_need_termination();
        }
    });

    let spin_evaluation = {
        let _lock = vm.jsc_vm().get_api_lock();
        evaluate_generated_spin(
            vm.jsc_vm(),
            global,
            b"nimbus-bun-embed-probe-external-cancel-spin.js",
            52,
            53,
        )
    };
    let result = spin_evaluation.and_then(|evaluation| {
        if let SpinEvaluation::Promise(promise) = evaluation {
            let _lock = ProbeApiLock::new(vm.jsc_vm());
            vm.wait_for_promise(AnyPromise::Normal(promise));
            if !vm.jsc_vm().has_termination_request() && !global.has_exception() {
                return Err(54);
            }
        }
        Ok(())
    });
    completed.store(true, Ordering::SeqCst);
    if canceller.join().is_err() {
        return Err(55);
    }
    global.clear_termination_exception();

    result?;
    if !cancel_ack_observed.load(Ordering::SeqCst) {
        return Err(56);
    }
    if !cancel_fired.load(Ordering::SeqCst) {
        return Err(57);
    }
    if vm.jsc_vm().has_termination_request() {
        return Err(58);
    }
    if vm.jsc_vm().has_execution_time_limit() {
        return Err(59);
    }
    Ok(())
}

fn evaluate_recovery_script(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
    exception_status: i32,
    mismatch_status: i32,
) -> Result<(), i32> {
    let recovered = {
        let _lock = vm.jsc_vm().get_api_lock();
        evaluate_program(
            global,
            b"40 + 2",
            b"nimbus-bun-embed-probe-timeout-recovery.js",
            exception_status,
        )?
    };
    if !recovered.is_number() || recovered.as_number() as i32 != 42 {
        return Err(mismatch_status);
    }
    Ok(())
}

fn evaluate_program(
    global: &JSGlobalObject,
    source: &[u8],
    filename: &[u8],
    exception_status: i32,
) -> Result<JSValue, i32> {
    let mut exception = JSValue::ZERO;
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
        return Err(exception_status);
    }

    Ok(result)
}

const GENERATED_SPIN_INVOCATION_SOURCE: &[u8] = br#"
globalThis.__nimbusSpinEntered = false;
globalThis.__nimbusInvoke({
  kind: "mutation",
  function_name: "messages:spinForever",
  args: {
    body: {
      trim() {
        globalThis.__nimbusSpinEntered = true;
        if (typeof globalThis.__nimbusAcknowledgeSpinEntered === "function") {
          globalThis.__nimbusAcknowledgeSpinEntered();
        }
        return "hello";
      },
    },
  },
})
"#;

enum SpinEvaluation {
    Promise(*mut JSPromise),
    Terminated,
}

fn evaluate_generated_spin(
    jsc_vm: &VM,
    global: &JSGlobalObject,
    filename: &[u8],
    exception_status: i32,
    promise_status: i32,
) -> Result<SpinEvaluation, i32> {
    global.to_js_value().put(
        global,
        b"__nimbusAcknowledgeSpinEntered",
        JSFunction::create(
            global,
            "__nimbusAcknowledgeSpinEntered",
            __jsc_host_nimbus_bun_embed_spin_entered_ack,
            0,
            Default::default(),
        ),
    );

    let mut exception = JSValue::ZERO;
    // SAFETY: `global` is the live VM global; source and filename byte slices
    // are valid for the duration of this synchronous program evaluation; and
    // `exception` is a unique writable out-parameter.
    let value = unsafe {
        Bun__REPL__evaluate(
            core::ptr::from_ref(global),
            GENERATED_SPIN_INVOCATION_SOURCE.as_ptr(),
            GENERATED_SPIN_INVOCATION_SOURCE.len(),
            filename.as_ptr(),
            filename.len(),
            &mut exception,
        )
    };

    if !exception.is_empty() {
        if is_termination_signal(jsc_vm, exception)
            || jsc_vm.has_termination_request()
            || !global.clear_exception_except_termination()
        {
            return Ok(SpinEvaluation::Terminated);
        }
        global.clear_exception();
        if generated_spin_loop_was_entered(global) {
            return Ok(SpinEvaluation::Terminated);
        }
        return Err(exception_status);
    }
    if global.has_exception() {
        if jsc_vm.has_termination_request() || !global.clear_exception_except_termination() {
            return Ok(SpinEvaluation::Terminated);
        }
        global.clear_exception();
        if generated_spin_loop_was_entered(global) {
            return Ok(SpinEvaluation::Terminated);
        }
        return Err(exception_status);
    }

    match value.as_promise() {
        Some(promise) => Ok(SpinEvaluation::Promise(promise)),
        None => Err(promise_status),
    }
}

fn generated_spin_loop_was_entered(global: &JSGlobalObject) -> bool {
    let mut exception = JSValue::ZERO;
    // SAFETY: `global` is the live VM global; source and filename byte slices
    // are valid for this synchronous read; and `exception` is a unique
    // writable out-parameter.
    let value = unsafe {
        Bun__REPL__evaluate(
            core::ptr::from_ref(global),
            b"globalThis.__nimbusSpinEntered === true ? 1 : 0".as_ptr(),
            b"globalThis.__nimbusSpinEntered === true ? 1 : 0".len(),
            b"nimbus-bun-embed-probe-spin-entered.js".as_ptr(),
            b"nimbus-bun-embed-probe-spin-entered.js".len(),
            &mut exception,
        )
    };
    exception.is_empty()
        && !global.has_exception()
        && value.is_number()
        && value.as_number() as i32 == 1
}

fn is_termination_signal(jsc_vm: &VM, exception: JSValue) -> bool {
    if exception.is_termination_exception() {
        return true;
    }
    let Some(exception) = exception.as_exception(core::ptr::from_ref(jsc_vm).cast_mut()) else {
        return false;
    };
    // SAFETY: `as_exception` proved the JS value is backed by a live JSC
    // Exception cell for this VM.
    jsc_vm.is_termination_exception(unsafe { &*exception })
}

struct ProbeApiLock {
    vm: *const VM,
}

impl ProbeApiLock {
    fn new(vm: &VM) -> Self {
        JSC__VM__getAPILock(vm);
        Self {
            vm: core::ptr::from_ref(vm),
        }
    }
}

impl Drop for ProbeApiLock {
    fn drop(&mut self) {
        // SAFETY: `vm` was captured from the live `VirtualMachine` for this
        // probe and the guard is dropped before VM teardown.
        JSC__VM__releaseAPILock(unsafe { &*self.vm });
    }
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

struct AsyncHostCallTask {
    global: GlobalRef,
    promise: *mut JSPromise,
    returned: i32,
}

impl AsyncHostCallTask {
    fn run(this: *mut Self) -> bun_event_loop::JsResult<()> {
        // SAFETY: `this` was heap-allocated in the async host function and is
        // invoked exactly once by `ManagedTask`.
        let this = unsafe { bun_core::heap::take(this) };
        let global = &*this.global;
        let promise = JSPromise::opaque_mut(this.promise);

        ASYNC_TASK_RUN_COUNT.fetch_add(1, Ordering::SeqCst);
        ASYNC_TASK_RETURNED.store(this.returned, Ordering::SeqCst);

        promise
            .resolve(global, JSValue::js_number_from_int32(this.returned))
            .map_err(Into::into)
    }
}

#[bun_jsc::host_fn]
pub fn nimbus_bun_embed_async_host_call(
    global: &JSGlobalObject,
    frame: &CallFrame,
) -> JsResult<JSValue> {
    let payload = frame.argument(0);
    let payload = if payload.is_number() {
        payload.as_number() as i32
    } else {
        -1
    };
    let returned = payload + 1;

    ASYNC_HOST_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
    ASYNC_HOST_CALL_PAYLOAD.store(payload, Ordering::SeqCst);

    let promise = JSPromise::create(global);
    let promise_ptr = core::ptr::from_mut(promise);
    ASYNC_PROMISE.store(promise_ptr, Ordering::SeqCst);

    let task = bun_core::heap::into_raw(Box::new(AsyncHostCallTask {
        global: GlobalRef::from(global),
        promise: promise_ptr,
        returned,
    }));
    global
        .bun_vm()
        .as_mut()
        .enqueue_task(bun_jsc::ManagedTask::ManagedTask::new(
            task,
            AsyncHostCallTask::run,
        ));

    Ok(promise.to_js())
}

#[bun_jsc::host_fn]
pub fn nimbus_bun_embed_spin_entered_ack(
    _global: &JSGlobalObject,
    _frame: &CallFrame,
) -> JsResult<JSValue> {
    SPIN_ENTERED_ACK.store(true, Ordering::SeqCst);
    Ok(JSValue::js_number_from_int32(1))
}
