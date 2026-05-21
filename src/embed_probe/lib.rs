//! Non-CLI Bun/JSC embed proof root.
//!
//! This crate deliberately avoids `bun_bin`: no process `main`, no global
//! allocator override, no crash/signal/stdio setup, no CLI dispatch, and no
//! process exit. The native build graph links this archive with Bun's normal
//! C++/WebKit/JSC objects through the opt-in `check-bun-embed-probe` target.

use core::sync::atomic::{AtomicI32, AtomicPtr, Ordering};

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

fn construct_vm_and_run(run: impl FnOnce(&mut VirtualMachine) -> i32) -> i32 {
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
