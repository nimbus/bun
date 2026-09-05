//! Non-CLI Bun/JSC embed proof root.
//!
//! This crate deliberately avoids `bun_bin`: no process `main`, no global
//! allocator override, no crash/signal/stdio setup, no CLI dispatch, and no
//! process exit. The native build graph links this archive with Bun's normal
//! C++/WebKit/JSC objects through the opt-in `check-bun-embed-probe` target.
//! This root is a size exception because its private ABI, JSC lifetime state,
//! and same-process release probes share one unsafe ownership boundary. Split
//! it only when a child can own its state without widening that boundary.

use core::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, Ordering};
use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    slice, str,
    sync::{Arc, Condvar, Mutex, Once},
    thread,
    time::{Duration, Instant},
};

use bun_core::EncodedSlice;
use bun_jsc::virtual_machine::{InitOptions, VirtualMachine};
use bun_jsc::{
    AnyPromise, CallFrame, EncodedSliceJsc as _, GlobalRef, JSFunction, JSGlobalObject, JSPromise,
    JSValue, JsResult, PromiseStatus, VM,
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
static EMBEDDER_PROCESS_INITIALIZED: Once = Once::new();

fn trace_embed_probe(phase: &'static [u8]) {
    if std::env::var_os("NIMBUS_BUN_EMBED_PROBE_TRACE").is_none() {
        return;
    }

    // Use one direct write so diagnostics do not take Bun's shared output
    // locks and accidentally serialize the concurrent first-VM proof.
    // A failed write is diagnostic loss only; it must not alter the probe.
    unsafe {
        let _ = libc::write(libc::STDERR_FILENO, phase.as_ptr().cast(), phase.len());
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpinEntryWait {
    Entered,
    Completed,
    TimedOut,
}

#[derive(Default)]
struct SpinEntryState {
    entered: bool,
    completed: bool,
}

#[derive(Default)]
struct SpinEntrySignal {
    state: Mutex<SpinEntryState>,
    changed: Condvar,
}

impl SpinEntrySignal {
    fn acknowledge(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.entered = true;
        self.changed.notify_all();
    }

    fn complete(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.completed = true;
        self.changed.notify_all();
    }

    fn wait(&self, timeout: Duration) -> SpinEntryWait {
        let deadline = Instant::now() + timeout;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        loop {
            if state.entered {
                return SpinEntryWait::Entered;
            }
            if state.completed {
                return SpinEntryWait::Completed;
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return SpinEntryWait::TimedOut;
            }
            let (next, timed_out) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|error| error.into_inner());
            state = next;
            if timed_out.timed_out() && !state.entered && !state.completed {
                return SpinEntryWait::TimedOut;
            }
        }
    }
}

thread_local! {
    static CURRENT_SPIN_ENTRY_SIGNAL: RefCell<Option<Arc<SpinEntrySignal>>> = const { RefCell::new(None) };
}

struct SpinEntrySignalGuard {
    previous: Option<Arc<SpinEntrySignal>>,
}

impl SpinEntrySignalGuard {
    fn install(signal: Arc<SpinEntrySignal>) -> Self {
        let previous = CURRENT_SPIN_ENTRY_SIGNAL.with(|slot| slot.replace(Some(signal)));
        Self { previous }
    }
}

impl Drop for SpinEntrySignalGuard {
    fn drop(&mut self) {
        CURRENT_SPIN_ENTRY_SIGNAL.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

struct InitProofGateState {
    arrivals: usize,
    released: bool,
    cancelled: bool,
}

struct InitProofGate {
    expected: usize,
    state: Mutex<InitProofGateState>,
    changed: Condvar,
}

impl InitProofGate {
    fn new(expected: usize) -> Self {
        Self {
            expected,
            state: Mutex::new(InitProofGateState {
                arrivals: 0,
                released: false,
                cancelled: false,
            }),
            changed: Condvar::new(),
        }
    }

    fn cancel(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.cancelled = true;
        state.released = true;
        self.changed.notify_all();
    }

    fn arrive_and_wait(&self) -> bool {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.cancelled {
            return false;
        }
        state.arrivals += 1;
        if state.arrivals == self.expected {
            state.released = true;
            self.changed.notify_all();
        }

        while !state.released {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                state.cancelled = true;
                state.released = true;
                self.changed.notify_all();
                return false;
            }
            let (next, timeout) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|error| error.into_inner());
            state = next;
            if timeout.timed_out() && !state.released {
                state.cancelled = true;
                state.released = true;
                self.changed.notify_all();
                return false;
            }
        }
        !state.cancelled
    }
}

pub type NimbusBunEmbedHostCallJsonFn = unsafe extern "C" fn(
    context: *mut c_void,
    request_ptr: *const u8,
    request_len: usize,
    output_ptr: *mut u8,
    output_cap: usize,
    output_len: *mut usize,
) -> i32;
// ABI 3 calls the host once with a non-null request. If that call returns 307,
// the host retains the completed response and a null, zero-length request takes
// it without repeating the operation.
pub type NimbusBunEmbedIsCancelledFn = unsafe extern "C" fn(context: *mut c_void) -> bool;

#[derive(Clone, Copy)]
struct HostBridgeInvocation {
    context: *mut c_void,
    call_json: NimbusBunEmbedHostCallJsonFn,
}

thread_local! {
    static CURRENT_HOST_BRIDGE_INVOCATION: Cell<Option<HostBridgeInvocation>> = const { Cell::new(None) };
    static PENDING_INVOCATION_RESPONSE: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

struct HostBridgeInvocationGuard {
    previous: Option<HostBridgeInvocation>,
}

impl HostBridgeInvocationGuard {
    fn install(context: *mut c_void, call_json: NimbusBunEmbedHostCallJsonFn) -> Self {
        let previous = CURRENT_HOST_BRIDGE_INVOCATION.with(|slot| {
            let previous = slot.get();
            slot.set(Some(HostBridgeInvocation { context, call_json }));
            previous
        });
        Self { previous }
    }
}

impl Drop for HostBridgeInvocationGuard {
    fn drop(&mut self) {
        CURRENT_HOST_BRIDGE_INVOCATION.with(|slot| slot.set(self.previous));
    }
}

const GENERATED_NIMBUS_PROGRAM_BUNDLE: &[u8] = include_bytes!("nimbus_generated_program_bundle.js");

const NIMBUS_HOST_BRIDGE_TRANSPORT_SOURCE: &[u8] = br#"
(() => {
const __nimbusRawHostBridgeCallJson = globalThis.__nimbusHostBridgeCallJson;
if (typeof __nimbusRawHostBridgeCallJson !== "function") {
  return 0;
}
if (!delete globalThis.__nimbusHostBridgeCallJson || "__nimbusHostBridgeCallJson" in globalThis) {
  return 0;
}

const __nimbusJsonParse = JSON.parse;
const __nimbusJsonStringify = JSON.stringify;
const __nimbusArrayIsArray = Array.isArray;
const __nimbusObjectKeys = Object.keys;
const __nimbusObjectSetPrototypeOf = Object.setPrototypeOf;
const __nimbusWeakSet = WeakSet;
const __nimbusWeakSetAdd = Function.prototype.call.bind(WeakSet.prototype.add);
const __nimbusWeakSetDelete = Function.prototype.call.bind(WeakSet.prototype.delete);
const __nimbusWeakSetHas = Function.prototype.call.bind(WeakSet.prototype.has);
const __nimbusHostOperationNames = Object.freeze({
  __proto__: null,
  op_nimbus_http_route: "http_route",
  op_nimbus_ctx_query: "ctx_query",
  op_nimbus_ctx_paginated_query: "ctx_paginated_query",
  op_nimbus_ctx_mutation: "ctx_mutation",
  op_nimbus_ctx_action: "ctx_action",
  op_nimbus_ctx_run_query: "ctx_run_query",
  op_nimbus_ctx_run_mutation: "ctx_run_mutation",
  op_nimbus_ctx_run_action: "ctx_run_action",
  op_nimbus_document_get: "document_get",
  op_nimbus_ctx_query_start: "query_builder_start",
  op_nimbus_ctx_query_with_index: "query_builder_with_index",
  op_nimbus_ctx_query_filter: "query_builder_filter",
  op_nimbus_ctx_query_order: "query_builder_order",
  op_nimbus_ctx_query_collect: "query_read_collect",
  op_nimbus_ctx_query_take: "query_read_take",
  op_nimbus_ctx_query_paginate: "query_read_paginate",
  op_nimbus_ctx_query_first: "query_read_first",
  op_nimbus_ctx_query_unique: "query_read_unique",
  op_nimbus_document_insert: "document_insert",
  op_nimbus_document_patch: "document_patch",
  op_nimbus_document_delete: "document_delete",
  op_nimbus_ctx_scheduler_run_after: "ctx_scheduler_run_after",
  op_nimbus_ctx_scheduler_run_at: "ctx_scheduler_run_at",
  op_nimbus_ctx_scheduler_cancel: "ctx_scheduler_cancel",
  op_nimbus_ctx_service_lookup: "ctx_service_lookup",
  op_nimbus_ctx_runtime_enter_nested_call: "ctx_runtime_enter_nested_call",
  op_nimbus_ctx_resolve_callee_lane: "ctx_resolve_callee_lane",
  op_nimbus_cf_kv_get: "cf_kv_get",
  op_nimbus_cf_kv_put: "cf_kv_put",
  op_nimbus_cf_kv_delete: "cf_kv_delete",
  op_nimbus_cf_kv_list: "cf_kv_list",
  op_nimbus_runtime_extension_call: "runtime_extension_call",
});

function __nimbusFormatHostError(error) {
  if (error === null || error === undefined) {
    return "unknown host error";
  }
  if (typeof error === "string") {
    return error;
  }
  try {
    return __nimbusJsonStringify(error);
  } catch (_error) {
    return String(error);
  }
}

function __nimbusNormalizeHostOperationName(opName) {
  const operation = __nimbusHostOperationNames[opName];
  if (typeof operation !== "string") {
    throw new Error(`Nimbus Bun/JSC host op not found: ${opName}`);
  }
  return operation;
}

function __nimbusCloneJsonValue(value, seen) {
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (__nimbusWeakSetHas(seen, value)) {
    throw new TypeError("Nimbus Bun/JSC host payload must not contain a cycle");
  }
  __nimbusWeakSetAdd(seen, value);

  let clone;
  if (__nimbusArrayIsArray(value)) {
    clone = [];
    __nimbusObjectSetPrototypeOf(clone, null);
    for (let index = 0; index < value.length; index++) {
      clone[index] = __nimbusCloneJsonValue(value[index], seen);
    }
  } else {
    clone = { __proto__: null };
    const keys = __nimbusObjectKeys(value);
    for (let index = 0; index < keys.length; index++) {
      const key = keys[index];
      if (key !== "toJSON") {
        clone[key] = __nimbusCloneJsonValue(value[key], seen);
      }
    }
  }

  __nimbusWeakSetDelete(seen, value);
  return clone;
}

function __nimbusCallHostBridge(opName, payload) {
  const request = {
    __proto__: null,
    abi_version: 1,
    operation: __nimbusNormalizeHostOperationName(opName),
    payload: __nimbusCloneJsonValue(payload ?? null, new __nimbusWeakSet()),
  };
  const responseText = __nimbusRawHostBridgeCallJson(__nimbusJsonStringify(request));
  const response = __nimbusJsonParse(responseText);
  if (!response || response.status !== "ok") {
    const error = new Error(
      `Nimbus Bun/JSC host call failed for ${opName}: ${__nimbusFormatHostError(response?.error)}`,
    );
    error.nimbusHostError = response?.error ?? null;
    throw error;
  }
  return response.value;
}

function __nimbusNormalizeFunctionReference(functionRef, label) {
  if (!functionRef || typeof functionRef !== "object") {
    throw new Error(`ctx.${label}(...) requires a generated function reference`);
  }
  if (typeof functionRef.name !== "string" || functionRef.name.length === 0) {
    throw new Error(`ctx.${label}(...) requires a named generated function reference`);
  }
  return {
    name: functionRef.name,
    visibility: typeof functionRef.visibility === "string" ? functionRef.visibility : "public",
  };
}

function __nimbusNormalizeFieldName(field) {
  if (typeof field === "string" && field.length > 0) {
    return field;
  }
  if (
    field !== null &&
    typeof field === "object" &&
    typeof field.__fieldName === "string" &&
    field.__fieldName.length > 0
  ) {
    return field.__fieldName;
  }
  throw new Error("ctx.db field constraints require a non-empty field name");
}

function __nimbusCreateConstraintBuilder() {
  const filters = [];
  const builder = {
    field(name) {
      return { __fieldName: __nimbusNormalizeFieldName(name) };
    },
    eq(field, value) {
      filters.push({ field: __nimbusNormalizeFieldName(field), op: "eq", value });
      return builder;
    },
    neq(field, value) {
      filters.push({ field: __nimbusNormalizeFieldName(field), op: "neq", value });
      return builder;
    },
    gt(field, value) {
      filters.push({ field: __nimbusNormalizeFieldName(field), op: "gt", value });
      return builder;
    },
    gte(field, value) {
      filters.push({ field: __nimbusNormalizeFieldName(field), op: "gte", value });
      return builder;
    },
    lt(field, value) {
      filters.push({ field: __nimbusNormalizeFieldName(field), op: "lt", value });
      return builder;
    },
    lte(field, value) {
      filters.push({ field: __nimbusNormalizeFieldName(field), op: "lte", value });
      return builder;
    },
  };
  return Object.assign(builder, { __filters: filters });
}

function __nimbusCollectConstraintFilters(builderFn, label) {
  const builder = __nimbusCreateConstraintBuilder();
  const result = builderFn ? builderFn(builder) : builder;
  if (result !== undefined && result !== builder && result?.__filters !== builder.__filters) {
    throw new Error(`ctx.db.${label}(...) must return the provided builder`);
  }
  return [...builder.__filters];
}

function __nimbusCreateQueryBuilder(syncHostValue, asyncHostValue, builderId) {
  return Object.freeze({
    __builderId: builderId,
    withIndex(indexName, builderFn) {
      syncHostValue("op_nimbus_ctx_query_with_index", {
        builder_id: builderId,
        index_name: indexName,
        filters: __nimbusCollectConstraintFilters(builderFn, "withIndex"),
      });
      return __nimbusCreateQueryBuilder(syncHostValue, asyncHostValue, builderId);
    },
    filter(builderFn) {
      syncHostValue("op_nimbus_ctx_query_filter", {
        builder_id: builderId,
        filters: __nimbusCollectConstraintFilters(builderFn, "filter"),
      });
      return __nimbusCreateQueryBuilder(syncHostValue, asyncHostValue, builderId);
    },
    order(direction) {
      syncHostValue("op_nimbus_ctx_query_order", {
        builder_id: builderId,
        direction,
      });
      return __nimbusCreateQueryBuilder(syncHostValue, asyncHostValue, builderId);
    },
    collect() {
      return asyncHostValue("op_nimbus_ctx_query_collect", { builder_id: builderId });
    },
    take(limit) {
      return asyncHostValue("op_nimbus_ctx_query_take", { builder_id: builderId, limit });
    },
    paginate(options = {}) {
      return asyncHostValue("op_nimbus_ctx_query_paginate", {
        builder_id: builderId,
        page_size: options.numItems,
        cursor: typeof options.cursor === "string" ? options.cursor : null,
      });
    },
    first() {
      return asyncHostValue("op_nimbus_ctx_query_first", { builder_id: builderId });
    },
    unique() {
      return asyncHostValue("op_nimbus_ctx_query_unique", { builder_id: builderId });
    },
  });
}

let __nimbusNextSessionId = 1;

const __nimbusSyncHostValue = function(opName, payload) {
  return __nimbusCallHostBridge(opName, payload);
};

const __nimbusAsyncHostValue = async function(opName, payload) {
  return __nimbusCallHostBridge(opName, payload);
};

const __nimbusCreateContext = function(options = {}) {
  const hostCallSessionId =
    typeof options.hostCallSessionId === "string" && options.hostCallSessionId.length > 0
      ? options.hostCallSessionId
      : `session-${__nimbusNextSessionId++}`;
  const services =
    options.request !== null &&
    typeof options.request === "object" &&
    options.request.services !== null &&
    typeof options.request.services === "object"
      ? options.request.services
      : null;
  const withSession = (payload) => ({
    host_call_session_id: hostCallSessionId,
    ...(payload ?? {}),
  });
  const syncHostValue = (opName, payload) => __nimbusSyncHostValue(opName, withSession(payload));
  const asyncHostValue = (opName, payload) => __nimbusAsyncHostValue(opName, withSession(payload));
  const runFunction = (opName, kind, label, functionRef, args = {}) => {
    const normalized = __nimbusNormalizeFunctionReference(functionRef, label);
    return asyncHostValue(opName, {
      ...normalized,
      args,
    });
  };
  return {
    db: {
      get(tableOrId, maybeId) {
        if (maybeId === undefined) {
          if (
            tableOrId &&
            typeof tableOrId === "object" &&
            typeof tableOrId.table === "string" &&
            typeof tableOrId.id === "string"
          ) {
            return asyncHostValue("op_nimbus_document_get", {
              table: tableOrId.table,
              id: tableOrId.id,
            });
          }
          throw new Error("Nimbus Bun/JSC ctx.db.get requires table and id");
        }
        return asyncHostValue("op_nimbus_document_get", {
          table: tableOrId,
          id: maybeId,
        });
      },
      query(table) {
        const builderId = syncHostValue("op_nimbus_ctx_query_start", { table });
        return __nimbusCreateQueryBuilder(syncHostValue, asyncHostValue, builderId);
      },
      insert(table, fields) {
        return asyncHostValue("op_nimbus_document_insert", { table, fields });
      },
      patch(table, id, patch) {
        return asyncHostValue("op_nimbus_document_patch", { table, id, patch });
      },
      delete(table, id) {
        return asyncHostValue("op_nimbus_document_delete", { table, id });
      },
    },
    scheduler: {
      runAfter(delayMs, functionRef, args = {}) {
        const normalized = __nimbusNormalizeFunctionReference(functionRef, "scheduler.runAfter");
        return asyncHostValue("op_nimbus_ctx_scheduler_run_after", {
          delay_ms: delayMs,
          ...normalized,
          args,
        });
      },
      runAt(timestampMs, functionRef, args = {}) {
        const normalized = __nimbusNormalizeFunctionReference(functionRef, "scheduler.runAt");
        return asyncHostValue("op_nimbus_ctx_scheduler_run_at", {
          timestamp_ms: timestampMs,
          ...normalized,
          args,
        });
      },
      cancel(jobId) {
        return asyncHostValue("op_nimbus_ctx_scheduler_cancel", { job_id: jobId });
      },
    },
    services: Object.freeze({
      get(serviceName) {
        if (services && Object.prototype.hasOwnProperty.call(services, serviceName)) {
          return services[serviceName];
        }
        return asyncHostValue("op_nimbus_ctx_service_lookup", { service_name: serviceName });
      },
    }),
    runQuery(functionRef, args = {}) {
      return runFunction("op_nimbus_ctx_run_query", "query", "runQuery", functionRef, args);
    },
    runMutation(functionRef, args = {}) {
      return runFunction("op_nimbus_ctx_run_mutation", "mutation", "runMutation", functionRef, args);
    },
    runAction(functionRef, args = {}) {
      return runFunction("op_nimbus_ctx_run_action", "action", "runAction", functionRef, args);
    },
  };
};

Object.freeze(__nimbusSyncHostValue);
Object.freeze(__nimbusAsyncHostValue);
Object.freeze(__nimbusCreateContext);
Object.defineProperties(globalThis, {
  __nimbusSyncHostValue: {
    value: __nimbusSyncHostValue,
    writable: false,
    configurable: false,
    enumerable: false,
  },
  __nimbusAsyncHostValue: {
    value: __nimbusAsyncHostValue,
    writable: false,
    configurable: false,
    enumerable: false,
  },
  __nimbusCreateContext: {
    value: __nimbusCreateContext,
    writable: false,
    configurable: false,
    enumerable: false,
  },
});
return 1;
})()
"#;

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

    fn Bun__embedderApplyNativePermissionDenyProfile(global_object: *mut JSGlobalObject) -> bool;
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_probe_construct_and_destroy_vm() -> i32 {
    const CONCURRENT_FIRST_VMS: usize = 4;

    let gate = Arc::new(InitProofGate::new(CONCURRENT_FIRST_VMS));
    let mut workers = Vec::with_capacity(CONCURRENT_FIRST_VMS);
    let mut spawn_failed = false;
    for worker_id in 0..CONCURRENT_FIRST_VMS {
        let gate_for_worker = Arc::clone(&gate);
        let spawned = thread::Builder::new()
            .name(format!("nimbus-bun-init-proof-{worker_id}"))
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    construct_vm_and_run_with_init_gate(|_| 0, Some(gate_for_worker.as_ref()))
                }));
                match result {
                    Ok(status) => status,
                    Err(_) => {
                        gate_for_worker.cancel();
                        319
                    }
                }
            });
        match spawned {
            Ok(worker) => workers.push(worker),
            Err(_) => {
                spawn_failed = true;
                gate.cancel();
                break;
            }
        }
    }

    let mut aggregate_status = if spawn_failed { 317 } else { 0 };
    for worker in workers {
        match worker.join() {
            Ok(0) => {}
            Ok(status) if aggregate_status == 0 => aggregate_status = status,
            Ok(_) => {}
            Err(_) if aggregate_status == 0 => aggregate_status = 319,
            Err(_) => {}
        }
    }
    aggregate_status
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

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_invoke_program_wrapper_json(
    bundle_ptr: *const u8,
    bundle_len: usize,
    expected_sha256_ptr: *const u8,
    expected_sha256_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    output_ptr: *mut u8,
    output_cap: usize,
    output_len: *mut usize,
    cancellation_context: *mut c_void,
    is_cancelled: Option<NimbusBunEmbedIsCancelledFn>,
) -> i32 {
    clear_pending_invocation_response();
    if bundle_ptr.is_null() || request_ptr.is_null() || output_ptr.is_null() || output_len.is_null()
    {
        return 300;
    }
    if cancellation_context.is_null() {
        return 300;
    }
    let Some(is_cancelled) = is_cancelled else {
        return 300;
    };

    // SAFETY: pointer validity is the caller's ABI contract. The slices are
    // consumed synchronously before the function returns and are never stored.
    let bundle_source = unsafe { slice::from_raw_parts(bundle_ptr, bundle_len) };
    if verify_bundle_sha256(bundle_source, expected_sha256_ptr, expected_sha256_len).is_err() {
        return 313;
    }
    // SAFETY: pointer validity is the caller's ABI contract. The slices are
    // consumed synchronously before the function returns and are never stored.
    let request_bytes = unsafe { slice::from_raw_parts(request_ptr, request_len) };
    let request_json = match str::from_utf8(request_bytes) {
        Ok(request_json) => request_json.as_bytes(),
        Err(_) => return 301,
    };

    construct_vm_and_run(|vm| {
        run_program_wrapper_json_invocation(
            vm,
            bundle_source,
            request_json,
            output_ptr,
            output_cap,
            output_len,
            cancellation_context,
            is_cancelled,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_invoke_program_wrapper_json_with_host_bridge(
    bundle_ptr: *const u8,
    bundle_len: usize,
    expected_sha256_ptr: *const u8,
    expected_sha256_len: usize,
    request_ptr: *const u8,
    request_len: usize,
    output_ptr: *mut u8,
    output_cap: usize,
    output_len: *mut usize,
    host_context: *mut c_void,
    host_call_json: Option<NimbusBunEmbedHostCallJsonFn>,
    cancellation_context: *mut c_void,
    is_cancelled: Option<NimbusBunEmbedIsCancelledFn>,
) -> i32 {
    clear_pending_invocation_response();
    if bundle_ptr.is_null()
        || request_ptr.is_null()
        || output_ptr.is_null()
        || output_len.is_null()
        || host_context.is_null()
    {
        return 300;
    }
    if cancellation_context.is_null() {
        return 300;
    }
    let Some(is_cancelled) = is_cancelled else {
        return 300;
    };
    let Some(host_call_json) = host_call_json else {
        return 308;
    };

    // SAFETY: pointer validity is the caller's ABI contract. The slices are
    // consumed synchronously before the function returns and are never stored.
    let bundle_source = unsafe { slice::from_raw_parts(bundle_ptr, bundle_len) };
    if verify_bundle_sha256(bundle_source, expected_sha256_ptr, expected_sha256_len).is_err() {
        return 313;
    }
    // SAFETY: pointer validity is the caller's ABI contract. The slices are
    // consumed synchronously before the function returns and are never stored.
    let request_bytes = unsafe { slice::from_raw_parts(request_ptr, request_len) };
    let request_json = match str::from_utf8(request_bytes) {
        Ok(request_json) => request_json.as_bytes(),
        Err(_) => return 301,
    };

    construct_vm_and_run(|vm| {
        let _host_bridge_guard = HostBridgeInvocationGuard::install(host_context, host_call_json);
        run_program_wrapper_json_invocation(
            vm,
            bundle_source,
            request_json,
            output_ptr,
            output_cap,
            output_len,
            cancellation_context,
            is_cancelled,
        )
    })
}

/// Copies the completed response retained by the most recent invocation on
/// this thread. Status 307 leaves the response available for a larger retry;
/// status 320 means no completed response is pending. A new invocation on the
/// same thread invalidates any response that was not taken.
#[unsafe(no_mangle)]
pub extern "C" fn nimbus_bun_embed_take_pending_response(
    output_ptr: *mut u8,
    output_cap: usize,
    output_len: *mut usize,
) -> i32 {
    if output_ptr.is_null() || output_len.is_null() {
        return 300;
    }

    PENDING_INVOCATION_RESPONSE.with(|slot| {
        let mut pending = slot.borrow_mut();
        let Some(response) = pending.as_ref() else {
            return 320;
        };

        // SAFETY: `output_len` was validated non-null and is owned by the
        // caller for this synchronous ABI call.
        unsafe {
            *output_len = response.len();
        }
        if response.len() > output_cap {
            return 307;
        }

        // SAFETY: `output_ptr` was validated non-null and the capacity check
        // bounds this copy into the caller-provided output buffer.
        unsafe {
            core::ptr::copy_nonoverlapping(response.as_ptr(), output_ptr, response.len());
        }
        pending.take();
        0
    })
}

fn clear_pending_invocation_response() {
    PENDING_INVOCATION_RESPONSE.with(|slot| {
        slot.borrow_mut().take();
    });
}

fn verify_bundle_sha256(
    bundle_source: &[u8],
    expected_sha256_ptr: *const u8,
    expected_sha256_len: usize,
) -> Result<(), ()> {
    if expected_sha256_ptr.is_null() || expected_sha256_len != 64 {
        return Err(());
    }

    // SAFETY: a non-null 64-byte digest pointer is part of the synchronous ABI
    // contract and is consumed before the call returns.
    let expected = unsafe { slice::from_raw_parts(expected_sha256_ptr, expected_sha256_len) };
    let mut actual = [0_u8; 32];
    // SAFETY: SHA256 writes exactly the 32-byte digest and a null engine selects
    // BoringSSL's default implementation.
    unsafe {
        bun_sha_hmac::SHA256::hash(bundle_source, &mut actual, core::ptr::null_mut());
    }

    let mut different = 0_u8;
    for (index, byte) in actual.iter().enumerate() {
        let high = decode_hex(expected[index * 2]).ok_or(())?;
        let low = decode_hex(expected[index * 2 + 1]).ok_or(())?;
        different |= byte ^ ((high << 4) | low);
    }
    if different == 0 { Ok(()) } else { Err(()) }
}

fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

struct InvocationCancellationWatcher {
    completed: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    context: usize,
    is_cancelled: NimbusBunEmbedIsCancelledFn,
    worker: Option<thread::JoinHandle<()>>,
}

impl InvocationCancellationWatcher {
    fn start(
        vm: &VirtualMachine,
        context: *mut c_void,
        is_cancelled: NimbusBunEmbedIsCancelledFn,
    ) -> Result<Self, ()> {
        let completed = Arc::new(AtomicBool::new(false));
        let cancelled = Arc::new(AtomicBool::new(false));
        let completed_for_thread = Arc::clone(&completed);
        let cancelled_for_thread = Arc::clone(&cancelled);
        let context = context as usize;
        let vm_handle = vm.handle();
        let worker = thread::Builder::new()
            .name("nimbus-bun-cancellation".to_owned())
            .spawn(move || {
                bun_core::StackCheck::configure_thread();
                while !completed_for_thread.load(Ordering::SeqCst) {
                    // SAFETY: the invocation owns the callback context until this
                    // watcher is joined before VM teardown.
                    if unsafe { is_cancelled(context as *mut c_void) } {
                        cancelled_for_thread.store(true, Ordering::SeqCst);
                        // Bun's any-thread termination API closes the script
                        // gate before it raises the JSC trap. This invocation
                        // owns a fresh VM and discards it after cancellation.
                        vm_handle.request_termination();
                        return;
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            })
            .map_err(|_| ())?;
        Ok(Self {
            completed,
            cancelled,
            context,
            is_cancelled,
            worker: Some(worker),
        })
    }

    fn finish(mut self) -> Result<bool, ()> {
        self.completed.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| ())?;
        }
        let watcher_observed_cancellation = self.cancelled.load(Ordering::SeqCst);
        // The invocation can complete between the watcher's polling intervals.
        // Read the token once on the owner thread before the callback context
        // leaves scope so a fast guest catch cannot hide host cancellation.
        let token_is_cancelled = unsafe { (self.is_cancelled)(self.context as *mut c_void) };
        Ok(watcher_observed_cancellation || token_is_cancelled)
    }
}

impl Drop for InvocationCancellationWatcher {
    fn drop(&mut self) {
        self.completed.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn construct_vm_and_run(run: impl FnOnce(&mut VirtualMachine) -> i32) -> i32 {
    construct_vm_and_run_with_init_gate(run, None)
}

fn construct_vm_and_run_with_init_gate(
    run: impl FnOnce(&mut VirtualMachine) -> i32,
    init_proof_gate: Option<&InitProofGate>,
) -> i32 {
    bun_core::output::init_embedder_thread();
    bun_core::StackCheck::configure_thread();
    trace_embed_probe(b"nimbus-embed-probe: thread initialized\n");

    trace_embed_probe(b"nimbus-embed-probe: waiting at initialization gate\n");
    if init_proof_gate.is_some_and(|gate| !gate.arrive_and_wait()) {
        trace_embed_probe(b"nimbus-embed-probe: initialization gate failed\n");
        return 318;
    }
    trace_embed_probe(b"nimbus-embed-probe: initialization gate released\n");

    EMBEDDER_PROCESS_INITIALIZED.call_once(|| {
        trace_embed_probe(b"nimbus-embed-probe: process initialization started\n");
        bun_jsc::initialize(bun_jsc::InitializeOptions::default());

        // Touch the high-tier runtime hooks so this staticlib root owns
        // `__BUN_RUNTIME_HOOKS` without depending on Bun's process-owned CLI
        // root.
        bun_runtime::jsc_hooks::embedder_touch_runtime_state();
        trace_embed_probe(b"nimbus-embed-probe: process initialization completed\n");
    });
    trace_embed_probe(b"nimbus-embed-probe: process initialization observed\n");

    let opts = InitOptions {
        is_main_thread: false,
        ..Default::default()
    };

    trace_embed_probe(b"nimbus-embed-probe: VM initialization started\n");
    match VirtualMachine::init(opts) {
        Ok(vm) => {
            trace_embed_probe(b"nimbus-embed-probe: VM initialization completed\n");
            // SAFETY: `VirtualMachine::init` returned a fresh VM pointer for
            // this probe invocation. The closure runs before teardown and does
            // not store the mutable reference beyond this stack frame.
            let vm = unsafe { &mut *vm };
            let status = run(vm);
            // Bun's teardown contract closes the native-to-script gate before
            // event-loop destruction can land a pending termination outside a
            // script frame. The embedder owns the same ordering even though it
            // uses the smaller direct-destroy path for its fresh VM.
            {
                let _lock = ProbeApiLock::new(vm.jsc_vm());
                // The C ABI reports JavaScript failures as integer statuses.
                // Consume any translated exception before this fresh VM is
                // destroyed; termination also owns a separate request flag.
                vm.global().clear_termination_exception();
                vm.global().clear_exception();
                vm.forbid_script();
            }
            trace_embed_probe(b"nimbus-embed-probe: VM destruction started\n");
            vm.destroy();
            trace_embed_probe(b"nimbus-embed-probe: VM destruction completed\n");
            status
        }
        Err(_) => {
            trace_embed_probe(b"nimbus-embed-probe: VM initialization returned an error\n");
            1
        }
    }
}

struct EmbedderResolutionDenyGuard {
    previous: bool,
}

impl EmbedderResolutionDenyGuard {
    fn new() -> Self {
        Self {
            previous: bun_jsc::ModuleLoader::set_embedder_deny_all_module_resolution(true),
        }
    }
}

impl Drop for EmbedderResolutionDenyGuard {
    fn drop(&mut self) {
        bun_jsc::ModuleLoader::set_embedder_deny_all_module_resolution(self.previous);
    }
}

struct ProofCancellationToken {
    cancelled: AtomicBool,
}

impl ProofCancellationToken {
    fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
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
        if vm.wait_for_promise(AnyPromise::Normal(promise)).is_err() {
            return 17;
        }

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
        if vm
            .wait_for_promise(AnyPromise::Normal(async_invocation_promise))
            .is_err()
        {
            return 38;
        }

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

fn run_program_wrapper_json_invocation(
    vm: &mut VirtualMachine,
    bundle_source: &[u8],
    request_json: &[u8],
    output_ptr: *mut u8,
    output_cap: usize,
    output_len: *mut usize,
    cancellation_context: *mut c_void,
    is_cancelled: NimbusBunEmbedIsCancelledFn,
) -> i32 {
    // SAFETY: the public ABI requires a non-null context and callback, and the
    // caller owns the context until this synchronous call returns.
    if unsafe { is_cancelled(cancellation_context) } {
        return 314;
    }
    let _lock = ProbeApiLock::new(vm.jsc_vm());
    // Cross-thread termination needs the sentinel to exist before the watcher
    // can request a trap on this VM.
    let _ = vm.jsc_vm().termination_exception();
    let cancellation =
        match InvocationCancellationWatcher::start(vm, cancellation_context, is_cancelled) {
            Ok(cancellation) => cancellation,
            Err(()) => return 316,
        };
    let status = run_program_wrapper_json_invocation_inner(
        vm,
        bundle_source,
        request_json,
        output_ptr,
        output_cap,
        output_len,
    );
    let cancelled = match cancellation.finish() {
        Ok(cancelled) => cancelled,
        Err(()) => {
            vm.global().clear_termination_exception();
            return 316;
        }
    };
    if cancelled {
        vm.global().clear_termination_exception();
        314
    } else {
        status
    }
}

fn run_program_wrapper_json_invocation_inner(
    vm: &mut VirtualMachine,
    bundle_source: &[u8],
    request_json: &[u8],
    output_ptr: *mut u8,
    output_cap: usize,
    output_len: *mut usize,
) -> i32 {
    vm.event_loop_mut().ensure_waker();

    let global = vm.global();
    let _resolution_deny_guard = EmbedderResolutionDenyGuard::new();
    let _lock = ProbeApiLock::new(vm.jsc_vm());

    let request = match EncodedSlice::utf8(request_json).to_json_object(global) {
        Ok(request) => request,
        Err(_) => return 301,
    };
    if request.is_empty() || !request.is_object() || global.has_exception() {
        return 301;
    }
    // Bundle evaluation can allocate and collect. Keep the parsed request live
    // until the guest invocation consumes it.
    let request = request.protected();

    // SAFETY: this fresh embedder VM is single-threaded under the JSC API lock.
    // The native helper mutates only the current global object's permission
    // profile before tenant code is evaluated.
    if bun_jsc::call_false_is_throw(global, || unsafe {
        Bun__embedderApplyNativePermissionDenyProfile(
            global as *const JSGlobalObject as *mut JSGlobalObject,
        )
    })
    .is_err()
    {
        global.clear_exception();
        return 315;
    }

    if current_host_bridge_invocation_installed() {
        if let Err(status) = install_host_bridge_transport(global) {
            return status;
        }
    }

    if let Err(status) = evaluate_program(
        global,
        bundle_source,
        b"nimbus-bun-linked-adapter-program-wrapper.js",
        302,
    ) {
        return status;
    }

    let invoke = match global.to_js_value().get(global, b"__nimbusInvoke") {
        Ok(Some(invoke)) if invoke.is_callable() => invoke,
        _ => return 303,
    };
    let result = match invoke.call_with_global_this(global, &[request.value()]) {
        Ok(result) => result,
        Err(_) => return 303,
    };

    let result = match result.as_promise() {
        Some(promise) => {
            if vm.wait_for_promise(AnyPromise::Normal(promise)).is_err() {
                return 304;
            }
            let promise = JSPromise::opaque_mut(promise);
            if promise.status() != PromiseStatus::Fulfilled {
                return 304;
            }
            promise.result(vm.jsc_vm())
        }
        None => result,
    };

    let json_output = match result.json_stringify_fast(global) {
        Ok(output) => output,
        Err(_) => return 305,
    };
    if json_output.tag() == bun_core::Tag::Dead {
        return 306;
    }
    let json_output = json_output.to_utf8();
    let response = json_output.slice();

    // SAFETY: `output_len` was validated non-null and is owned by the caller.
    unsafe {
        *output_len = response.len();
    }
    if response.len() > output_cap {
        PENDING_INVOCATION_RESPONSE.with(|slot| {
            slot.replace(Some(response.to_vec()));
        });
        return 307;
    }

    // SAFETY: `output_ptr` was validated non-null and the caller advertised at
    // least `output_cap` writable bytes. The length check above bounds the copy.
    unsafe {
        core::ptr::copy_nonoverlapping(response.as_ptr(), output_ptr, response.len());
    }
    0
}

fn current_host_bridge_invocation_installed() -> bool {
    CURRENT_HOST_BRIDGE_INVOCATION.with(|slot| slot.get().is_some())
}

fn install_host_bridge_transport(global: &JSGlobalObject) -> Result<(), i32> {
    global.to_js_value().put(
        global,
        b"__nimbusHostBridgeCallJson",
        JSFunction::create(
            global,
            "__nimbusHostBridgeCallJson",
            __jsc_host_nimbus_bun_embed_host_bridge_call_json,
            1,
            Default::default(),
        ),
    );

    let result = evaluate_program(
        global,
        NIMBUS_HOST_BRIDGE_TRANSPORT_SOURCE,
        b"nimbus-bun-linked-adapter-host-bridge-transport.js",
        309,
    )?;
    if result.is_number() && result.as_number() as i32 == 1 {
        Ok(())
    } else {
        Err(310)
    }
}

fn run_timeout_and_cancel_probe(vm: &mut VirtualMachine) -> i32 {
    vm.event_loop_mut().ensure_waker();

    let global = vm.global();
    {
        let _lock = vm.jsc_vm().get_api_lock();
        // Cross-thread `notify_need_termination` expects JSC's termination
        // exception to have been materialized by the owning thread first.
        let _ = vm.jsc_vm().termination_exception();

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

    if let Err(status) = evaluate_cancel_before_guest_entry(vm, global) {
        return status;
    }
    if let Err(status) = evaluate_recovery_script(vm, global, 67, 68) {
        return status;
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

    eprintln!("nimbus bun embed cancellation policy:");
    eprintln!("  before_guest_entry: owner_entry_gate_denied_and_recovered");
    eprintln!("  after_guest_entry_sync_loop: spin_entered_ack");
    eprintln!("  recovery_after_deadline_cancel: ok");
    eprintln!("  recovery_after_external_cancel: ok");
    eprintln!("  cancellation_timing_policy: state_ack_not_sleep");

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

macro_rules! denied_bun_surface {
    ($property:literal) => {
        PermissionSurfaceProbe {
            name: concat!("Bun.", $property),
            source: concat!(
                "globalThis.__nimbusPermissionProbeFunction(globalThis.Bun?.",
                $property,
                ")"
            )
            .as_bytes(),
        }
    };
}

macro_rules! denied_global_surface {
    ($property:literal) => {
        PermissionSurfaceProbe {
            name: $property,
            source: concat!(
                "globalThis.__nimbusPermissionProbeFunction(globalThis.",
                $property,
                ")"
            )
            .as_bytes(),
        }
    };
}

macro_rules! denied_console_surface {
    ($property:literal) => {
        PermissionSurfaceProbe {
            name: concat!("console.", $property),
            source: concat!(
                "globalThis.__nimbusPermissionProbeFunction(globalThis.console?.",
                $property,
                ")"
            )
            .as_bytes(),
        }
    };
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
    denied_bun_surface!("$"),
    denied_bun_surface!("Archive"),
    denied_bun_surface!("FileSystemRouter"),
    denied_bun_surface!("Glob"),
    denied_bun_surface!("Image"),
    denied_bun_surface!("RedisClient"),
    denied_bun_surface!("S3Client"),
    denied_bun_surface!("SQL"),
    denied_bun_surface!("Terminal"),
    denied_bun_surface!("WebView"),
    denied_bun_surface!("allocUnsafe"),
    denied_bun_surface!("argv"),
    denied_bun_surface!("build"),
    denied_bun_surface!("connect"),
    denied_bun_surface!("cron"),
    denied_bun_surface!("cwd"),
    denied_bun_surface!("dns"),
    denied_bun_surface!("embeddedFiles"),
    denied_bun_surface!("enableANSIColors"),
    denied_bun_surface!("fetch"),
    denied_bun_surface!("file"),
    denied_bun_surface!("gc"),
    denied_bun_surface!("generateHeapSnapshot"),
    denied_bun_surface!("isStandaloneExecutable"),
    denied_bun_surface!("jest"),
    denied_bun_surface!("listen"),
    denied_bun_surface!("main"),
    denied_bun_surface!("mmap"),
    denied_bun_surface!("openInEditor"),
    denied_bun_surface!("origin"),
    denied_bun_surface!("password"),
    denied_bun_surface!("plugin"),
    denied_bun_surface!("postgres"),
    denied_bun_surface!("redis"),
    denied_bun_surface!("registerMacro"),
    denied_bun_surface!("resolve"),
    denied_bun_surface!("resolveSync"),
    denied_bun_surface!("s3"),
    denied_bun_surface!("secrets"),
    denied_bun_surface!("serve"),
    denied_bun_surface!("shrink"),
    denied_bun_surface!("sleep"),
    denied_bun_surface!("sleepSync"),
    denied_bun_surface!("spawn"),
    denied_bun_surface!("spawnSync"),
    denied_bun_surface!("sql"),
    denied_bun_surface!("stderr"),
    denied_bun_surface!("stdin"),
    denied_bun_surface!("stdout"),
    denied_bun_surface!("udpSocket"),
    denied_bun_surface!("unsafe"),
    denied_bun_surface!("which"),
    denied_bun_surface!("write"),
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
        name: "Bun property coverage",
        source: br#"
(() => {
  const allowed = new Set([
    "ArrayBufferSink", "Cookie", "CookieMap", "CryptoHasher", "CSRF", "FFI",
    "JSON5", "JSONC", "JSONL", "MD4", "MD5", "SHA1", "SHA224", "SHA256",
    "SHA384", "SHA512", "SHA512_256", "TOML", "Transpiler", "XML", "YAML",
    "__nimbusNativePermissionProfile", "color", "concatArrayBuffers", "deepEquals",
    "deepMatch", "deflateSync", "env", "escapeHTML", "fileURLToPath", "gunzipSync",
    "gzipSync", "hash", "indexOfLine", "inflateSync", "inspect", "isMainThread",
    "markdown", "nanoseconds", "pathToFileURL", "peek", "randomUUIDv5",
    "randomUUIDv7", "readableStreamToArray", "readableStreamToArrayBuffer",
    "readableStreamToBlob", "readableStreamToBytes", "readableStreamToFormData",
    "readableStreamToJSON", "readableStreamToText", "revision", "semver", "sha",
    "sliceAnsi", "stringWidth", "stripANSI", "version", "version_with_sha", "wrapAnsi",
    "zstdCompress", "zstdCompressSync", "zstdDecompress", "zstdDecompressSync",
  ]);
  return Object.getOwnPropertyNames(globalThis.Bun).every((name) =>
    allowed.has(name) || globalThis.Bun[name]?.__nimbusDeniedNativeCapability === true
  ) ? 3 : 5;
})()
"#,
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
        name: "Buffer unsafe allocation zero fill",
        source: br#"
(() => {
  const fast = Buffer.allocUnsafe(4096);
  const slow = Buffer.allocUnsafeSlow(4096);
  return fast.every((byte) => byte === 0) && slow.every((byte) => byte === 0) ? 3 : 5;
})()
"#,
    },
    denied_console_surface!("_stderr"),
    denied_console_surface!("_stdout"),
    denied_console_surface!("write"),
    PermissionSurfaceProbe {
        name: "console[Symbol.asyncIterator]",
        source: br#"globalThis.__nimbusPermissionProbeFunction(globalThis.console?.[Symbol.asyncIterator])"#,
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
    denied_global_surface!("BroadcastChannel"),
    denied_global_surface!("EventSource"),
    denied_global_surface!("SharedWorker"),
    denied_global_surface!("WebSocket"),
    denied_global_surface!("WebSocketStream"),
    denied_global_surface!("Worker"),
    denied_global_surface!("alert"),
    denied_global_surface!("confirm"),
    denied_global_surface!("fetch"),
    denied_global_surface!("gc"),
    denied_global_surface!("postMessage"),
    denied_global_surface!("prompt"),
    denied_global_surface!("reportError"),
    denied_global_surface!("setImmediate"),
    denied_global_surface!("setInterval"),
    denied_global_surface!("setTimeout"),
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
    if bun_jsc::call_false_is_throw(global, || unsafe {
        Bun__embedderApplyNativePermissionDenyProfile(
            global as *const JSGlobalObject as *mut JSGlobalObject,
        )
    })
    .is_err()
    {
        global.clear_exception();
        return 105;
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
    let heap_before_load = vm.heap_size();

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
        if vm
            .wait_for_promise(AnyPromise::Normal(invocation_promise))
            .is_err()
        {
            return 185;
        }

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

    let heap_after_load = vm.heap_size();
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
        vm.heap_size()
    };

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
    eprintln!("  growth_assertion: full_gc_live_size");
    eprintln!("  pressure_signal: vm.heap_size_and_sync_gc");
    eprintln!("  safe_first_policy: fresh_vm_or_discard_on_pressure");

    // Heap::size() counts currently marked cells. Before a collection, its
    // value can stay flat while JSC allocates into unmarked cells. Compare two
    // completed full collections so this assertion has the same meaning on
    // every supported platform.
    if heap_retained_after_gc <= heap_after_setup_gc {
        return 192;
    }

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

    // node:vm is the module that would let guest code build a
    // NodeVMGlobalObject (whose module loader is a distinct hook from the
    // main global's). Prove it is unreachable to untrusted guests: dynamic
    // import of it is denied by the same embedder gate. This keeps the
    // NodeVMGlobalObject deny gate (defense-in-depth) anchored to a proven
    // guest-facing property — the vm module cannot be obtained here.
    let node_vm_module_status = match evaluate_dynamic_import_status(
        vm,
        global,
        br#"
import("node:vm").then(
  () => 7,
  (error) => String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 6
)
"#,
        b"nimbus-bun-embed-probe-dynamic-import-node-vm.js",
        247,
        248,
        249,
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

        let generated_node_builtin_status = match evaluate_number_or_promise(
            vm,
            global,
            br#"
nodeBuiltinModule("node:fs").then(
  () => 7,
  (error) => String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 6
)
"#,
            b"nimbus-bun-embed-probe-generated-node-builtin-status.js",
            219,
            220,
            223,
        ) {
            Ok(status) => status,
            Err(status) => return status,
        };

        let generated_external_package_status = match evaluate_number_or_promise(
            vm,
            global,
            br#"
nodeExternalPackage("left-pad").then(
  () => 7,
  (error) => String(error && error.message || error).includes("Bun embedder denied module resolution") ? 8 : 6
)
"#,
            b"nimbus-bun-embed-probe-generated-external-package-status.js",
            221,
            222,
            224,
        ) {
            Ok(status) => status,
            Err(status) => return status,
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
    eprintln!(
        "  node_vm_module_import: {}",
        module_policy_status_name(node_vm_module_status)
    );
    eprintln!("  require: {}", module_policy_status_name(require_status));
    eprintln!("  require.resolve: unreachable_with_require");
    eprintln!("  import.meta.resolve: unreachable_with_static_esm");
    eprintln!("  node_vm_global: unreachable_with_node_vm_module");
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
        "  generated_node_builtin_helper: {}",
        module_policy_status_name(generated_node_builtin_status)
    );
    eprintln!(
        "  generated_external_package_helper: {}",
        module_policy_status_name(generated_external_package_status)
    );
    eprintln!("  selected_next_lane: program_wrapper");
    eprintln!("  resolver_policy_hook: native_embedder_deny_all");
    eprintln!("  required_resolver_api: nimbus_owned_bun_package_resolver");

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
    if node_vm_module_status != 8 {
        return 247;
    }
    if bun_resolve_status != 8 || bun_resolve_sync_status != 8 {
        return 244;
    }
    if native_addon_resolve_sync_status != 8 {
        return 245;
    }
    if generated_node_builtin_status != 8 || generated_external_package_status != 8 {
        return 246;
    }

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
        if vm.wait_for_promise(AnyPromise::Normal(promise)).is_err() {
            return Err(promise_rejected_status);
        }

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
const CANCELLATION_PROOF_ACK_TIMEOUT: Duration = Duration::from_secs(30);

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
        // Materialize the sentinel on the owning thread before a cancellation
        // thread can request termination at a JSC safepoint.
        let _ = vm.jsc_vm().termination_exception();

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
    eprintln!("  normal_completion_before_cancel: retained_invocations_ok");
    eprintln!("  promise_microtask_progress: async_host_bridge_ok");
    eprintln!("  teardown_loop: fresh_vm_create_invoke_destroy_ok");
    eprintln!("  retained_vm_post_cancel_invocation: ok");
    eprintln!("  retained_vm_reuse: trusted_generated_wrapper_ok");
    eprintln!("  product_first_policy: fresh_vm_or_discard_with_outer_quota_required");

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
    if vm.wait_for_promise(AnyPromise::Normal(promise)).is_err() {
        return Err(status_base + 2);
    }

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

fn evaluate_cancel_before_guest_entry(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
) -> Result<(), i32> {
    let token = ProofCancellationToken::new();
    token.cancel();

    match evaluate_program_with_entry_gate(
        global,
        b"globalThis.__nimbusPreEntryCancelRan = true; 1",
        b"nimbus-bun-embed-probe-before-entry-cancel.js",
        65,
        &token,
        66,
    ) {
        Err(66) => {}
        Err(status) => return Err(status),
        Ok(_) => return Err(69),
    }

    let script_did_not_run = {
        let _lock = ProbeApiLock::new(vm.jsc_vm());
        evaluate_program(
            global,
            br#"typeof globalThis.__nimbusPreEntryCancelRan === "undefined" ? 1 : 0"#,
            b"nimbus-bun-embed-probe-before-entry-cancel-state.js",
            70,
        )?
    };
    if !script_did_not_run.is_number() || script_did_not_run.as_number() as i32 != 1 {
        return Err(71);
    }

    Ok(())
}

fn evaluate_generated_spin_with_deadline_timeout(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
) -> Result<(), i32> {
    let spin_entry = Arc::new(SpinEntrySignal::default());
    let deadline_fired = Arc::new(AtomicBool::new(false));
    let deadline_ack_observed = Arc::new(AtomicBool::new(false));
    let jsc_vm_ptr = core::ptr::from_ref(vm.jsc_vm()) as usize;
    let spin_entry_for_thread = Arc::clone(&spin_entry);
    let deadline_for_thread = Arc::clone(&deadline_fired);
    let deadline_ack_for_thread = Arc::clone(&deadline_ack_observed);
    let deadline = thread::spawn(move || {
        bun_core::StackCheck::configure_thread();
        match spin_entry_for_thread.wait(CANCELLATION_PROOF_ACK_TIMEOUT) {
            SpinEntryWait::Entered => {
                deadline_ack_for_thread.store(true, Ordering::SeqCst);
            }
            SpinEntryWait::Completed => return,
            SpinEntryWait::TimedOut => {}
        }
        deadline_for_thread.store(true, Ordering::SeqCst);
        // SAFETY: the proof joins this thread before the VM can be torn down.
        let jsc_vm = unsafe { &*(jsc_vm_ptr as *const VM) };
        jsc_vm.notify_need_termination();
    });

    let spin_evaluation = {
        let _spin_entry_guard = SpinEntrySignalGuard::install(Arc::clone(&spin_entry));
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
            let _stopped_by_deadline = vm.wait_for_promise(AnyPromise::Normal(promise)).is_err();
            if !vm.jsc_vm().has_termination_request() && !global.has_exception() {
                return Err(45);
            }
        }
        Ok(())
    });
    spin_entry.complete();
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
    if vm.jsc_vm().has_termination_request() {
        return Err(64);
    }
    Ok(())
}

fn evaluate_generated_spin_with_external_cancel(
    vm: &mut VirtualMachine,
    global: &JSGlobalObject,
) -> Result<(), i32> {
    let spin_entry = Arc::new(SpinEntrySignal::default());
    let cancel_fired = Arc::new(AtomicBool::new(false));
    let cancel_ack_observed = Arc::new(AtomicBool::new(false));
    let jsc_vm_ptr = core::ptr::from_ref(vm.jsc_vm()) as usize;
    let spin_entry_for_thread = Arc::clone(&spin_entry);
    let cancel_for_thread = Arc::clone(&cancel_fired);
    let cancel_ack_for_thread = Arc::clone(&cancel_ack_observed);
    let canceller = thread::spawn(move || {
        bun_core::StackCheck::configure_thread();
        match spin_entry_for_thread.wait(CANCELLATION_PROOF_ACK_TIMEOUT) {
            SpinEntryWait::Entered => {
                cancel_ack_for_thread.store(true, Ordering::SeqCst);
            }
            SpinEntryWait::Completed => return,
            SpinEntryWait::TimedOut => {}
        }
        cancel_for_thread.store(true, Ordering::SeqCst);
        // SAFETY: the proof joins this thread before the VM can be torn down.
        let jsc_vm = unsafe { &*(jsc_vm_ptr as *const VM) };
        jsc_vm.notify_need_termination();
    });

    let spin_evaluation = {
        let _spin_entry_guard = SpinEntrySignalGuard::install(Arc::clone(&spin_entry));
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
            let _stopped_by_cancellation =
                vm.wait_for_promise(AnyPromise::Normal(promise)).is_err();
            if !vm.jsc_vm().has_termination_request() && !global.has_exception() {
                return Err(54);
            }
        }
        Ok(())
    });
    spin_entry.complete();
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

fn evaluate_program_with_entry_gate(
    global: &JSGlobalObject,
    source: &[u8],
    filename: &[u8],
    exception_status: i32,
    token: &ProofCancellationToken,
    cancelled_status: i32,
) -> Result<JSValue, i32> {
    if token.is_cancelled() {
        return Err(cancelled_status);
    }
    evaluate_program(global, source, filename, exception_status)
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
        if is_termination_signal(exception)
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

fn is_termination_signal(exception: JSValue) -> bool {
    exception.is_termination_exception()
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
pub fn nimbus_bun_embed_host_bridge_call_json(
    global: &JSGlobalObject,
    frame: &CallFrame,
) -> JsResult<JSValue> {
    let request = frame.argument(0);
    if !request.is_string() {
        return Ok(host_bridge_response_json_to_js(
            global,
            br#"{"status":"error","error":{"code":"invalid_host_bridge_request","message":"request must be a JSON string"}}"#,
        ));
    }

    let request = request.to_bun_string(global)?;
    let request = request.to_utf8();
    let request = request.slice();
    let mut output = vec![0_u8; NIMBUS_HOST_BRIDGE_OUTPUT_CAP];
    let mut output_len = 0_usize;

    let mut status = CURRENT_HOST_BRIDGE_INVOCATION.with(|slot| {
        let Some(invocation) = slot.get() else {
            return 311;
        };
        // SAFETY: Nimbus installs the callback only for the duration of one
        // synchronous VM invocation. Pointers are borrowed for this call and
        // the callback must copy any response before returning.
        unsafe {
            (invocation.call_json)(
                invocation.context,
                request.as_ptr(),
                request.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut output_len,
            )
        }
    });

    if status == 307 {
        if output_len <= output.len() {
            return Ok(host_bridge_response_json_to_js(
                global,
                br#"{"status":"error","error":{"code":"invalid_host_bridge_response_length","message":"host bridge reported overflow without a larger response length"}}"#,
            ));
        }
        let required_len = output_len;
        let mut completed = Vec::new();
        if completed.try_reserve_exact(required_len).is_err() {
            return Ok(host_bridge_response_json_to_js(
                global,
                br#"{"status":"error","error":{"code":"host_bridge_response_allocation_failed","message":"host bridge response could not be retained within the runtime memory budget"}}"#,
            ));
        }
        completed.resize(required_len, 0);
        output_len = 0;
        status = CURRENT_HOST_BRIDGE_INVOCATION.with(|slot| {
            let Some(invocation) = slot.get() else {
                return 311;
            };
            // SAFETY: ABI 3 uses a null, zero-length request to take the
            // response retained by the first callback without another host
            // operation. The output buffer remains borrowed for this call.
            unsafe {
                (invocation.call_json)(
                    invocation.context,
                    core::ptr::null(),
                    0,
                    completed.as_mut_ptr(),
                    completed.len(),
                    &mut output_len,
                )
            }
        });
        if status == 0 && output_len != required_len {
            return Ok(host_bridge_response_json_to_js(
                global,
                br#"{"status":"error","error":{"code":"host_bridge_response_length_changed","message":"host bridge changed the completed response length during retrieval"}}"#,
            ));
        }
        output = completed;
    }

    if status != 0 {
        return Ok(host_bridge_response_json_to_js(
            global,
            host_bridge_status_error_response(status).as_bytes(),
        ));
    }
    if output_len > output.len() {
        return Ok(host_bridge_response_json_to_js(
            global,
            br#"{"status":"error","error":{"code":"host_bridge_response_too_large","message":"host bridge response exceeded the ABI output buffer"}}"#,
        ));
    }

    Ok(host_bridge_response_json_to_js(
        global,
        &output[..output_len],
    ))
}

const NIMBUS_HOST_BRIDGE_OUTPUT_CAP: usize = 4 * 1024 * 1024;

fn host_bridge_response_json_to_js(global: &JSGlobalObject, response: &[u8]) -> JSValue {
    EncodedSlice::utf8(response).to_js(global)
}

fn host_bridge_status_error_response(status: i32) -> String {
    format!(
        r#"{{"status":"error","error":{{"code":"host_bridge_callback_failed","status":{status}}}}}"#
    )
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
    CURRENT_SPIN_ENTRY_SIGNAL.with(|slot| {
        if let Some(signal) = slot.borrow().as_ref() {
            signal.acknowledge();
        }
    });
    Ok(JSValue::js_number_from_int32(1))
}
