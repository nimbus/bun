/**
 * Regression tests for ninja ordering of post-link steps (strip, smoke test,
 * dsymutil) in scripts/build/bun.ts.
 *
 * The smoke_test and dsymutil rule commands are wrapped through
 * `cfg.jsRuntime` (= process.execPath). When `bun` on PATH resolves inside the
 * build directory, that path is the strip output itself (build/release/bun),
 * and without an ordering edge ninja will run strip and the wrapper exec
 * concurrently, failing with "Permission denied" on the half-written file.
 *
 * These exercise the ninja-emission logic only (no compiler or ninja needed),
 * so they run on every host.
 */
import { describe, expect, test } from "bun:test";
import { isMacOS, tempDir } from "harness";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { emitPostLink } from "../../scripts/build/bun.ts";
import {
  resolveConfig,
  type BuildMode,
  type Config,
  type PartialConfig,
  type Toolchain,
} from "../../scripts/build/config.ts";
import { Ninja } from "../../scripts/build/ninja.ts";

/** A fully-populated fake toolchain; resolveConfig never spawns any of these. */
function mockToolchain(overrides: Partial<Toolchain> = {}): Toolchain {
  return {
    cc: "/fake/llvm/bin/clang",
    cxx: "/fake/llvm/bin/clang++",
    hostCc: undefined,
    hostCxx: undefined,
    clangVersion: "21.1.8",
    clangResourceDir: "/fake/llvm/lib/clang/21",
    ar: "/fake/llvm/bin/llvm-ar",
    ranlib: "/fake/llvm/bin/llvm-ranlib",
    ld: "/fake/llvm/bin/ld.lld",
    ld64Lld: "/fake/llvm/bin/ld64.lld",
    rustLld: undefined,
    rustLlvmVersion: "22.1.4",
    strip: "/fake/bin/strip",
    llvmStrip: "/fake/llvm/bin/llvm-strip",
    nm: "/fake/llvm/bin/llvm-nm",
    dsymutil: "/fake/llvm/bin/dsymutil",
    bun: "/fake/bin/bun",
    jsRuntime: "/fake/bin/bun",
    esbuild: "/fake/bin/esbuild",
    ccache: undefined,
    cmake: "/fake/bin/cmake",
    cargo: undefined,
    cargoHome: undefined,
    rustupHome: undefined,
    msvcLinker: undefined,
    rc: undefined,
    mt: undefined,
    nasm: undefined,
    ...overrides,
  };
}

/**
 * Resolve a host-targeted config: no os/arch override, so `canRunOnHost` is
 * true and the smoke_test rule emits the real edge (not the phony short-circuit).
 */
function hostConfig(partial: PartialConfig, buildDir: string): Config {
  return resolveConfig(
    { buildDir, ...partial },
    // jsRuntime = the strip output: what resolveToolchain() produces when
    // `bun` on PATH resolves into build/release/.
    mockToolchain({ jsRuntime: join(buildDir, "bun") }),
  );
}

/** Find one build-edge line in the generated ninja text (continuations unwrapped). */
function buildEdge(ninja: string, rule: string): string {
  const flat = ninja.replace(/ \$\n +/g, " ");
  const line = flat.split("\n").find(l => l.startsWith("build ") && l.includes(`: ${rule} `));
  if (line === undefined) throw new Error(`no '${rule}' edge in ninja output:\n${ninja}`);
  return line;
}

describe("emitPostLink ninja ordering", () => {
  test("release smoke_test is ordered after strip", () => {
    using dir = tempDir("build-post-link", {});
    const buildDir = String(dir);
    const cfg = hostConfig({ buildType: "Release" }, buildDir);
    expect(cfg.canRunOnHost).toBe(true);

    const n = new Ninja({ buildDir });
    const exe = resolve(buildDir, `bun-profile${cfg.exeSuffix}`);
    const { strippedExe } = emitPostLink(n, cfg, exe, "bun-profile", []);
    const out = n.toString();

    expect(strippedExe).toBe(resolve(buildDir, `bun${cfg.exeSuffix}`));
    // strip writes `bun`; the smoke_test wrapper execs cfg.jsRuntime
    // (= `bun` here). Without `|| bun` ninja schedules them concurrently
    // and the wrapper sees a half-written file.
    expect(buildEdge(out, "smoke_test")).toBe(
      `build bun-profile.smoke-test-passed: smoke_test bun-profile${cfg.exeSuffix} || bun${cfg.exeSuffix}`,
    );
    expect(buildEdge(out, "strip")).toBe(`build bun${cfg.exeSuffix}: strip bun-profile${cfg.exeSuffix}`);
  });

  test("debug smoke_test has no strip dep (nothing to order against)", () => {
    using dir = tempDir("build-post-link", {});
    const buildDir = String(dir);
    const cfg = hostConfig({ buildType: "Debug", assertions: true }, buildDir);

    const n = new Ninja({ buildDir });
    const exe = resolve(buildDir, `bun-debug${cfg.exeSuffix}`);
    const { strippedExe, dsym } = emitPostLink(n, cfg, exe, "bun-debug", []);
    const out = n.toString();

    expect({ strippedExe, dsym }).toEqual({ strippedExe: undefined, dsym: undefined });
    expect(buildEdge(out, "smoke_test")).toBe(
      `build bun-debug.smoke-test-passed: smoke_test bun-debug${cfg.exeSuffix}`,
    );
    expect(buildEdge(out, "phony")).toBe(`build bun: phony bun-debug${cfg.exeSuffix}`);
  });

  // Cross-config path only: on macOS, resolveConfig({ os: "darwin" }) probes
  // xcode-select for the real SDK, which belongs to the native test above.
  // The ordering logic is identical to the smoke_test case.
  test.skipIf(isMacOS)("darwin release dsymutil is ordered after strip", () => {
    using dir = tempDir("build-post-link", {});
    const buildDir = String(dir);
    const cfg = resolveConfig({ os: "darwin", arch: "aarch64", buildType: "Release", buildDir }, mockToolchain());
    expect(cfg.canRunOnHost).toBe(false);

    const n = new Ninja({ buildDir });
    const exe = resolve(buildDir, "bun-profile");
    const { dsym } = emitPostLink(n, cfg, exe, "bun-profile", []);
    const out = n.toString();

    expect(dsym).toBe(resolve(buildDir, "bun-profile.dSYM"));
    expect(buildEdge(out, "dsymutil")).toBe("build bun-profile.dSYM: dsymutil bun-profile || bun");
    // Cross-compile: smoke_test short-circuits to a `check` phony (the
    // binary can't run on this host), so the strip race can't happen there.
    expect(buildEdge(out, "phony")).toBe("build check: phony bun-profile");
  });
});

describe("Nimbus embedder build contract", () => {
  function linuxEmbedderConfig(mode: BuildMode): Config {
    using dir = tempDir("build-embedder-config", {});
    const buildDir = String(dir);
    return resolveConfig(
      {
        os: "linux",
        arch: "x64",
        abi: "gnu",
        buildType: "Release",
        buildDir,
        linuxSysroot: buildDir,
        webkit: "local",
        embedderShared: true,
        mode,
      },
      mockToolchain(),
    );
  }

  test("shared adapter is accepted only in modes that emit its target", () => {
    expect([linuxEmbedderConfig("full").mode, linuxEmbedderConfig("archive-link").mode]).toEqual([
      "full",
      "archive-link",
    ]);

    for (const mode of ["cpp-only", "rust-only", "link-only", "rust-and-link"] as const) {
      expect(() => linuxEmbedderConfig(mode)).toThrow(`--embedder-shared cannot be used with --mode=${mode}`);
    }
  });

  test("generated POSIX smoke driver maps native statuses to portable nonzero exits", () => {
    const source = readFileSync(resolve(import.meta.dir, "..", "..", "scripts", "build", "bun.ts"), "utf8");
    const start = source.indexOf("function emitEmbedProbeTarget");
    const end = source.indexOf("function emitEmbedderSharedTarget", start);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    const driverGenerator = source.slice(start, end);

    expect(driverGenerator).toContain("return status > 0 && status <= 255 ? status : 1;");
    expect(driverGenerator).not.toContain('"  if (status != 0) return status;"');
    expect(driverGenerator).not.toContain('"  if (status != 300) return 256;"');
    expect(driverGenerator).not.toContain('"  if (status != 300) return 257;"');
  });

  test("native probe covers pre-execution request validation and completed-response retrieval", () => {
    const source = readFileSync(resolve(import.meta.dir, "..", "..", "scripts", "build", "bun.ts"), "utf8");
    const start = source.indexOf("function emitEmbedProbeTarget");
    const end = source.indexOf("function emitEmbedderSharedTarget", start);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    const driverGenerator = source.slice(start, end);

    expect(driverGenerator).toContain("invalid_request_side_effect_bundle");
    expect(driverGenerator).toContain("nimbus_bun_embed_driver_host_calls.load() != 0");
    expect(driverGenerator).toContain("nimbus_bun_embed_take_pending_response");
    expect(driverGenerator).toContain("status != 307 || retry_len != pending_len");
    expect(driverGenerator).toContain("nimbus_bun_embed_driver_host_calls.load() != 2");
    expect(driverGenerator).toContain("nimbus_bun_embed_driver_overflow_host_bridge");
    expect(driverGenerator).toContain("nimbus_bun_embed_driver_overflow_host_calls.load() != 1");
  });

  test("probe archive tracks its embedded JavaScript bundle", () => {
    const source = readFileSync(resolve(import.meta.dir, "..", "..", "scripts", "build", "bun.ts"), "utf8");
    const start = source.indexOf('packageName: "bun_embed_probe"');
    expect(start).toBeGreaterThanOrEqual(0);
    const archiveInputs = source.slice(Math.max(0, start - 700), start);

    expect(archiveInputs).toContain(
      'embeddedSources: [resolve(cfg.cwd, "src/embed_probe/nimbus_generated_program_bundle.js")]',
    );
  });

  test("generated wrapper never sends a guest-held HTTP route plan to the host", () => {
    const source = readFileSync(
      resolve(import.meta.dir, "..", "..", "src", "embed_probe", "nimbus_generated_program_bundle.js"),
      "utf8",
    );

    expect(source).toContain('__nimbusAsyncHostValue("op_nimbus_http_route", {');
    expect(source).not.toMatch(/op_nimbus_http_route[\s\S]{0,120}\n\s*route,/);
  });
});
