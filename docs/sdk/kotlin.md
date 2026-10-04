# Kotlin/Android SDK

Use the GitHub Packages Maven registry for `Mesh-LLM/mesh-llm`.

## Install

```text
ai.meshllm:meshllm-android:<version>
```

Configure the Maven repository:

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/Mesh-LLM/mesh-llm")
        credentials {
            username = providers.gradleProperty("gpr.user")
                .orElse(System.getenv("GITHUB_ACTOR"))
                .get()
            password = providers.gradleProperty("gpr.key")
                .orElse(System.getenv("GITHUB_TOKEN"))
                .get()
        }
    }
}
```

## Client: Public Mesh

```kotlin
import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.Client
import ai.meshllm.Event
import ai.meshllm.PublicMeshQuery
import kotlinx.coroutines.flow.collect
import uniffi.mesh_ffi.generateOwnerKeypairHex

val ownerKeypair = generateOwnerKeypairHex()
val client = Client.connectPublic(
    ownerKeypair,
    PublicMeshQuery(
        model = "Qwen3",
        minVramGb = null,
        region = null,
        targetName = null,
        relays = emptyList(),
    ),
)

client.start()
val publicModels = client.inference.listModels()
client.inference.chatFlow(
    ChatRequest(publicModels.first().id, listOf(ChatMessage("user", "Say hello from a public mesh."))),
).collect(::printToken)
client.stop()
```

## Client: Private Mesh

```kotlin
import ai.meshllm.Client
import ai.meshllm.InviteToken
import uniffi.mesh_ffi.generateOwnerKeypairHex

val ownerKeypair = generateOwnerKeypairHex()
val client = Client(InviteToken(System.getenv("MESH_PRIVATE_INVITE")), ownerKeypair)

client.start()
val models = client.inference.listModels()
client.inference.chatFlow(
    ChatRequest(models.first().id, listOf(ChatMessage("user", "Say hello from a private mesh."))),
).collect(::printToken)
client.stop()
```

## Inference Helper

```kotlin
fun printToken(event: Event) {
    if (event is Event.TokenDelta) print(event.delta)
    if (event is Event.Completed) println()
}
```

## Serving: Install Runtime

Resolve or install the native runtime before local serving:

```kotlin
import ai.meshllm.NativeRuntime
import ai.meshllm.NativeRuntimeResolveOptions
import java.io.File

val runtime = NativeRuntime.resolve(
    NativeRuntimeResolveOptions(
        artifactDir = System.getenv("MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR")?.let(::File),
        allowDownload = System.getenv("MESH_SDK_RUNTIME_ALLOW_DOWNLOAD") == "1",
    ),
)
println("using ${runtime.nativeRuntimeId} from ${runtime.path}")
```

## Serving: Public Mesh

```kotlin
import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.DevicePolicy
import ai.meshllm.InviteToken
import ai.meshllm.LoadModelOptions
import ai.meshllm.Node
import ai.meshllm.UnloadModelOptions

val ownerKeypair = generateOwnerKeypairHex()
val node = Node(InviteToken(System.getenv("MESH_PUBLIC_INVITE")), ownerKeypair)
node.start()

val modelRef = System.getenv("MESH_SDK_MODEL_REF") ?: "Qwen2.5-3B-Instruct-Q4_K_M"
node.models.download(modelRef)
val served = node.serving.load(modelRef, LoadModelOptions(DevicePolicy.Auto))
node.inference.chatFlow(
    ChatRequest(served.modelId, listOf(ChatMessage("user", "Say hello from a public serving node."))),
).collect(::printToken)
node.serving.unloadModel(served.modelId, UnloadModelOptions(drainTimeoutMs = 1_000UL, force = false))
node.stop()
```

## Serving: Private Mesh

Private mesh serving uses the same lifecycle with `MESH_PRIVATE_INVITE`:

```kotlin
val ownerKeypair = generateOwnerKeypairHex()
val node = Node(InviteToken(System.getenv("MESH_PRIVATE_INVITE")), ownerKeypair)
```

## JVM Example

```bash
scripts/package-native-runtime.sh \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-runtimes

MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR=dist/native-runtimes/meshllm-native-runtime-darwin-aarch64-metal \
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
./gradlew --no-daemon run -p sdk/kotlin/example/example-jvm
```

## Console Assets

Published Kotlin packages that advertise console support include the built web
console as JVM resources. Use the packaged resource helper in normal package
usage:

```kotlin
val options = ConsoleAssets.packagedOptions()
```

## Native build and ABI contract (maintainers)

`buildNativeLibs` in `sdk/kotlin/build.gradle.kts` is the authoritative recipe.
It targets exactly three ABIs, and those three are the contract the rest of the
tree has to agree with:

| ABI | Rust target | artifact copied to `jniLibs/<abi>/` |
|---|---|---|
| `arm64-v8a` | `aarch64-linux-android` | `libmeshllm_ffi.so` |
| `armeabi-v7a` | `armv7-linux-androideabi` | `libmeshllm_ffi.so` |
| `x86_64` | `x86_64-linux-android` | `libmeshllm_ffi.so` |

The Android API floor is `android-26` (Android 8.0), overridable with
`MESH_LLM_ANDROID_PLATFORM`. Building above it narrows the set of devices that
can load the library, which matters because the target audience is phones and
TV boxes of mixed vintages; `armeabi-v7a` in particular is what covers 32-bit
ARM boxes.

Each ABI is built in two steps, in this order.

1. **llama.cpp for that ABI**, through the NDK toolchain:

   ```bash
   LLAMA_STAGE_BACKEND=cpu \
   LLAMA_STAGE_BUILD_DIR=<repo>/.deps/llama-build/build-stage-abi-android-<abi>-cpu \
   bash scripts/build-llama.sh \
     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
     -DANDROID_ABI=<abi> -DANDROID_PLATFORM=android-26
   ```

2. **The FFI library**, which links the step-1 output:

   ```bash
   cargo ndk -t <abi> build --release -p mesh-llm-ffi \
     --no-default-features --features embedded-runtime
   ```

`cargo-ndk` is required for step 2 and is not part of the workspace toolchain;
install it before attempting an Android build.

### `embedded-runtime` is what makes the library usable

The feature is declared empty in `crates/mesh-llm-ffi/Cargo.toml`; it only gates
`cfg` branches. The actual embedded serving runtime comes from
`mesh-llm-sdk`'s `serving` feature, which the FFI crate enables unconditionally
through its dependency list. Building without `embedded-runtime` therefore
succeeds and produces a normal-looking `libmeshllm_ffi.so` that **refuses at
call time**: `crates/mesh-llm-ffi/src/node.rs` compiles in a branch returning
"this native library was built without embedded-runtime support".

That refusal string is the cheap way to tell the two builds apart. `findstr` for
it in the produced `.so` — a build that contains it is the wrong build,
regardless of how it is named or where it was copied.

### The console `dist/` must exist first

`mesh-llm-ffi` depends on `mesh-llm-sdk` with `console` enabled, and
`console` enables `mesh-llm-embedded-runtime/web-ui`, which enables
`mesh-llm-ui`. That crate's default feature is `embed-assets`, and its
`build.rs` fails the build when the assets are requested and
`crates/mesh-llm-ui/dist/index.html` is absent. Build the console before the
Android library, or step 2 fails for a reason that has nothing to do with
Android.

### NDK compiler wrapper names

The NDK spells the 32-bit ARM wrapper with an `a` that the Rust target does not
have, so a manual linker setup has to use the NDK spelling:

```
aarch64-linux-android26-clang.cmd
armv7a-linux-androideabi26-clang.cmd     <- target is armv7-linux-androideabi
x86_64-linux-android26-clang.cmd
```

### Packaged native runtimes

`scripts/package-native-runtime.sh` labels Android artifacts by ABI
(`meshllm-native-runtime-android-arm64-v8a-cpu`, and so on), and its
`target_runtime_os`/`target_platform` functions handle the triples. Two details
there are easy to get wrong and both are load-bearing:

- `target_runtime_os` must test `*android*` **before** `*linux*`, because every
  Android triple contains `linux`.
- `rewrite_linux_runtime_paths` has to guard on the resolved runtime os rather
  than on `*linux*` in the triple. Android has no rpath — the platform linker
  searches the application's own library directory — so a triple-based guard
  sends Android packaging into the `patchelf` path, which then fails.

Runtime selection does not match on the artifact id: `select-native-runtime.py`
compares the manifest's `runtime.platform.os` and `runtime.platform.arch`
against what the host reports, so Android runtimes carry `os = "android"` with
`arch` of `aarch64`, `arm`, or `x86_64`.

### The Kotlin bindings are generated, not checked in

`sdk/kotlin` imports `uniffi.mesh_ffi.*`, and that package does not exist in the
tree. It is produced from `crates/mesh-llm-ffi/src/mesh_ffi.udl` by
`sdk/kotlin/scripts/generate-kotlin-bindings.sh`, which installs
`uniffi-bindgen` 0.32.0 on demand and writes the result to
`sdk/kotlin/src/main/kotlin/uniffi/mesh_ffi/mesh_ffi.kt`. Nothing compiles until
that has been run, which is also why `jniLibs/` was empty and the module had never
been built. Note the script tests `[ -x "$BINDGEN" ]` without the `.exe` suffix,
so on Windows it never finds an installed bindgen and must be invoked directly.

### The shared library name does not match, and the AAR path ships the wrong one

The generated bindings resolve their JNA library as
`findLibraryName(componentName = "mesh_ffi")`, which returns **`uniffi_mesh_ffi`**
unless the `uniffi.component.mesh_ffi.libraryOverride` system property is set. So
the file the loader looks for on Android is **`libuniffi_mesh_ffi.so`**.

The three places that produce or consume that file disagree:

| producer / consumer | name it uses |
|---|---|
| generated UniFFI bindings (the actual loader) | `uniffi_mesh_ffi` |
| `scripts/package-native-sdk.sh` | `libuniffi_mesh_ffi.so` — correct |
| `sdk/kotlin/build.gradle.kts` (`buildNativeLibs`, `assembleAar`) | `libmeshllm_ffi.so` — the crate's `[lib] name` |

Nothing in the tree sets `libraryOverride`, so an AAR assembled by that task
packages `jni/arm64-v8a/libmeshllm_ffi.so`, which the bindings never ask for. The
failure appears on the device at first call, not at build time, which is why it
survives CI. Either the AAR task should copy the artifact under
`libuniffi_mesh_ffi.so`, or a consumer must set the override property before the
first call. The naming in `package-native-sdk.sh` is the one to follow.

### The FFI library does not contain llama — the runtime is resolved at run time

Measured, not inferred. `libmeshllm_ffi.so` built for `aarch64-linux-android`
reports exactly three `DT_NEEDED` entries — `libdl.so`, `libm.so`, `libc.so` —
and no `libllama.so` or `libggml*.so` of any kind. The runtime is opened with
`libloading::Library::new` from a resolved runtime artifact directory, and a
native runtime package on this platform is `lib/libllama.so` plus its ggml
libraries.

Two consequences follow, and the second is a defect in the recipe above.

1. **A per-ABI shared llama runtime is required on Android**, packaged like any
   other platform's runtime (`meshllm-native-runtime-android-<abi>-cpu`). It is
   not optional and it is not embedded in the FFI library.

2. **The llama step above builds the wrong thing.** It runs `build-llama.sh`
   without setting `LLAMA_STAGE_LINK_MODE`, so the link mode stays at the
   script's `static` default and the step produces `.a` archives in
   `build-stage-abi-android-<abi>-cpu`. Nothing in the FFI link closure reads
   them: the FFI linked successfully with no Android llama build present at all,
   which is only possible because it does not link llama. Passing
   `LLAMA_STAGE_LINK_MODE=dynamic` produces the `libllama.so` set the runtime
   resolution actually needs.

Until that is fixed, an Android device can build and install the SDK but cannot
serve: the library will fail to resolve a runtime that was never produced.
