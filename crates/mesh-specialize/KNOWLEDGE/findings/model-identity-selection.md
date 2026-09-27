# Exact model and weight selection

Status: H01 policy implemented; resident artifact discovery and serving integration
remain pending. This is host policy, with no GPU or performance evidence.

`mesh-llm-native-runtime::model_identity` owns opaque `model_id` and `weights_id`
values. Validation rejects empty, oversized, surrounding-whitespace and control
character values. It does not normalize case or infer a hash from a model name.
The future artifact reader must establish identity from verified resident bytes.

`NativeRuntimeArtifact.serves` is additive and defaults to an empty list for
existing manifests. An empty list retains general runtime eligibility. A nonempty
list requires an exact valid pair. The existing resolver API has no verified
requested identity and therefore rejects specializations, including explicit
runtime-ID requests. The new pure `model_selection` APIs accept a caller-verified
identity, apply the existing platform/ABI/backend checks, and rank compatible
candidates. They perform no I/O, installation or download.

Bundle and cache lookup must preserve the selected artifact's `serves` declaration.
A runtime ID/version/ABI match is insufficient if the local manifest describes
different accepted weights. Both sources now use the same identity check before
being offered as the selected source.

Focused tests cover old manifests, invalid identities, unknown or differing weights,
exact ranking, general fallback, retained ABI/platform rejection and mismatched
local sources. The broader product build and live startup routing are not yet
qualified. Existing Rust struct literals receive only `serves: Vec::new()`.

Local validation on September 27: 55 native-runtime tests, 23 hardware-profile
tests, and 51 runtime-install unit/integration tests passed. Native-runtime Clippy
passed with warnings denied. Commands used `just with-lld cargo test -p ...` and
`just with-lld cargo clippy -p mesh-llm-native-runtime --all-targets -- -D warnings`.
These focused checks are not a product build or a live specialized startup test.

The resolver is still over 1,000 lines. New model policy and its tests are extracted
into the owning `model_identity` and `model_selection` modules; the existing
backend policy and test suite remain in place. This avoids mixing artifact
discovery, download or GPU-probe responsibilities into the new pure policy.

Specialized manifests remain local/internal. Older MeshLLM releases ignore unknown
JSON fields and cannot enforce `serves`; do not put these prototype manifests into
release catalogs or caches shared with older hosts. Existing general manifests
remain unchanged on the wire.

Next gates: enforce selected-device/driver/memory eligibility (H02), verify a
resident `.mspec` before constructing the requested identity (H03), and exercise
the actual ABI/startup fallback (H04). Passing these policy tests is not proof that
the host can serve a specialized model.
