//! Small standalone starter catalog. Skippy owns these references independently of Mesh.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct StarterModel {
    pub name: &'static str,
    pub reference: &'static str,
    pub size_bytes: u64,
    pub description: &'static str,
}

pub const STARTERS: &[StarterModel] = &[
    StarterModel {
        name: "Qwen3-0.6B-Q4_K_M",
        reference: "unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
        size_bytes: 397_000_000,
        description: "Small starter model for trying local serving.",
    },
    StarterModel {
        name: "Qwen3-4B-Q4_K_M",
        reference: "unsloth/Qwen3-4B-GGUF:Q4_K_M",
        size_bytes: 2_500_000_000,
        description: "Balanced local chat model.",
    },
    StarterModel {
        name: "Qwen3-8B-Q4_K_M",
        reference: "unsloth/Qwen3-8B-GGUF:Q4_K_M",
        size_bytes: 5_000_000_000,
        description: "Larger local chat model.",
    },
];

pub fn resolve(reference: &str) -> &str {
    STARTERS
        .iter()
        .find(|entry| entry.name.eq_ignore_ascii_case(reference))
        .map_or(reference, |entry| entry.reference)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starter_aliases_resolve_without_changing_hub_references() {
        assert_eq!(
            resolve("qwen3-0.6b-q4_k_m"),
            "unsloth/Qwen3-0.6B-GGUF:Q4_K_M"
        );
        assert_eq!(resolve("org/model:Q4_K_M"), "org/model:Q4_K_M");
    }
}
