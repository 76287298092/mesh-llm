//! Opaque identifiers used after a caller verifies a resident model artifact.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Exact model and weight identity; constructing this value does not verify files.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_id: String,
    pub weights_id: String,
}

impl ModelIdentity {
    pub fn validate(&self) -> Result<()> {
        validate_opaque_id(&self.model_id, "model_id")?;
        validate_opaque_id(&self.weights_id, "weights_id")
    }
}

fn validate_opaque_id(value: &str, field: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{field} must not be empty");
    ensure!(value.len() <= 1024, "{field} must be at most 1024 bytes");
    ensure!(
        value.trim() == value,
        "{field} must not have surrounding whitespace"
    );
    ensure!(
        !value.chars().any(char::is_control),
        "{field} must not contain control characters"
    );
    Ok(())
}

/// General runtimes accept any identity; specializations require a valid exact pair.
pub fn serves_model(serves: &[ModelIdentity], requested: Option<&ModelIdentity>) -> bool {
    if serves.is_empty() {
        return true;
    }
    let Some(requested) = requested else {
        return false;
    };
    requested.validate().is_ok()
        && serves
            .iter()
            .any(|served| served.validate().is_ok() && served == requested)
}

#[cfg(test)]
mod tests {
    use super::{ModelIdentity, serves_model};

    fn identity(model_id: &str, weights_id: &str) -> ModelIdentity {
        ModelIdentity {
            model_id: model_id.to_string(),
            weights_id: weights_id.to_string(),
        }
    }

    #[test]
    fn empty_serves_list_keeps_general_runtime_behavior() {
        assert!(serves_model(&[], None));
        assert!(serves_model(&[], Some(&identity("model", "weights"))));
    }

    #[test]
    fn nonempty_serves_list_requires_an_exact_valid_identity() {
        let served = identity("model-a", "weights-a");
        assert!(!serves_model(std::slice::from_ref(&served), None));
        assert!(serves_model(std::slice::from_ref(&served), Some(&served)));
        assert!(!serves_model(
            std::slice::from_ref(&served),
            Some(&identity("model-a", "weights-b"))
        ));
        assert!(!serves_model(
            std::slice::from_ref(&served),
            Some(&identity("model-b", "weights-a"))
        ));
        assert!(!serves_model(
            std::slice::from_ref(&served),
            Some(&identity("Model-a", "weights-a"))
        ));
    }

    #[test]
    fn invalid_ids_are_rejected_and_never_match() {
        let invalid = [
            identity("", "weights"),
            identity(" model", "weights"),
            identity("model ", "weights"),
            identity("model\u{0007}id", "weights"),
            identity("model", ""),
            identity("model", " weights"),
            identity("model", "weights\n"),
            identity("model", "weight\u{0000}s"),
            identity(&"m".repeat(1025), "weights"),
            identity("model", &"w".repeat(1025)),
        ];
        for candidate in &invalid {
            assert!(candidate.validate().is_err());
            assert!(!serves_model(
                std::slice::from_ref(candidate),
                Some(candidate)
            ));
        }
        assert!(identity(&"m".repeat(1024), "weights").validate().is_ok());
    }

    #[test]
    fn serde_roundtrip_preserves_ids_without_normalizing_them() {
        let original = identity("Qwen3.8/FP16-α", "opaque:Case-sensitive-X");
        let encoded = serde_json::to_string(&original).unwrap();
        let decoded: ModelIdentity = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, original);
        assert!(decoded.validate().is_ok());
    }
}
