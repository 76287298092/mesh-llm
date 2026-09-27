//! Canonical content identity for an `.mspec` logical inventory.

use super::schema::Directory;
use sha2::{Digest, Sha256};

const DOMAIN: &[u8] = b"mesh-specialize/mspec/weights/v1\0";

pub(super) fn calculate(directory: &Directory) -> String {
    let mut digest = Sha256::new();
    digest.update(DOMAIN);
    digest.update(directory.schema_version.to_le_bytes());
    update_string(&mut digest, &directory.identity.model_id);
    update_string(&mut digest, &directory.source.repository);
    update_string(&mut digest, &directory.source.revision);
    update_string(&mut digest, &directory.recipe_sha256);
    digest.update((directory.objects.len() as u64).to_le_bytes());

    for object in &directory.objects {
        update_string(&mut digest, &object.name);
        update_string(&mut digest, object.kind.as_str());
        update_string(&mut digest, object.dtype.as_str());
        update_string(&mut digest, &object.layout);
        digest.update((object.shape.len() as u64).to_le_bytes());
        for dimension in &object.shape {
            digest.update(dimension.to_le_bytes());
        }
        digest.update(object.length.to_le_bytes());
        update_string(&mut digest, &object.sha256);
    }

    format!("sha256:{}", hex::encode(digest.finalize()))
}

fn update_string(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::calculate;
    use crate::artifact::schema::{DType, Directory, Object, ObjectKind, SourceCheckpoint};
    use mesh_llm_native_runtime::model_identity::ModelIdentity;

    fn repeated_digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn directory() -> Directory {
        let recipe_sha256 = repeated_digest('b');
        let mut directory = Directory {
            schema_version: 1,
            identity: ModelIdentity {
                model_id: "model-a".to_string(),
                weights_id: "pending-identity".to_string(),
            },
            recipe_sha256: recipe_sha256.clone(),
            source: SourceCheckpoint {
                repository: "https://example.invalid/model-a".to_string(),
                revision: "a".repeat(40),
            },
            objects: vec![
                Object {
                    name: "config.json".to_string(),
                    kind: ObjectKind::Config,
                    dtype: DType::U8,
                    shape: vec![3],
                    layout: "raw-v1".to_string(),
                    offset: 0,
                    length: 3,
                    sha256: repeated_digest('a'),
                },
                Object {
                    name: "recipe.json".to_string(),
                    kind: ObjectKind::Recipe,
                    dtype: DType::U8,
                    shape: vec![64],
                    layout: "raw-v1".to_string(),
                    offset: 256,
                    length: 64,
                    sha256: recipe_sha256,
                },
                Object {
                    name: "weights/layer-000".to_string(),
                    kind: ObjectKind::Tensor,
                    dtype: DType::U4,
                    shape: vec![3],
                    layout: "opaque-block-layout-v7".to_string(),
                    offset: 512,
                    length: 2,
                    sha256: repeated_digest('c'),
                },
            ],
        };
        directory.identity.weights_id = calculate(&directory);
        directory
    }

    fn assert_hash_changes(directory: &Directory, mutate: impl FnOnce(&mut Directory)) {
        let original = calculate(directory);
        let mut changed = directory.clone();
        mutate(&mut changed);
        assert_ne!(calculate(&changed), original);
    }

    #[test]
    fn digest_is_deterministic_and_has_canonical_prefix_and_length() {
        let directory = directory();
        let first = calculate(&directory);
        let second = calculate(&directory);

        assert_eq!(first, second);
        // Independently hashed from the documented domain and little-endian field encoding.
        assert_eq!(
            first,
            "sha256:04a341e6de69e56815fd2920f0a8b300baf300c150186d372f7e2ec8688fb726"
        );
        assert!(first.starts_with("sha256:"));
        assert_eq!(first.len(), 7 + 64);
        assert!(
            first[7..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
    }

    #[test]
    fn adjacent_strings_are_length_prefixed() {
        let mut left = directory();
        left.identity.model_id = "ab".to_string();
        left.source.repository = "c".to_string();
        let mut right = directory();
        right.identity.model_id = "a".to_string();
        right.source.repository = "bc".to_string();

        assert_ne!(calculate(&left), calculate(&right));
    }

    #[test]
    fn weights_identity_and_payload_offsets_are_excluded() {
        let directory = directory();
        let original = calculate(&directory);
        let mut changed = directory.clone();
        changed.identity.weights_id = "a different claimed identity".to_string();
        for object in &mut changed.objects {
            object.offset = object.offset.saturating_add(37);
        }

        assert_eq!(calculate(&changed), original);
    }

    #[test]
    fn every_directory_and_object_binding_field_changes_the_digest() {
        let directory = directory();
        // These mutations test hash binding only; some intentionally violate
        // layout/schema constraints and are not passed to Directory::validate.
        assert_hash_changes(&directory, |changed| changed.schema_version += 1);
        assert_hash_changes(&directory, |changed| {
            changed.identity.model_id.push_str("-other")
        });
        assert_hash_changes(&directory, |changed| {
            changed.source.repository.push_str("/other")
        });
        assert_hash_changes(&directory, |changed| {
            changed.source.revision.replace_range(..1, "f")
        });
        assert_hash_changes(&directory, |changed| {
            changed.recipe_sha256.replace_range(..1, "f")
        });
        assert_hash_changes(&directory, |changed| {
            changed.objects[0].name.push_str(".bak")
        });
        assert_hash_changes(&directory, |changed| {
            changed.objects[0].kind = ObjectKind::Tokenizer
        });
        assert_hash_changes(&directory, |changed| changed.objects[0].dtype = DType::I8);
        assert_hash_changes(&directory, |changed| changed.objects[0].shape[0] += 1);
        assert_hash_changes(&directory, |changed| changed.objects[0].shape.push(2));
        assert_hash_changes(&directory, |changed| {
            changed.objects[0].layout.push_str("-other")
        });
        assert_hash_changes(&directory, |changed| changed.objects[0].length += 1);
        assert_hash_changes(&directory, |changed| {
            changed.objects[0].sha256 = repeated_digest('d')
        });
        assert_hash_changes(&directory, |changed| changed.objects.swap(0, 1));
        assert_hash_changes(&directory, |changed| {
            changed.objects.push(changed.objects[0].clone())
        });
    }
}
