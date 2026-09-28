mod binding;
mod inventory;
mod plane_layout;
mod planes;
mod proposal;

use self::{
    binding::{MatrixSelection, interleave_query_gate, selected_rows},
    inventory::{
        Projections, select_norms as select_norm_views, validate_component_config,
        validate_mtp_inventory, validate_proposal_inventory,
    },
    planes::{q8_projection, q8_view},
    proposal::{TOKEN_MAP_BYTES, q4_projection, select_token_map, validate_distinct_target_head},
};
use super::{NativeMtpNormViews, ProposalTokenMap, Q4MatrixView, Q8MatrixView};
use crate::artifact::ninfer::Directory;
use anyhow::{Result, ensure};
use std::collections::BTreeSet;

pub(super) struct Selected {
    pub fc: Q8MatrixView,
    pub query_gate: Q8MatrixView,
    pub key: Q8MatrixView,
    pub value: Q8MatrixView,
    pub attention_output: Q8MatrixView,
    pub mlp_gate: Q8MatrixView,
    pub mlp_up: Q8MatrixView,
    pub mlp_down: Q8MatrixView,
    pub norms: NativeMtpNormViews,
    pub proposal_head: Q4MatrixView,
    pub token_map_object: String,
}

pub(super) fn plan(directory: &Directory) -> Result<Selected> {
    validate_component_config(directory)?;
    let projections = select_projections(directory)?;
    let norms = select_norms(directory)?;
    let proposal_binding_count = directory
        .bindings
        .keys()
        .filter(|name| name.starts_with("proposal/"))
        .count();
    ensure!(
        proposal_binding_count == 2,
        "native proposal binding count mismatch"
    );
    let mtp_binding_count = directory
        .bindings
        .keys()
        .filter(|name| name.starts_with("mtp/"))
        .count();
    ensure!(mtp_binding_count == 16, "native MTP binding count mismatch");
    let (proposal_head, token_map_object) = select_proposal(directory)?;
    validate_mtp_inventory(directory, &projections, &norms)?;
    validate_proposal_inventory(directory, &proposal_head, &token_map_object)?;
    validate_view_object_selection(
        directory,
        &projections,
        &norms,
        &proposal_head,
        &token_map_object,
    )?;
    Ok(Selected {
        fc: projections.fc,
        query_gate: projections.query_gate,
        key: projections.key,
        value: projections.value,
        attention_output: projections.attention_output,
        mlp_gate: projections.mlp_gate,
        mlp_up: projections.mlp_up,
        mlp_down: projections.mlp_down,
        norms,
        proposal_head,
        token_map_object,
    })
}

fn validate_view_object_selection(
    directory: &Directory,
    projections: &Projections,
    norms: &NativeMtpNormViews,
    proposal_head: &Q4MatrixView,
    token_map_object: &str,
) -> Result<()> {
    let mut selected = BTreeSet::new();
    for matrix in [
        &projections.fc,
        &projections.query_gate,
        &projections.key,
        &projections.value,
        &projections.attention_output,
        &projections.mlp_gate,
        &projections.mlp_up,
        &projections.mlp_down,
    ] {
        selected.insert(matrix.object_id.as_str());
    }
    for norm in norms.iter() {
        selected.insert(norm.object_id.as_str());
    }
    selected.insert(&proposal_head.object_id);
    selected.insert(token_map_object);
    let bound = bound_object_ids(directory)?;
    ensure!(
        selected == bound,
        "native MTP/proposal view selection does not account for every bound object"
    );
    ensure!(
        selected.len() == 14,
        "native MTP/proposal selected object count mismatch"
    );
    Ok(())
}

fn bound_object_ids(directory: &Directory) -> Result<BTreeSet<&str>> {
    directory
        .bindings
        .iter()
        .filter(|(name, _)| name.starts_with("mtp/") || name.starts_with("proposal/"))
        .flat_map(|(_, binding)| match binding {
            crate::artifact::ninfer::Binding::Object { object } => vec![object.as_str()],
            crate::artifact::ninfer::Binding::Parts { parts } => {
                parts.iter().map(|part| part.object.as_str()).collect()
            }
        })
        .try_fold(BTreeSet::new(), |mut names, object| {
            ensure!(
                directory.objects.iter().any(|record| record.id == object),
                "native MTP binding references missing source object"
            );
            names.insert(object);
            Ok(names)
        })
}

fn select_projections(directory: &Directory) -> Result<Projections> {
    let fc = q8_projection(directory, "mtp/input_projection", [5_120, 10_240])?;
    let query_rows = selected_rows(
        directory,
        MatrixSelection {
            binding: "mtp/layers/0/attention/query",
            shape: [6_144, 5_120],
        },
    )?;
    let gate_rows = selected_rows(
        directory,
        MatrixSelection {
            binding: "mtp/layers/0/attention/gate",
            shape: [6_144, 5_120],
        },
    )?;
    ensure!(
        query_rows.object.id == gate_rows.object.id,
        "native MTP query/gate views must share one physical parent"
    );
    ensure!(
        query_rows.object.format.as_deref() == Some("q8_g32_fp16")
            && query_rows.object.shape.as_slice() == [14_336, 5_120],
        "native MTP Q/K/gate/V parent metadata mismatch"
    );
    ensure!(
        query_rows.rows.start == 0
            && query_rows.rows.end == 6_144
            && gate_rows.rows.start == 7_168
            && gate_rows.rows.end == 13_312,
        "native MTP query/gate source ranges differ from the checked parent rows"
    );
    let query_gate = q8_view(
        query_rows.object,
        [12_288, 5_120],
        interleave_query_gate(&query_rows, &gate_rows)?,
    )?;
    let key_rows = selected_rows(
        directory,
        MatrixSelection {
            binding: "mtp/layers/0/attention/key",
            shape: [1_024, 5_120],
        },
    )?;
    let value_rows = selected_rows(
        directory,
        MatrixSelection {
            binding: "mtp/layers/0/attention/value",
            shape: [1_024, 5_120],
        },
    )?;
    ensure!(
        key_rows.object.id == query_rows.object.id && value_rows.object.id == query_rows.object.id,
        "native MTP Q/K/gate/V views must share one physical parent"
    );
    ensure!(
        key_rows.rows.start == 6_144
            && key_rows.rows.end == 7_168
            && value_rows.rows.start == 13_312
            && value_rows.rows.end == 14_336,
        "native MTP K/V parent row ranges differ from the checked contract"
    );
    let key = q8_view(key_rows.object, [1_024, 5_120], key_rows.rows.collect())?;
    let value = q8_view(value_rows.object, [1_024, 5_120], value_rows.rows.collect())?;
    let attention_output =
        q8_projection(directory, "mtp/layers/0/attention/output", [5_120, 6_144])?;
    let mlp_gate_rows = selected_rows(
        directory,
        MatrixSelection {
            binding: "mtp/layers/0/mlp/gate",
            shape: [17_408, 5_120],
        },
    )?;
    let mlp_up_rows = selected_rows(
        directory,
        MatrixSelection {
            binding: "mtp/layers/0/mlp/up",
            shape: [17_408, 5_120],
        },
    )?;
    ensure!(
        mlp_gate_rows.object.id == mlp_up_rows.object.id,
        "native MTP MLP gate/up views must share one physical parent"
    );
    ensure!(
        mlp_gate_rows.rows.start == 0
            && mlp_gate_rows.rows.end == 17_408
            && mlp_up_rows.rows.start == 17_408
            && mlp_up_rows.rows.end == 34_816,
        "native MTP MLP gate/up row extents differ from the checked parent ranges"
    );
    let mlp_gate = q8_view(
        mlp_gate_rows.object,
        [17_408, 5_120],
        mlp_gate_rows.rows.collect(),
    )?;
    let mlp_up = q8_view(
        mlp_up_rows.object,
        [17_408, 5_120],
        mlp_up_rows.rows.collect(),
    )?;
    ensure!(
        mlp_gate.object_id == mlp_up.object_id,
        "native MTP MLP parents differ"
    );
    let mlp_down = q8_projection(directory, "mtp/layers/0/mlp/down", [5_120, 17_408])?;
    Ok(Projections {
        fc,
        query_gate,
        key,
        value,
        attention_output,
        mlp_gate,
        mlp_up,
        mlp_down,
    })
}

fn select_norms(directory: &Directory) -> Result<NativeMtpNormViews> {
    select_norm_views(directory)
}

fn select_proposal(directory: &Directory) -> Result<(Q4MatrixView, String)> {
    let head = q4_projection(directory)?;
    validate_distinct_target_head(directory, &head)?;
    let token_map_object = select_token_map(directory)?;
    Ok((head, token_map_object))
}

pub(super) fn parse_token_map(bytes: &[u8]) -> Result<ProposalTokenMap> {
    proposal::parse_token_map(bytes)
}

pub(super) fn token_map_bytes() -> u64 {
    TOKEN_MAP_BYTES
}
