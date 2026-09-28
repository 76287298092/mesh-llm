use crate::artifact::ninfer::{Binding, Directory, Object, Part};
use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy)]
pub(super) struct MatrixSelection<'a> {
    pub binding: &'a str,
    pub shape: [usize; 2],
}

pub(super) struct SelectedRows<'a> {
    pub object: &'a Object,
    pub rows: std::ops::Range<usize>,
}

pub(super) fn selected_rows<'a>(
    directory: &'a Directory,
    selection: MatrixSelection<'_>,
) -> Result<SelectedRows<'a>> {
    let binding = directory
        .bindings
        .get(selection.binding)
        .with_context(|| format!("missing native MTP binding {}", selection.binding))?;
    let (object_id, range) = match binding {
        Binding::Object { object } => (object.as_str(), None),
        Binding::Parts { parts } => {
            ensure!(parts.len() == 1, "native MTP binding needs one parent part");
            let part = parts
                .first()
                .context("native MTP binding part is missing")?;
            (part.object.as_str(), Some(part))
        }
    };
    let object = directory
        .objects
        .iter()
        .find(|object| object.id == object_id)
        .with_context(|| format!("native MTP object {object_id} is missing"))?;
    ensure!(
        object.kind == "tensor",
        "native MTP binding is not a tensor"
    );
    object.require_supported_encoding()?;
    let [parent_rows, parent_columns]: [u64; 2] = object
        .shape
        .as_slice()
        .try_into()
        .context("native MTP parent must be a matrix")?;
    let columns = usize::try_from(parent_columns)?;
    ensure!(columns == selection.shape[1], "native MTP K shape mismatch");
    let logical_elements = object.logical_elements()?;
    let (first, end) = match range {
        Some(Part { range, .. }) => (range[0], range[1]),
        None => (0, logical_elements),
    };
    ensure!(
        first < end && end <= logical_elements,
        "native MTP range is invalid"
    );
    let first = usize::try_from(first)?;
    let end = usize::try_from(end)?;
    ensure!(
        first.is_multiple_of(columns) && end.is_multiple_of(columns),
        "native MTP view must cover complete parent rows"
    );
    let rows = first / columns..end / columns;
    ensure!(
        rows.len() == selection.shape[0],
        "native MTP N shape mismatch"
    );
    ensure!(
        rows.end <= usize::try_from(parent_rows)?,
        "native MTP row selection exceeds its parent"
    );
    ensure!(
        object.layout.as_deref() == Some("row_split_k128_v1"),
        "unsupported native MTP quantized layout"
    );
    Ok(SelectedRows { object, rows })
}

pub(super) fn interleave_query_gate(
    query: &SelectedRows<'_>,
    gate: &SelectedRows<'_>,
) -> Result<Vec<usize>> {
    ensure!(
        query.rows.len() == 6_144 && gate.rows.len() == 6_144,
        "native MTP Q/gate interleave shape mismatch"
    );
    ensure!(
        query.rows.start == 0 && gate.rows.start == 7_168,
        "native MTP Q/gate parent row offsets differ from the checked contract"
    );
    let mut rows = Vec::new();
    rows.try_reserve_exact(12_288)
        .context("cannot reserve native MTP Q/gate row map")?;
    for head in 0..24 {
        let query_start = query.rows.start + head * 256;
        let gate_start = gate.rows.start + head * 256;
        rows.extend(query_start..query_start + 256);
        rows.extend(gate_start..gate_start + 256);
    }
    ensure!(rows.len() == 12_288, "native MTP Q/gate row map mismatch");
    Ok(rows)
}
