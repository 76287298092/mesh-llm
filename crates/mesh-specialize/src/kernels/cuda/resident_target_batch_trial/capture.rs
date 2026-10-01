use super::{super::resident_model::Session, Request};
use crate::kernels::cuda::{driver::Buffer, resident_model::DetailedOutput};
use anyhow::{Context as _, Result};

pub(super) struct Capture {
    pub layers: Vec<Vec<u16>>,
    pub hidden: Vec<u16>,
    pub logits: Vec<u16>,
    pub recovery_records: usize,
}

pub(super) fn recorded(
    request: &Request<'_, '_, '_>,
    session: &mut Session<'_>,
    tokens: &[u32],
) -> Result<Capture> {
    let mut layers = empty_layers(request.config.layers.len());
    let mut observer = |index: usize, buffer: &Buffer<'_>| -> Result<()> {
        layers
            .get_mut(index)
            .context("recorded observer layer is out of range")?
            .extend(words(buffer)?);
        Ok(())
    };
    let output = request.model.forward_recorded_observed(
        request.context,
        request.module,
        tokens,
        session,
        Some(&mut observer),
    )?;
    from_output(output, layers)
}

pub(super) fn decode_sequence(
    request: &Request<'_, '_, '_>,
    session: &mut Session<'_>,
    tokens: &[u32],
) -> Result<Capture> {
    let mut layers = empty_layers(request.config.layers.len());
    let mut hidden = Vec::new();
    let mut logits = Vec::new();
    let mut recovery_records = 0_usize;
    for (index, &token) in tokens.iter().enumerate() {
        let mut observer = |layer: usize, buffer: &Buffer<'_>| -> Result<()> {
            layers
                .get_mut(layer)
                .context("decode observer layer is out of range")?
                .extend(words(buffer)?);
            Ok(())
        };
        let output = request.model.forward_detailed_decode(
            request.context,
            request.module,
            token,
            session,
            Some(&mut observer),
        )?;
        if output.logits.len() != request.config.vocabulary {
            anyhow::bail!("decode token {index} returned an incomplete vocabulary row");
        }
        logits.extend(output.logits);
        hidden.extend(words(&output.hidden)?);
        recovery_records = recovery_records
            .checked_add(output.recovery.len())
            .context("decode recovery record count overflows usize")?;
    }
    Ok(Capture {
        layers,
        hidden,
        logits,
        recovery_records,
    })
}

fn from_output(output: DetailedOutput<'_>, layers: Vec<Vec<u16>>) -> Result<Capture> {
    Ok(Capture {
        layers,
        hidden: words(&output.hidden)?,
        logits: output.logits,
        recovery_records: output.recovery.len(),
    })
}

fn empty_layers(count: usize) -> Vec<Vec<u16>> {
    (0..count).map(|_| Vec::new()).collect()
}

fn words(buffer: &Buffer<'_>) -> Result<Vec<u16>> {
    anyhow::ensure!(
        buffer.len().is_multiple_of(2),
        "BF16 buffer has odd byte length"
    );
    let mut raw = vec![0; buffer.len()];
    buffer.download(&mut raw)?;
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}
