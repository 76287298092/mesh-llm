//! Ordered decoder execution with a transaction spanning every stateful layer.
use super::{
    driver::{Buffer, Context, Module},
    resident_attention,
    resident_embedding::Embedding,
    resident_gdn,
    resident_head::Head,
    resident_projection::Quantization,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    engine::{sampling, session::Cursor},
    kernels::{DecoderBlockKind, DecoderConfig, DecoderMlpKind},
};
use anyhow::{Result, ensure};
use std::cell::RefCell;

enum Block<'w, 'ctx> {
    Gdn(Box<resident_gdn::Layer<'w, 'ctx>>),
    Attention(Box<resident_attention::Layer<'w, 'ctx>>),
}
pub(super) struct Model<'w, 'ctx> {
    embedding: Embedding<'w, 'ctx>,
    blocks: Vec<Block<'w, 'ctx>>,
    head: Head<'w, 'ctx>,
    vocabulary: usize,
    greedy: Option<RefCell<super::resident_greedy::Selector<'ctx>>>,
}
pub(super) struct Session<'ctx> {
    pub state: ResidentState<'ctx>,
    pub cursor: Cursor,
}
pub(super) struct SelectedOutput {
    pub token: u32,
    pub past: usize,
}
pub(super) struct Output {
    pub logits: Vec<u16>,
    pub token: u32,
    pub past: usize,
}
#[derive(Clone, Copy)]
pub(super) enum LogitsSelection {
    Last,
    All,
}
struct ExecutionOptions {
    selection: LogitsSelection,
    record: bool,
    device_selection: bool,
}
pub(super) struct DetailedOutput<'ctx> {
    pub recovery: Vec<super::resident_recovery::LayerRecord<'ctx>>,
    pub hidden: Buffer<'ctx>,
    pub logits: Vec<u16>,
    pub tokens: Vec<u32>,
    pub past: usize,
}
pub(super) type Observer<'a> = dyn FnMut(usize, &Buffer<'_>) -> Result<()> + 'a;

impl<'ctx> Session<'ctx> {
    pub(super) fn new(ctx: &'ctx Context, config: &DecoderConfig) -> Result<Self> {
        Ok(Self {
            state: ResidentState::new(ctx, &config.state_layout)?,
            cursor: Cursor::new(config.capacity)?,
        })
    }

    pub(super) fn fork<'a>(&self, ctx: &'a Context) -> Result<Session<'a>> {
        let cursor = self.cursor.fork()?;
        let state = self.state.fork(ctx)?;
        Ok(Session { state, cursor })
    }
}
impl<'w, 'ctx> Model<'w, 'ctx> {
    pub(super) fn new(weights: &'w ResidentWeights<'ctx>, config: &DecoderConfig) -> Result<Self> {
        ensure!(
            !config.layers.is_empty() && config.layers.len() <= 256,
            "invalid decoder layer count"
        );
        ensure!(
            config.gdn_shape.hidden == config.hidden
                && config.attention_shape.hidden == config.hidden,
            "decoder hidden widths disagree"
        );
        let workspace = super::model_workspace::shared(weights.context())?;
        let mut blocks = Vec::new();
        for layer in &config.layers {
            let quantization = match layer.mlp {
                DecoderMlpKind::Nvfp4 => Quantization::Nvfp4,
                DecoderMlpKind::Fp8 => Quantization::Fp8,
            };
            let mut block = match layer.block {
                DecoderBlockKind::Gdn => Block::Gdn(Box::new(resident_gdn::Layer::new(
                    weights,
                    &layer.prefix,
                    &layer.state_prefix,
                    &config.gdn_shape,
                    quantization,
                )?)),
                DecoderBlockKind::Attention => {
                    Block::Attention(Box::new(resident_attention::Layer::new(
                        weights,
                        &layer.prefix,
                        &layer.state_prefix,
                        &config.attention_shape,
                        quantization,
                    )?))
                }
            };
            if let Some(shared) = workspace.as_ref() {
                match &mut block {
                    Block::Gdn(layer) => layer.attach_workspace(shared.clone()),
                    Block::Attention(layer) => layer.attach_workspace(shared.clone()),
                }
            }
            blocks.push(block);
        }
        Ok(Self {
            embedding: Embedding::new(
                weights,
                &config.embedding_table,
                &config.first_norm,
                [config.vocabulary, config.hidden],
                1e-6,
            )?,
            blocks,
            head: Head::new(
                weights,
                &config.final_norm,
                &config.head_prefix,
                config.hidden,
                config.vocabulary,
            )?,
            vocabulary: config.vocabulary,
            greedy: if super::model_greedy::enabled()? {
                Some(RefCell::new(super::resident_greedy::Selector::new(
                    weights.context(),
                    config.vocabulary,
                )?))
            } else {
                None
            },
        })
    }

    /// Optional observers only inspect completed hidden rows; they never supply inputs.
    /// Final logits are downloaded for greedy selection before committing the cursor.
    pub(super) fn forward(
        &self,
        ctx: &Context,
        module: &Module<'_>,
        tokens: &[u32],
        session: &mut Session<'_>,
        observer: Option<&mut Observer<'_>>,
    ) -> Result<Output> {
        let detailed = self.forward_detailed(
            ctx,
            module,
            tokens,
            session,
            LogitsSelection::Last,
            observer,
        )?;
        ensure!(
            detailed.tokens.len() == 1,
            "last-row model forward did not return one token"
        );
        Ok(Output {
            logits: detailed.logits,
            token: detailed.tokens[0],
            past: detailed.past,
        })
    }

    /// Ordinary generation may return only a token. Diagnostic forwards always
    /// retain full host logits. A failed device selection never commits the cursor.
    pub(super) fn forward_selected(
        &self,
        ctx: &Context,
        module: &Module<'_>,
        tokens: &[u32],
        session: &mut Session<'_>,
    ) -> Result<SelectedOutput> {
        if self.greedy.is_none() {
            let output = self.forward(ctx, module, tokens, session, None)?;
            return Ok(SelectedOutput {
                token: output.token,
                past: output.past,
            });
        }
        let output = self.execute(
            ctx,
            module,
            tokens,
            session,
            ExecutionOptions {
                selection: LogitsSelection::Last,
                record: false,
                device_selection: true,
            },
            None,
        )?;
        ensure!(
            output.tokens.len() == 1 && output.logits.is_empty(),
            "invalid device-only selection result"
        );
        Ok(SelectedOutput {
            token: output.tokens[0],
            past: output.past,
        })
    }

    pub(super) fn forward_detailed<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        tokens: &[u32],
        session: &mut Session<'_>,
        selection: LogitsSelection,
        observer: Option<&mut Observer<'_>>,
    ) -> Result<DetailedOutput<'a>> {
        self.execute(
            ctx,
            module,
            tokens,
            session,
            ExecutionOptions {
                selection,
                record: false,
                device_selection: false,
            },
            observer,
        )
    }

    pub(super) fn forward_recorded<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        tokens: &[u32],
        session: &mut Session<'_>,
    ) -> Result<DetailedOutput<'a>> {
        ensure!(
            (1..=5).contains(&tokens.len()),
            "verification recording requires 1..=5 rows"
        );
        self.execute(
            ctx,
            module,
            tokens,
            session,
            ExecutionOptions {
                selection: LogitsSelection::All,
                record: true,
                device_selection: false,
            },
            None,
        )
    }

    fn execute<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        tokens: &[u32],
        session: &mut Session<'_>,
        options: ExecutionOptions,
        mut observer: Option<&mut Observer<'_>>,
    ) -> Result<DetailedOutput<'a>> {
        let ExecutionOptions {
            selection,
            record,
            device_selection,
        } = options;
        let mut recovery = Vec::new();
        ensure!(
            tokens.iter().all(|&id| (id as usize) < self.vocabulary),
            "decoder token is outside vocabulary"
        );
        ensure!(
            session.state.belongs_to(ctx) && module.belongs_to(ctx),
            "decoder context mismatch"
        );
        validate_selection_rows(selection, tokens.len())?;
        let transaction = session.cursor.begin(tokens.len())?;
        let entry = self.embedding.run(ctx, module, tokens)?;
        let mut hidden = entry.residual;
        drop(entry.normalized);
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = match block {
                Block::Gdn(layer) => {
                    if record {
                        let (next, layer_record) = layer.forward_recorded(
                            ctx,
                            module,
                            &hidden,
                            &mut session.state,
                            transaction.rows(),
                        )?;
                        recovery.push(layer_record);
                        next
                    } else {
                        layer.forward(
                            ctx,
                            module,
                            &hidden,
                            &mut session.state,
                            transaction.rows(),
                        )?
                    }
                }
                Block::Attention(layer) => layer.forward(
                    ctx,
                    module,
                    &hidden,
                    &mut session.state,
                    &resident_attention::Step {
                        rows: transaction.rows(),
                        past: transaction.past(),
                        capacity: transaction.capacity(),
                    },
                )?,
            };
            if let Some(inspect) = observer.as_deref_mut() {
                inspect(index, &hidden)?;
            }
        }
        let (output, logit_rows) = match selection {
            LogitsSelection::Last => (self.head.run(ctx, module, &hidden, transaction.rows())?, 1),
            LogitsSelection::All => (
                self.head
                    .run_all(ctx, module, &hidden, transaction.rows())?,
                transaction.rows(),
            ),
        };
        let expected_bytes = checked_logit_bytes(logit_rows, self.vocabulary)?;
        ensure!(
            output.values.len() == expected_bytes,
            "decoder logit extent mismatch"
        );
        let (logits, selected_tokens) = if device_selection {
            ensure!(
                matches!(selection, LogitsSelection::Last),
                "device-only selection requires one logit row"
            );
            let mut selector = self
                .greedy
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing GPU selector"))?
                .try_borrow_mut()
                .map_err(|_| anyhow::anyhow!("GPU selector is already borrowed"))?;
            let chosen = selector.select(module, &output.values)?;
            (Vec::new(), vec![chosen.token])
        } else {
            download_selection(&output.values, self.vocabulary, selection)?
        };
        let past = transaction.commit();
        Ok(DetailedOutput {
            recovery,
            hidden,
            logits,
            tokens: selected_tokens,
            past,
        })
    }
}

fn download_selection(
    values: &Buffer<'_>,
    vocabulary: usize,
    selection: LogitsSelection,
) -> Result<(Vec<u16>, Vec<u32>)> {
    let mut raw = vec![0; values.len()];
    values.download(&mut raw)?;
    let logits = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect::<Vec<_>>();
    let selected = match selection {
        LogitsSelection::Last => vec![sampling::greedy(&logits)?],
        LogitsSelection::All => logits
            .chunks_exact(vocabulary)
            .map(sampling::greedy)
            .collect::<Result<Vec<_>>>()?,
    };
    Ok((logits, selected))
}

fn checked_logit_bytes(rows: usize, vocabulary: usize) -> Result<usize> {
    let elements = rows
        .checked_mul(vocabulary)
        .ok_or_else(|| anyhow::anyhow!("decoder logit element count overflows usize"))?;
    elements
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("decoder logit byte count overflows usize"))
}

fn validate_selection_rows(selection: LogitsSelection, rows: usize) -> Result<()> {
    if matches!(selection, LogitsSelection::All) {
        ensure!(rows <= 17, "all-row logits are limited to 17 input rows");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LogitsSelection, checked_logit_bytes, validate_selection_rows};

    #[test]
    fn bounds_all_row_logits_without_limiting_last_row_selection() {
        assert!(validate_selection_rows(LogitsSelection::All, 17).is_ok());
        assert!(validate_selection_rows(LogitsSelection::All, 18).is_err());
        assert!(validate_selection_rows(LogitsSelection::Last, 2048).is_ok());
    }

    #[test]
    fn logit_output_byte_count_is_checked() {
        assert_eq!(checked_logit_bytes(17, 4).unwrap(), 136);
        assert!(checked_logit_bytes(usize::MAX, 2).is_err());
        assert!(checked_logit_bytes(usize::MAX / 2 + 1, 1).is_err());
    }
}
