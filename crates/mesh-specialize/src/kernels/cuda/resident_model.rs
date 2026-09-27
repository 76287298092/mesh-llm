//! Ordered decoder execution with a transaction spanning every stateful layer.
use super::{
    driver::{Buffer, Context, Module},
    resident_attention,
    resident_embedding::Embedding,
    resident_gdn,
    resident_head::Head,
    resident_mlp::Quantization,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    engine::{sampling, session::Cursor},
    kernels::{DecoderBlockKind, DecoderConfig, DecoderMlpKind},
};
use anyhow::{Result, ensure};

enum Block<'w, 'ctx> {
    Gdn(Box<resident_gdn::Layer<'w, 'ctx>>),
    Attention(Box<resident_attention::Layer<'w, 'ctx>>),
}
pub(super) struct Model<'w, 'ctx> {
    embedding: Embedding<'w, 'ctx>,
    blocks: Vec<Block<'w, 'ctx>>,
    head: Head<'w, 'ctx>,
    vocabulary: usize,
}
pub(super) struct Session<'ctx> {
    pub state: ResidentState<'ctx>,
    pub cursor: Cursor,
}
pub(super) struct Output {
    pub logits: Vec<u16>,
    pub token: u32,
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
        let mut blocks = Vec::new();
        for layer in &config.layers {
            let quantization = match layer.mlp {
                DecoderMlpKind::Nvfp4 => Quantization::Nvfp4,
                DecoderMlpKind::Fp8 => Quantization::Fp8,
            };
            blocks.push(match layer.block {
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
            });
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
        mut observer: Option<&mut Observer<'_>>,
    ) -> Result<Output> {
        ensure!(
            tokens.iter().all(|&id| (id as usize) < self.vocabulary),
            "decoder token is outside vocabulary"
        );
        ensure!(
            session.state.belongs_to(ctx) && module.belongs_to(ctx),
            "decoder context mismatch"
        );
        let transaction = session.cursor.begin(tokens.len())?;
        let entry = self.embedding.run(ctx, module, tokens)?;
        let mut hidden = entry.residual;
        drop(entry.normalized);
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = match block {
                Block::Gdn(layer) => {
                    layer.forward(ctx, module, &hidden, &mut session.state, transaction.rows())?
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
        let output = self.head.run(ctx, module, &hidden, transaction.rows())?;
        let mut raw = vec![0; output.values.len()];
        output.values.download(&mut raw)?;
        let logits = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect::<Vec<_>>();
        ensure!(
            logits.len() == self.vocabulary,
            "decoder logit extent mismatch"
        );
        let token = sampling::greedy(&logits)?;
        let past = transaction.commit();
        Ok(Output {
            logits,
            token,
            past,
        })
    }
}
