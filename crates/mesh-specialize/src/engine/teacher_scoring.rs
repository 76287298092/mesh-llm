//! Fixed-window teacher-forced scoring plan and per-position record format.
//!
//! Window semantics (fixed-window, truncated-context causal perplexity):
//! every token after `x[0]` is scored exactly once. The first window is
//! `[0, min(context, N))` and scores targets `[1, min(context, N))`. Each later
//! window advances the target end by `stride`, keeps up to `context - stride`
//! preceding tokens as history, and scores only the new targets. Every window
//! starts from empty state; streams never share history.

use anyhow::{Result, ensure};

/// Retained top candidates per scored position.
pub const TOP_K: usize = 64;
/// Bytes per record: target id, target logprob, logsumexp, 64 ids, 64 logprobs.
pub const RECORD_BYTES: usize = 4 + 4 + 4 + TOP_K * 4 + TOP_K * 4;

/// One evaluation window over a token stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// Input tokens are `[input_begin, input_end)` of the stream.
    pub input_begin: usize,
    pub input_end: usize,
    /// Scored targets are stream positions `[target_begin, target_end)`.
    pub target_begin: usize,
    pub target_end: usize,
}

impl Window {
    /// Number of scored targets.
    pub fn scored(&self) -> usize {
        self.target_end - self.target_begin
    }
    /// Local input row whose logits predict the first scored target.
    pub fn first_row(&self) -> usize {
        self.target_begin - 1 - self.input_begin
    }
}

/// Plan windows for a stream of `tokens` tokens.
pub fn plan_windows(tokens: usize, context: usize, stride: usize) -> Result<Vec<Window>> {
    ensure!(tokens >= 2, "a scored stream needs at least two tokens");
    ensure!(
        context >= 2 && stride >= 1 && stride < context,
        "scoring requires context >= 2 and 1 <= stride < context"
    );
    let mut end = tokens.min(context);
    let mut windows = vec![Window {
        input_begin: 0,
        input_end: end,
        target_begin: 1,
        target_end: end,
    }];
    while end < tokens {
        let next_end = tokens.min(end + stride);
        let begin = next_end.saturating_sub(context);
        ensure!(
            begin < end && end < next_end,
            "window plan produced an empty target suffix"
        );
        windows.push(Window {
            input_begin: begin,
            input_end: next_end,
            target_begin: end,
            target_end: next_end,
        });
        end = next_end;
    }
    Ok(windows)
}

/// One scored position.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub target: u32,
    pub target_logprob: f32,
    pub logsumexp: f32,
    pub top_ids: [u32; TOP_K],
    pub top_logprobs: [f32; TOP_K],
}

impl Record {
    /// Append the little-endian fixed-size encoding.
    pub fn encode_into(&self, output: &mut Vec<u8>) {
        output.reserve(RECORD_BYTES);
        output.extend_from_slice(&self.target.to_le_bytes());
        output.extend_from_slice(&self.target_logprob.to_le_bytes());
        output.extend_from_slice(&self.logsumexp.to_le_bytes());
        for id in self.top_ids {
            output.extend_from_slice(&id.to_le_bytes());
        }
        for value in self.top_logprobs {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }

    /// Decode one record from exactly `RECORD_BYTES` bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() == RECORD_BYTES, "score record extent mismatch");
        let word = |index: usize| -> [u8; 4] {
            bytes[index * 4..index * 4 + 4]
                .try_into()
                .expect("four-byte word")
        };
        Ok(Self {
            target: u32::from_le_bytes(word(0)),
            target_logprob: f32::from_le_bytes(word(1)),
            logsumexp: f32::from_le_bytes(word(2)),
            top_ids: std::array::from_fn(|i| u32::from_le_bytes(word(3 + i))),
            top_logprobs: std::array::from_fn(|i| f32::from_le_bytes(word(3 + TOP_K + i))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{RECORD_BYTES, Record, TOP_K, Window, plan_windows};

    fn window(input: (usize, usize), target: (usize, usize)) -> Window {
        Window {
            input_begin: input.0,
            input_end: input.1,
            target_begin: target.0,
            target_end: target.1,
        }
    }

    #[test]
    fn slice_protocol_windows_match_reference_semantics() {
        let windows = plan_windows(12_288, 512, 256).unwrap();
        assert_eq!(windows.len(), 47);
        assert_eq!(windows[0], window((0, 512), (1, 512)));
        assert_eq!(windows[1], window((256, 768), (512, 768)));
        assert_eq!(windows[2], window((512, 1024), (768, 1024)));
        assert_eq!(windows[46], window((11_776, 12_288), (12_032, 12_288)));
        assert_eq!(windows.iter().map(Window::scored).sum::<usize>(), 12_287);
        assert_eq!(windows[0].first_row(), 0);
        assert_eq!(windows[1].first_row(), 255);
        for pair in windows.windows(2) {
            assert_eq!(pair[0].target_end, pair[1].target_begin);
            assert!(pair[1].input_end - pair[1].input_begin <= 512);
        }
    }

    #[test]
    fn short_tail_and_short_stream_cases() {
        // 700 tokens: second window ends at 700 and keeps 512 inputs.
        let windows = plan_windows(700, 512, 256).unwrap();
        assert_eq!(
            windows,
            [window((0, 512), (1, 512)), window((188, 700), (512, 700))]
        );
        assert_eq!(windows[1].first_row(), 323);
        // Shorter than one context: a single window.
        assert_eq!(plan_windows(5, 512, 256).unwrap(), [window((0, 5), (1, 5))]);
        // Exactly one context.
        assert_eq!(
            plan_windows(512, 512, 256).unwrap(),
            [window((0, 512), (1, 512))]
        );
        // One token past a context: tail of one target with full history.
        assert_eq!(
            plan_windows(513, 512, 256).unwrap()[1],
            window((1, 513), (512, 513))
        );
        // Stride below half a context keeps more history than new targets.
        assert_eq!(
            plan_windows(10, 4, 1).unwrap(),
            [
                window((0, 4), (1, 4)),
                window((1, 5), (4, 5)),
                window((2, 6), (5, 6)),
                window((3, 7), (6, 7)),
                window((4, 8), (7, 8)),
                window((5, 9), (8, 9)),
                window((6, 10), (9, 10)),
            ]
        );
    }

    #[test]
    fn rejects_invalid_protocols() {
        assert!(plan_windows(1, 512, 256).is_err());
        assert!(plan_windows(10, 1, 1).is_err());
        assert!(plan_windows(10, 4, 0).is_err());
        assert!(plan_windows(10, 4, 4).is_err());
    }

    #[test]
    fn record_round_trips_little_endian() {
        let record = Record {
            target: 7,
            target_logprob: -1.5,
            logsumexp: 12.25,
            top_ids: std::array::from_fn(|i| i as u32 * 3),
            top_logprobs: std::array::from_fn(|i| -(i as f32)),
        };
        let mut bytes = Vec::new();
        record.encode_into(&mut bytes);
        assert_eq!(bytes.len(), RECORD_BYTES);
        assert_eq!(RECORD_BYTES, 524);
        assert_eq!(&bytes[..4], &[7, 0, 0, 0]);
        assert_eq!(&bytes[4..8], &(-1.5_f32).to_le_bytes());
        assert_eq!(&bytes[12..16], &0_u32.to_le_bytes());
        assert_eq!(
            &bytes[12 + 4 * TOP_K..16 + 4 * TOP_K],
            &(-0.0_f32).to_le_bytes()
        );
        assert_eq!(Record::decode(&bytes).unwrap(), record);
        assert!(Record::decode(&bytes[1..]).is_err());
    }
}
