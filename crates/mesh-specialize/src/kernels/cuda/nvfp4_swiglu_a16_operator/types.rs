pub(super) struct Weight<'a> {
    pub(super) address: [u64; 2],
    pub(super) packed: &'a [u8],
    pub(super) scales: &'a [u8],
    pub(super) divisor: f32,
}

pub(super) struct Case<'a> {
    pub(super) source: &'a str,
    pub(super) input: &'a [u16],
    pub(super) width: usize,
    pub(super) gate: Weight<'a>,
    pub(super) up: Weight<'a>,
}
