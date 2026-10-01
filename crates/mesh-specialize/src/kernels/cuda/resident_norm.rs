mod extents;
mod norm;
#[cfg(test)]
mod tests;

pub(super) use norm::{Norm, Normalized, residual_add};
