#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(const_item_mutation)]

use numfmt::{Formatter, Precision, Scales};
use std::sync::LazyLock;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

pub mod config;
pub mod eval;
pub mod optimizer;
pub mod quantify;
pub mod sample;
pub mod util;

pub static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

static FMT: fn() -> Formatter = || -> Formatter {
    Formatter::new()
        .scales(Scales::short())
        .precision(Precision::Significance(3))
};


#[cfg(feature = "live_svg")]
pub const EXPORT_LIVE_SVG: bool = true;

#[cfg(not(feature = "live_svg"))]
pub const EXPORT_LIVE_SVG: bool = false;

#[cfg(feature = "only_final_svg")]
pub const EXPORT_ONLY_FINAL_SVG: bool = true;

#[cfg(not(feature = "only_final_svg"))]
pub const EXPORT_ONLY_FINAL_SVG: bool = false;

#[cfg(all(feature = "live_svg", feature = "only_final_svg"))]
compile_error!("The features `live_svg` and `only_final_svg` are mutually exclusive.");