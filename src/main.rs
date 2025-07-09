extern crate core;

use clap::Parser as Clap;
use log::{Level, info, warn};
use rand::SeedableRng;
use rand::prelude::SmallRng;
use sparrow::config::*;
use sparrow::optimizer::optimize;
use sparrow::util::io;
use sparrow::util::io::{MainCli, SPOutput};
use std::fs;
use std::path::Path;

use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::s_layout_to_svg;
use sparrow::EPOCH;

use anyhow::{Result, bail};

use sparrow::util::ctrlc_terminator::CtrlCTerminator;
use sparrow::util::svg_exporter::SvgExporter;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

fn main() -> Result<()> {
    fs::create_dir_all(OUTPUT_DIR)?;
    match cfg!(debug_assertions) {
        true => io::init_logger(LOG_LEVEL_FILTER_DEBUG)?,
        false => io::init_logger(LOG_LEVEL_FILTER_RELEASE)?,
    }

    let args = MainCli::parse();
    let input_file_path = &args.input;
    let (explore_dur, compress_dur) = match (args.global_time, args.exploration, args.compression) {
        (Some(gt), None, None) => (
            Duration::from_secs(gt).mul_f32(EXPLORE_TIME_RATIO),
            Duration::from_secs(gt).mul_f32(COMPRESS_TIME_RATIO),
        ),
        (None, Some(et), Some(ct)) => (Duration::from_secs(et), Duration::from_secs(ct)),
        (None, None, None) => {
            warn!("[MAIN] No time limit specified");
            (
                Duration::from_secs(600).mul_f32(EXPLORE_TIME_RATIO),
                Duration::from_secs(600).mul_f32(COMPRESS_TIME_RATIO),
            )
        }
        _ => bail!("invalid cli pattern (clap should have caught this)"),
    };

    info!(
        "[MAIN] Configured to explore for {}s and compress for {}s",
        explore_dur.as_secs(),
        compress_dur.as_secs()
    );

    let rng = match RNG_SEED {
        Some(seed) => {
            info!("[MAIN] using seed: {}", seed);
            SmallRng::seed_from_u64(seed as u64)
        }
        None => {
            let seed = rand::random();
            warn!("[MAIN] no seed provided, using: {}", seed);
            SmallRng::seed_from_u64(seed)
        }
    };

    info!("[MAIN] system time: {}", jiff::Timestamp::now());

    let ext_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;

    let importer = Importer::new(CDE_CONFIG, SIMPL_TOLERANCE, MIN_ITEM_SEPARATION);
    let instance = jagua_rs::probs::spp::io::import(&importer, &ext_instance)?;

    info!(
        "[MAIN] loaded instance {} with #{} items",
        ext_instance.name,
        instance.total_item_qty()
    );

    let mut svg_exporter = {
        let final_svg_path = Some(format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name));

        let intermediate_svg_dir = match cfg!(feature = "only_final_svg") {
            true => None,
            false => Some(format!("{OUTPUT_DIR}/sols_{}", ext_instance.name)),
        };

        let live_svg_path = match cfg!(feature = "live_svg") {
            true => Some(format!("{LIVE_DIR}/.live_solution.svg")),
            false => None,
        };

        SvgExporter::new(final_svg_path, intermediate_svg_dir, live_svg_path)
    };

    let mut ctrlc_terminator = CtrlCTerminator::new();

    let solution = optimize(
        instance.clone(),
        rng,
        &mut svg_exporter,
        &mut ctrlc_terminator,
        explore_dur,
        compress_dur,
    );

    {
        let svg = s_layout_to_svg(&solution.layout_snapshot, &instance, DRAW_OPTIONS, "final");
        io::write_svg(
            &svg,
            Path::new(format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name).as_str()),
            Level::Info,
        )?;
        if cfg!(feature = "live_svg") {
            io::write_svg(
                &svg,
                Path::new(format!("{LIVE_DIR}/.live_solution.svg").as_str()),
                Level::Trace,
            )?;
        }
        let json_path = format!("{OUTPUT_DIR}/final_{}.json", ext_instance.name);
        let json_output = SPOutput {
            instance: ext_instance,
            solution: jagua_rs::probs::spp::io::export(&instance, &solution, *EPOCH),
        };
        io::write_json(&json_output, Path::new(json_path.as_str()), Level::Info)?;
    }

    Ok(())
}
