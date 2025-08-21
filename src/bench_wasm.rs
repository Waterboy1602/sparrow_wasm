extern crate core;
use sparrow::util::terminator::Terminator;

use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use sparrow::config::*;
use sparrow::optimizer::lbf::LBFBuilder;
use sparrow::optimizer::separator::Separator;
use sparrow::util::io;
use std::env::args;
use std::fs;
use std::io::{self as std_io, BufRead, Write};
use std::path::Path;
use std::time::{Duration, Instant};
use sysinfo::System;

use anyhow::Result;
use jagua_rs::io::import::Importer;
use sparrow::consts::{
    DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DEFAULT_FAIL_DECAY_RATIO_CMPR,
    DEFAULT_MAX_CONSEQ_FAILS_EXPL, LBF_SAMPLE_CONFIG, LOG_LEVEL_FILTER_RELEASE,
};
use sparrow::optimizer::compress::compression_phase;
use sparrow::optimizer::explore::exploration_phase;
use sparrow::util::listener::DummySolListener;
use sparrow::util::terminator::BasicTerminator;

pub const OUTPUT_DIR: &str = "output";

fn main() -> Result<()> {
    let mut config = DEFAULT_SPARROW_CONFIG;

    //the input file is the first argument
    let input_file_path = args()
        .nth(1)
        .expect("first argument must be the input file");

    // Using early termination & hardcoded time limit (high enough)
    let time_limit = Duration::from_secs(60);
    config.expl_cfg.time_limit = time_limit.mul_f32(DEFAULT_EXPLORE_TIME_RATIO);
    config.cmpr_cfg.time_limit = time_limit.mul_f32(DEFAULT_COMPRESS_TIME_RATIO);

    config.expl_cfg.max_conseq_failed_attempts = Some(DEFAULT_MAX_CONSEQ_FAILS_EXPL);
    config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(DEFAULT_FAIL_DECAY_RATIO_CMPR);
    println!("[MAIN] early termination enabled!");

    fs::create_dir_all(OUTPUT_DIR).expect("could not create output directory");

    let log_file_path = format!("{}/bench_log.txt", OUTPUT_DIR);
    io::init_logger(LOG_LEVEL_FILTER_RELEASE, Path::new(&log_file_path))?;

    let start_time = Instant::now();
    let git_branch = get_git_branch();
    let git_commit_hash = get_git_commit_hash();
    println!("[BENCH] git branch: {}", git_branch);
    println!("[BENCH] git commit hash: {}", git_commit_hash);
    println!("[BENCH] system time: {}", jiff::Timestamp::now());

    config.rng_seed = Some(8780830896941405304);

    let mut rng = match config.rng_seed {
        Some(seed) => {
            println!("[BENCH] using provided seed: {}", seed);
            SmallRng::seed_from_u64(seed as u64)
        }
        None => {
            let seed = rand::random();
            println!("[BENCH] no seed provided, using: {}", seed);
            SmallRng::seed_from_u64(seed)
        }
    };

    let ext_intance = io::read_spp_instance_json(Path::new(&input_file_path))?;

    println!(
        "[BENCH] starting bench for {}, {} cores, {:?} timelimit)",
        ext_intance.name,
        num_cpus::get_physical(),
        time_limit
    );

    let importer = Importer::new(
        config.cde_config,
        config.poly_simpl_tolerance,
        config.min_item_separation,
        config.narrow_concavity_cutoff_ratio,
    );
    let instance = jagua_rs::probs::spp::io::import(&importer, &ext_intance)?;

    let rng = SmallRng::seed_from_u64(rng.random());
    let mut terminator = BasicTerminator::new();

    let builder = LBFBuilder::new(instance.clone(), rng.clone(), LBF_SAMPLE_CONFIG).construct();
    let mut expl_separator = Separator::new(
        builder.instance,
        builder.prob,
        rng.clone(),
        config.expl_cfg.separator_config,
    );

    terminator.new_timeout(time_limit.mul_f32(DEFAULT_EXPLORE_TIME_RATIO));
    let solutions = exploration_phase(
        &instance,
        &mut expl_separator,
        &mut DummySolListener,
        &terminator,
        &config.expl_cfg,
    );
    let final_explore_sol = solutions
        .last()
        .expect("no solutions found during exploration");

    let start_comp = Instant::now();

    terminator.new_timeout(time_limit.mul_f32(DEFAULT_COMPRESS_TIME_RATIO));
    let mut cmpr_separator = Separator::new(
        expl_separator.instance,
        expl_separator.prob,
        rng.clone(),
        config.cmpr_cfg.separator_config,
    );
    let cmpr_sol = compression_phase(
        &instance,
        &mut cmpr_separator,
        final_explore_sol,
        &mut DummySolListener,
        &terminator,
        &config.cmpr_cfg,
    );

    println!(
        "[BENCH] finished, expl: {:.3}% ({}s), cmpr: {:.3}% (+{:.3}%) ({}s)",
        final_explore_sol.density(&instance) * 100.0,
        time_limit.mul_f32(DEFAULT_EXPLORE_TIME_RATIO).as_secs(),
        cmpr_sol.density(&instance) * 100.0,
        cmpr_sol.density(&instance) * 100.0 - final_explore_sol.density(&instance) * 100.0,
        start_comp.elapsed().as_secs()
    );

    let end_time = Instant::now();

    let elapsed_time = end_time - start_time;
    println!("==== BENCH FINISHED ====");

    println!("Elapsed time {} ms", elapsed_time.as_millis());

    write_to_csv(
        &get_cpu_model(),
        Some(&git_branch),
        Some(&git_commit_hash),
        config.rng_seed,
        true,
        &elapsed_time.as_millis().to_string(),
    )?;

    Ok(())
}

pub fn write_to_csv(
    cpu: &str,
    branch: Option<&str>,
    commit_hash: Option<&str>,
    seed: Option<usize>,
    early_termination: bool,
    running_time: &str,
) -> std_io::Result<()> {
    let filename = "./../results/benchmark_results_native.csv";
    let header = "Timestamp,CPU,Branch,CommitHash,Seed,EarlyTermination,RunningTime\n";

    let parent_dir = Path::new(filename).parent().unwrap();
    fs::create_dir_all(parent_dir)?;

    if !fs::metadata(filename).is_ok() {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(filename)?;
        file.write_all(header.as_bytes())?;
    }

    let mut file = fs::OpenOptions::new().append(true).open(filename)?;

    let timestamp = jiff::Timestamp::now();
    let line = format!(
        "{},{},{},{},{:?},{},{}\n",
        timestamp,
        cpu,
        branch.unwrap_or(""),
        commit_hash.unwrap_or(""),
        seed.unwrap_or(0),
        early_termination,
        running_time,
    );

    file.write_all(line.as_bytes())?;

    Ok(())
}

pub fn get_git_commit_hash() -> String {
    let output = std::process::Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .expect("Failed to execute git command");

    match output.status.success() {
        true => String::from_utf8_lossy(&output.stdout).trim().to_string(),
        false => "unknown".to_string(),
    }
}

pub fn get_git_branch() -> String {
    let parent_dir = Path::new("..").canonicalize().unwrap_or_else(|_| {
        panic!("Failed to get parent directory path");
    });

    let output = std::process::Command::new("git")
        .current_dir(&parent_dir)
        .args(&["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .expect("Failed to execute git command");

    match output.status.success() {
        true => String::from_utf8_lossy(&output.stdout).trim().to_string(),
        false => "unknown".to_string(),
    }
}

pub fn get_cpu_model() -> String {
    let sys = System::new_all();

    if let Some(cpu) = sys.cpus().get(0) {
        return cpu.brand().to_string();
    }

    "Unknown".to_string()
}

pub fn get_max_eval() -> String {
    let path = Path::new("./output/bench_log.txt");

    let file = match fs::File::open(&path) {
        Ok(file) => file,
        Err(e) => return format!("Error: Could not open file {}: {}", path.display(), e),
    };

    let reader = std_io::BufReader::new(file);
    let mut max_evals: f64 = 0.0;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Warning: Skipping a line due to read error: {}", e);
                continue;
            }
        };

        if let Some((_, value_str)) = line.split_once("evals/s:") {
            let trimmed_value = value_str.trim();

            if let Some(first_part) = trimmed_value.split_whitespace().next() {
                match first_part.trim().parse::<f64>() {
                    Ok(evals) => {
                        if evals > max_evals {
                            max_evals = evals;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Could not parse number from line '{}': {}",
                            value_str, e
                        );
                    }
                }
            }
        }
    }

    if max_evals == 0.0 {
        "No 'eval/s:' values found in the log file.".to_string()
    } else {
        max_evals.to_string()
    }
}
