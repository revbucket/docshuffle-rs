use rand::thread_rng;
use rand::prelude::SliceRandom;
use std::fs;
use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicUsize;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::fs::OpenOptions;
use std::sync::Mutex;
use std::io::BufWriter;
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;
use std::io::BufRead;
use anyhow::Error;
use clap::Parser;
use std::path::PathBuf;
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;

pub mod s3;
pub mod io;




/*=================================================================
=                                  ARGS                           =
=================================================================*/



#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[arg(long, required=true, num_args=1..)]
    input: Vec<PathBuf>,

    #[arg(long, required=true)]
    output: PathBuf,

    #[arg(long, required=true)]
    local_cell_storage: PathBuf,

    #[arg(long, default_value_t=1024)]
    num_local_cells: usize,

    #[arg(long, default_value_t=50000)]
    docs_per_jsonl: usize,

    #[arg(long)]
    remove_locals: bool

}


/*=================================================================
=                             UTILITIES.                          =
=================================================================*/

fn build_pbar(num_items: usize, units: &str) -> ProgressBar {
    let mut template = String::from(units);
    template.push_str(" {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]");
    let pbar = ProgressBar::new(num_items as u64)
        .with_style(
            ProgressStyle::with_template(&template).unwrap()
        );

    pbar.inc(0);
    pbar
}


/*=================================================================
=                               COARSE-SHUFFLE                    =
=================================================================*/

fn build_local_mappers(mapper_loc: &PathBuf, num_local_cells: usize) -> (Vec<Arc<Mutex<BufWriter<File>>>>, Vec<PathBuf>) {
    let mut writers: Vec<Arc<Mutex<BufWriter<File>>>> = Vec::new();
    let mut filenames: Vec<PathBuf> = Vec::new();
    for i in 0..num_local_cells {
        let filename = mapper_loc.clone().join(format!("local_mapper_{:?}.bin", i));

        let writer = Arc::new(
            Mutex::new(
            BufWriter::new(
            OpenOptions::new()
            .append(true)
            .create(true)
            .mode(0o644)
            .open(filename.clone())
            .unwrap()
        )));
        writers.push(writer);
        filenames.push(filename.clone());
    }
    (writers, filenames)
}


fn coarse_shuffle(input_paths: &Vec<PathBuf>, local_cell_storage: &PathBuf, num_local_cells: usize, remove_locals: bool) -> Result<Vec<PathBuf>, Error> {
    let (writers, filenames) = build_local_mappers(local_cell_storage, num_local_cells);
    let pbar = build_pbar(input_paths.len(), "Paths");
    input_paths.par_iter()
        .for_each(|p| {
            coarse_shuffle_single(p, &writers).unwrap();

            if remove_locals {
                fs::remove_file(p.clone()).unwrap();
            }
            pbar.inc(1);
    });

    writers.par_iter()
        .for_each(|writer| writer.lock().unwrap().flush().unwrap());
    Ok(filenames)
}


fn coarse_shuffle_single(path: &PathBuf, writers: &Vec<Arc<Mutex<BufWriter<File>>>>) -> Result<(), Error> {
    let num_local_cells = writers.len();
    let contents = read_pathbuf_to_mem(path).unwrap();
    let mut rng = rand::thread_rng();
    for line in contents.lines() {
        let line = line.unwrap();
        let mut line = line.into_bytes();
        line.push(b'\n');
        let idx = rng.gen::<usize>() as usize % num_local_cells;
        writers[idx].lock().unwrap().write_all(&line).unwrap();
    }
    Ok(())
}


/*=================================================================
=                            FINE-SHUFFLE.                        =
=================================================================*/

fn finalize_chunks(filenames: Vec<PathBuf>, output: &PathBuf, docs_per_jsonl: usize, remove_locals: bool) -> Result<usize, Error> {
    let pbar = build_pbar(filenames.len(), "Local Cells");
    let counter = AtomicUsize::new(0);
    let output_file_count = AtomicUsize::new(0);
    filenames.par_iter()
        .for_each(|filename| {
            let contents = read_pathbuf_to_mem(&filename).unwrap();
            let mut lines: Vec<String> = contents.lines().map(|line| line.unwrap()).collect();
            lines.shuffle(&mut thread_rng());
            for chunk in lines.chunks(docs_per_jsonl) {
                write_chunk(chunk, &output, &counter, &output_file_count).unwrap();
            }
            if remove_locals {
                fs::remove_file(filename.clone()).unwrap();
            }
            pbar.inc(1);
        });

    Ok(output_file_count.into_inner())
}


fn write_chunk(chunk: &[String], output: &PathBuf, counter: &AtomicUsize, output_counter: &AtomicUsize) -> Result<Vec<String>, Error> {
    let output_path = output.clone().join(format!("shuffled_doc_{:08}.jsonl.gz", counter.fetch_add(1, Ordering::SeqCst)));
    output_counter.fetch_add(1, Ordering::SeqCst);
    let contents: Vec<u8> = chunk.iter()
                   .flat_map(|s| s.as_bytes().iter().chain(std::iter::once(&b'\n')))
                   .cloned()
                   .collect();

    write_mem_to_pathbuf(&contents, &output_path).unwrap();
    Ok(Vec::new())
}

/*=================================================================
=                                  MAIN                           =
=================================================================*/

fn main() {
    let start_main = Instant::now();
    let args = ArgParser::parse();
    let paths = expand_dirs(args.input.clone(), None).unwrap();

    println!("Starting coarse shuffle...");
    let start_coarse = Instant::now();
    let local_cells = coarse_shuffle(&paths, &args.local_cell_storage, args.num_local_cells, args.remove_locals).unwrap();
    println!("Finished coarse shuffle in {:?} secs", start_coarse.elapsed().as_secs());

    println!("Writing chunks...)");
    let start_chunks = Instant::now();
    let num_output_files = finalize_chunks(local_cells, &args.output, args.docs_per_jsonl, args.remove_locals).unwrap();
    println!("Finished writing chunks in {:?} secs", start_chunks.elapsed().as_secs());

    println!("-------------------------");
    println!("Finishing data shuffle in {:?} seconds", start_main.elapsed().as_secs());    
    println!("Generated {:?} output files", num_output_files);

}