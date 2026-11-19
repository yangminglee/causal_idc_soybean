# causal_idc_soybean

Code for "Piercing the Genomic Prediction Black Box: Reconstructing the Genetic Architecture of Soybean IDC using Topological Causal Networks".

## Environment

Python 3.10. With uv:

    uv sync

## Data

The genotype and phenotype data (1,000 lines, 1,200 SNPs) are from Xu et al. (2021), PLOS ONE 16:e0240948, and are not included here. Put the file in `data/IDC_data_for_MS2_PLOSOne.csv`. Expected columns: `LINCD`, `IDC_class` (1 = retain, 0 = discard), `IDC_score` (1 = retain, 9 = discard) and `Marker0001` ... `Marker1200` coded -1/0/1.

## Running

Full-data pipeline (PCA, DoubleML screening, PC algorithm, 10-fold validation):

    uv run python idc_compute.py --in_dir data --out_dir runs/E0 --n_jobs 8
    uv run python idc_compute.py --in_dir data --out_dir runs/E2 --n_jobs 8 --resid_pcs

Nested cross-validation, 3 repeats x 10 folds (each call is one outer fold and can run in parallel):

    for r in 0 1 2; do for k in $(seq 1 10); do
      uv run python task_nested_cv.py --in_dir data --out_dir runs/B2 --repeat $r --fold $k --n_jobs 2
    done; done

Panel-size and screening-off evaluation on the nested folds:

    uv run python causal_track.py --in_dir data --b2 runs/B2 --out runs/CT

Figures and tables:

    uv run python revision_analysis.py --e0 runs/E0 --e2 runs/E2 --b2 runs/B2 --ct runs/CT --out runs/figures
    uv run python revision_tables.py --analysis runs/figures --ct runs/CT --out runs/tables

One run of the PC algorithm takes about 30-60 minutes on a 32-thread workstation. Random seeds are fixed, so repeated runs give the same results.

## License

MIT
