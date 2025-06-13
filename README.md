# Diffusion Speech Quality Assesment (SQA)

This repository contains the official PyTorch implementations for the paper:
- Danilo de Oliveira, Julius Richter, Jean-Marie Lemercier, Simon Welker, Timo Gerkmann, [*"Non-intrusive Speech Quality Assessment with Diffusion Models Trained on Clean Speech"*](https://arxiv.org/abs/2410.17834), accepted at ISCA Interspecch 2025. [[bibtex]](#citations--references)

The code is largely based on the repository of [1]

## Installation

- Create a new virtual environment with Python >= 3.11 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt` and let pip resolve the dependencies for you
- Install the EDM2 code as a submodule: `git submodule update --init --recursive`

## Data

For training, we use the [EARS-WHAM](https://github.com/sp-uhh/ears_benchmark) dataset.

## Pretrained checkpoints

- The checkpoint used in the paper can be downloaded [here](https://www2.informatik.uni-hamburg.de/sp/audio/publications/interspeech2025-diffusion-sqa/checkpoints/phema-0037748-0.080.pkl)

## Training

Training is done by executing `train_sqa.py`. A minimal running example with default settings (as in our paper) can be run with

```bash
torchrun --standalone --nproc_per_node=<num-gpus> train_sqa.py --outdir=<log-dir> --data=<path-to-trainset> --batch-gpu=<batch-size-per-gpu>
```

where `<path-to-trainset>` should be a path to a folder containing clean `.wav` files (subdirectories are also supported).

## EMA Reconstruction

To reconstruct a new EMA profile with length 0.08, run

```bash
python edm2/reconstruct_phema.py --indir=<log-dir> --outdir=<reconstructed-ema-dir> --outstd=0.080
```

For more detailed on post-hoc EMA reconstruction, please refer to the [EDM2 repository](https://github.com/NVlabs/edm2/tree/main).

## SQA

To calculate the diffusion log likelihoods on a test set and save them in a csv file, run

```bash
torchrun --standalone --nproc_per_node=<num-gpus> calculate_likelihood.py --checkpoint=<path-to-pkl> --data_dir=<path-to-testset> --output_file=<path-to-csv>
```

The `--checkpoint` parameter should be the path to a snapshot or a reconstructed EMA profile.

## License

The code and checkpoints are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citations / References

We kindly ask you to cite our papers in your publication when using any of our research or code:
```bib
@misc{deoliveira2024nonintrusivespeechqualityassessment,
    title={Non-intrusive Speech Quality Assessment with Diffusion Models Trained on Clean Speech}, 
    author={Danilo de Oliveira and Julius Richter and Jean-Marie Lemercier and Simon Welker and Timo Gerkmann},
    year={2024},
    eprint={2410.17834},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2410.17834}, 
}
```
> [1] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, Samuli Laine, ["Analyzing and Improving the Training Dynamics of Diffusion Models"](https://arxiv.org/abs/2312.02696), CVPR 2024. [[Code]](https://github.com/NVlabs/edm2/tree/main)