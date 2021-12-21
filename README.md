# diffusion-models

The aim of *diffusion-models* is to trace the history and evolution of diffusion models for sampling from a data distribution. The resulting content will feature theory and toy examples written in Python/PyTorch. Expect something along the lines of my [autoregressive repository](https://github.com/cheind/autoregressive).

 However, currently this package is not meant to be used by anyone than myself, because code is frequently changing and lacking documentation.

## References

```bibtex
@article{hyvarinen2005estimation,
  title={Estimation of non-normalized statistical models by score matching.},
  author={Hyv{\"a}rinen, Aapo and Dayan, Peter},
  journal={Journal of Machine Learning Research},
  volume={6},
  number={4},
  year={2005}
}

@inproceedings{song2020sliced,
  title={Sliced score matching: A scalable approach to density and score estimation},
  author={Song, Yang and Garg, Sahaj and Shi, Jiaxin and Ermon, Stefano},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={574--584},
  year={2020},
  organization={PMLR}
}


@article{song2019generative,
  title={Generative modeling by estimating gradients of the data distribution},
  author={Song, Yang and Ermon, Stefano},
  journal={arXiv preprint arXiv:1907.05600},
  year={2019}
}


@article{vincent2011connection,
  title={A connection between score matching and denoising autoencoders},
  author={Vincent, Pascal},
  journal={Neural computation},
  volume={23},
  number={7},
  pages={1661--1674},
  year={2011},
  publisher={MIT Press}
}

```

Additional
https://courses.cs.washington.edu/courses/cse599i/20au/resources/L17_denoising.pdf
How to Train Your Energy-Based Models
https://arxiv.org/pdf/2101.03288.pdf
A Low Rank Approach to Automatic Differentiation
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.7201&rep=rep1&type=pdf
Sliced Score Matching
https://arxiv.org/pdf/1905.07088.pdf
https://arxiv.org/pdf/2101.09258.pdf