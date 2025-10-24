## Code for paper: Hybrid Monte Carlo Metadynamics

In paper, *[Hybrid Monte Carlo Metadynamics](https://arxiv.org/abs/2508.15942)*, we introduce a new algorithm that integrates the Hybrid Monte Carlo algorithm with Metadynamics, which we term "hybridMC-MetaD" (hMCMetaD in this git repository).

* `algorithm` folder: implementation of hybridMC-MetaD (hMCMetaD) algorithm and an illustrative example on a double-well model system
* `updater` folder: implementation of hybridMC-MetaD (hMCMetaD) as a HOOMD-blue custom updater and an example of using it on a hard bipyramid system