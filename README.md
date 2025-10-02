## Depth Pro + Parallax Effect

This software project is based on the research paper:
**[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**, 
*Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun*.

Thanks for their work on precise depth estimation. Baesd on this, I implement a simple parallax effect. This effect is made up with a depth-based pixel movement estimation, a back-to-forth pixel assignment and  Telea's traditional image-restoring algorithm (Alexandru Telea, 2004). The code is low in optimization.

This project is soly available on Windows.

## Getting Started

We recommend setting up a virtual environment. Using e.g. miniconda, the `depth_pro` package can be installed via:

```bash
conda create -n depth-pro -y python=3.10.18
conda activate depth-pro

pip install -e .
```

Besides Depth Pro, additional packages in need are also supplemented in this installation.

### How to run

Following `get_parallax.bat`, simply run the command below on Windows PowerShell or other terminals.
```bash
get_parallax.bat
```

### How to use Depth Pro

To download pretrained checkpoints follow the code snippet below:
```bash
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
```

We provide a helper script to directly run the model on a single image:
```bash
# Run prediction on a single image:
depth-pro-run -i ./data/example.jpg -o ./outputs/
# Run `depth-pro-run -h` for available options.
```
