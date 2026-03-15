<div align="center">
<h1>3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer</h1>

<a href="https://arxiv.org/abs/2501.01163"><img src="https://img.shields.io/badge/arXiv-2501.01163-b31b1b" alt="arXiv"></a>
<a href='https://huggingface.co/datasets/djiajunustc/3D-LLaVA-Data'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Processed_Data-blue'></a>
<a href="https://huggingface.co/djiajunustc/3D-LLaVA-7B-LoRA" target="_blank"><img src="https://img.shields.io/badge/Checkpoint-Orange" alt="checkpoint"></a>

[Jiajun Deng](https://djiajunustc.github.io/), [Tianyu He](https://www.microsoft.com/en-us/research/people/tianyuhe/), [Li Jiang](https://llijiang.github.io/), [Tianyu Wang](https://openreview.net/profile?id=~Tianyu_Wang5), [Feras Dayoub](https://ferasdayoub.com/), [Ian Reid](https://researchers.adelaide.edu.au/profile/ian.reid)
</div>

```bibtex
@inproceedings{deng20253dllava,
  title={3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer},
  author={Deng, Jiajun and He, Tianyu and Jiang, Li and Wang, Tianyu and Dayoub, Feras and Reid, Ian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Overview
<div style="text-align: center;">
    <img src="docs/framework.png" alt="Dialogue_Teaser" width=100% >
</div>
3D-LLaVA (CVPR 2025) is 3D Large Multimodal Model that takes point clouds and text instruction as input to perform VQA, Dense Captioning and 3D Referring Segmentation. At the core of 3D-LLaVA is a new Omni Superpoint Transformer (OST), which integrates three functionalities: (1) a visual feature selector that converts and selects visual tokens, (2) a visual prompt encoder that embeds interactive visual prompts into the visual token space, and (3) a referring mask decoder that produces 3D masks based on text description.


## Environment

### Docker

We provide the Docker Image to run our 3D-LLaVA. Please run the following code to pull the docker image:
```
docker pull djiajun1206/3d-llava-slim
```

### Conda

We provide the conda environment to run our 3D-LLaVA. Please run the following code to create the conda environment:
```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]" --no-build-isolation

pip install flash-attn==2.3.6 --no-build-isolation
pip install git+https://github.com/openai/CLIP.git
pip install spconv-cu120
pip install -U mmengine pycocoevalcap

# precise eval
cd libs/pointops
python setup.py install
cd ../..

conda install -c bioconda google-sparsehash 
cd libs/pointgroup_ops
python setup.py install --include_dirs=${CONDA_PREFIX}/include
cd ../..
```


## Data
We conduct experiments with the scans data from Scannet, as well as the text description from ScanRefer, ScanQA, SQA3D, ReferIt3D and Multi3DRefer. To enable conventiently getting access to the data, we provide the [processed data](https://huggingface.co/datasets/djiajunustc/3D-LLaVA-Data). The data are supposed to be placed in ./playground, and the data structure is as follows:
```
3D-LLaVA # project root
|── playground
|   |── data
│   |   ├── scannet
│   |   │   ├── super_points
|   │   │   ├── train
|   │   │   ├── val
|   │   │   └── scannet_axis_align_matrix_trainval.pkl
│   |   ├── train_info
│   │   |   ├── scanqa_train_3d_llava.json
│   │   |   ├── sqa3d_train_3d_llava.json
│   │   |   ├── scan2cap_train_3d_llava.json
│   │   |   ├── ...
│   │   └── eval_info
│   │   |   ├── scanqa
│   │   |   ├── sqa3d
│   │   |   ├── densecap_scanrefer
│   │   |   ├── ...
```

## Training
We exploit LoRA tuning by default. Please train the 3D-LLaVA with:
```
./scripts/train/finetune-3d-llava-lora.sh
```

## Evaluation
We provide the scripts to evaluate our model on ScanQA, SQA3D, Scan2Cap, ScanRefer, Multi3DRefer. Please run:
```
./scripts/eval/multigpu_eval_sqa3d.sh

./scripts/eval/multigpu_eval_scanqa.sh

./scripts/eval/multigpu_eval_scan2cap.sh

./scripts/eval/multigpu_eval_scanrefer.sh

./scripts/eval/multigpu_eval_multi3drefer.sh
```

## Acknowledgements
Thanks to the following great repositories: [LLaVA](https://github.com/haotian-liu/LLaVA), [PonderV2](https://github.com/OpenGVLab/PonderV2), [OneFormer3d](https://github.com/filaPro/oneformer3d).