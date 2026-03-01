# PSDZH
Official PyTorch implementation of  
**Progressive Semantic-Driven Hashing for Zero-Shot Image Retrieval**


## Experimental Results

We evaluate PSDZH on three benchmark datasets:

- **AWA2**
- **CUB**
- **SUN**

Under different hash code lengths (16 / 32 / 64 / 128 bits), PSDZH consistently achieves superior mAP performance compared with existing zero-shot hashing methods.
# Dataset Preparation
Download datasets from their official sources and organize them as:
data/
  ├── AWA2/
  ├── CUB/
  └── SUN/
Preprocessing scripts are provided in the data/ directory.
# Training
Example training command:
python train.py
