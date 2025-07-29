# A method of anti-backdoor learning for language models".
Code for AAAI 2025 paper "Backdoor Token Unlearning:  Exposing and Defending Backdoors in Pretrained Language Models"
[https://arxiv.org/abs/2501.03272](https://ojs.aaai.org/index.php/AAAI/article/view/34605/36760)

---


## Overview

BTU is designed to training clean model from poisoned datasets.

---

## Dataset

The AGNews dataset is not included due to its large size.  
Please download it from the [OpenBackdoor repository by THUNLP](https://github.com/thunlp/OpenBackdoor), which contains the relevant data splits used in this work.

---

## Installation

First, install project dependencies:

```
pip install -r requirements.txt
```

Ensure you're using Python 3.9. Dependencies and version info are listed in `requirements.txt`.

---

## Configuration

Edit the `config.json` file to customize settings such as:

- Path(s) of training data 
- Model path  
- Hyperparameters (learning rate, epochs, batch size, etc.)

Ensure all file paths and settings reflect your actual environment.

---

## Usage

Run the main script to start the pipeline:

```bash
python BTU.py
```

---

## Citation

If you use this code or method in your research, please cite the following:

```bibtex
@inproceedings{jiang2025backdoor,
  title={Backdoor Token Unlearning: Exposing and Defending Backdoors in Pretrained Language Models},
  author={Jiang, Peihai and Lyu, Xixiang and Li, Yige and Ma, Jing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={23},
  pages={24285--24293},
  year={2025}
}
```

---

## Acknowledgments

- Thanks for [OpenBackdoor repository](https://github.com/thunlp/OpenBackdoor).  
- Thanks to Peihai Jiang, Xixiang Lyu, Yige Li, and Jing Ma for their foundational research.

---

