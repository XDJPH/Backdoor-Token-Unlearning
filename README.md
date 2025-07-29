# 🧠 Backdoor Token Unlearning (BTU)

**Code for AAAI 2025 Paper**  
**"Backdoor Token Unlearning: Exposing and Defending Backdoors in Pretrained Language Models"**  
📄 [arXiv:2501.03272](https://arxiv.org/abs/2501.03272)  
📘 [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34605/36760)

---

## 📝 Overview

**Backdoor Token Unlearning (BTU)** is a novel anti-backdoor learning method designed to **train clean language models from poisoned datasets**.  
The method identifies and neutralizes backdoor triggers by unlearning their influence in token representations, achieving robust defense with minimal performance degradation on clean tasks.

---

## 📂 Dataset

The AGNews dataset used in our experiments is **not included** in this repository due to its size.  
Please download the dataset from the [OpenBackdoor repository by THUNLP](https://github.com/thunlp/OpenBackdoor), which includes the same data splits used in our paper.

---

## ⚙️ Installation

Ensure you're using **Python 3.9**. Then install the required dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains all necessary libraries and specific version constraints for reproducibility.

---

## ⚙️ Configuration

Customize your training setup by modifying the `config.json` file. You can specify:

- **Dataset paths** (tasks and datasets)
- **Model paths** (pretrained checkpoints)
- **Training hyperparameters**, such as:
  - `learning_rate`
  - `epochs`
  - `batch_size`
- **Unlearning parameters**:
  - Threshold

Ensure all paths and settings reflect your actual environment before running the script.

---

## 🚀 Usage

To start the BTU pipeline, simply run:

```bash
python BTU.py
```

Intermediate logs, model checkpoints, and evaluation results will be saved to the specified output directory in your configuration.

---

## 📈 Results Summary

Our BTU method demonstrates:

- **>90% reduction in backdoor attack success rate**
- **<1% drop in clean task accuracy**
- No need for clean validation data or known trigger patterns

For more results, refer to **Table 2** in the [paper](https://arxiv.org/abs/2501.03272).

---

## 📖 Citation

If you use this codebase or method in your research, please cite the following work:

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

## 🙏 Acknowledgments

- This project builds upon the [OpenBackdoor](https://github.com/thunlp/OpenBackdoor) framework by THUNLP.
- We thank the authors — **Peihai Jiang**, **Xixiang Lyu**, **Yige Li**, and **Jing Ma** — for their foundational work on backdoor defense for language models.

---

## 📬 Contact

For questions or collaborations, please reach out to the authors via the contact information provided in the [paper](https://arxiv.org/abs/2501.03272).

---
