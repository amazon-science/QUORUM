# QUORUM: QUality-Optimized Routing Using Multiple annotators

Data annotation remains a central bottleneck in natural language processing, requiring human effort to obtain high-quality labels at scale. While Large Language Models (LLMs) offer a fast and cost-effective alternative, their reliability is highly instance-dependent: they perform well on simple inputs but often fail on examples requiring nuanced reasoning or contextual understanding. In this work, we address this challenge with QUORUM (QUality-Optimized Routing Using Multiple annotators), a budget-aware routing framework that dynamically assigns each instance to human or LLM annotators under a fixed annotation budget. Unlike prior approaches relying on model confidence or uncertainty estimates, QUORUM leverages feature-based signals to estimate instance difficulty and supports multiple annotations per instance, combining them through agreement-based rewards to improve reliability. We evaluate QUORUM across diverse closed- and open-ended annotation tasks in English and multilingual settings, and QUORUM improves annotation quality by up to 34.4% while reducing costs by 8.8% over competing methods.

In order to run the code (tested on `Python 3.11.14`):

```
pip install -r requirements.txt
```

Then, to start a single experiment:

```
python3 main.py --dataset NAME_OF_THE_DATASET --eval_type EVAL_TYPE --budget BUDGET --annotator_cost ANNOTATOR_COST --methods METHODS
```

Example:

```
python3 main.py  --dataset pubmed --eval_type dollars --backup  --budget 444 --methods QUORUM
```

To run an experiment with multiple seeds:
```
python3 multi.py --dataset NAME_OF_THE_DATASET --eval_type EVAL_TYPE --budget BUDGET --annotator_cost ANNOTATOR_COST --methods METHODS
```

Example:

```
python3 multi.py  --dataset pubmed --eval_type dollars --n_seeds 3 --methods QUORUM
```

Results by default are saved in the `results` folder.


- NAME_OF_THE_DATASET could be: `pubmed`, `imdb`, `global-mmlu`, `mmlu-redux`, `xlsum`, `cnn`, `agnews`.
- In case you are using `XLSum`, select the language using `--language` (`spanish`, `japanese`). Same for `global-mmlu` (`ja`, `es`).
- EVAL_TYPE could be either `auditor_style` (one annotation per sample) or `dollars` (keeping into account the monetary constraint).
- ANNOTATOR_COST is a list which represent the cost of the LLMs (i.e., 0.01 0.03 0.05 0.1).
- METHODS could be `QUORUM, Random, SANT, CoAnnotating, PPI, Araida`.

SANT and ARAIDA could have small compatibility issues with `FastText`.

## Citation
If you use this code in your research or project, please cite us:
```bibtex
@misc{purificato2026quorumqualityoptimizedroutingusing,
      title={QUORUM: QUality-Optimized Routing Using Multiple annotators}, 
      author={Antonio Purificato and Maria Sofia Bucarelli and Andrea Bacciu and Amin Mantrach and Fabrizio Silvestri},
      year={2026},
      eprint={2608.27974},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.27974}, 
}
```
For doubts or errors feel free to ping purificato@diag.uniroma1.it!


## Acknowledgments

The implementation of competitor methods draws from the papers:
- [Can Unconfident LLM Annotations Be Used for Confident Conclusions?](https://aclanthology.org/2025.naacl-long.179/))
- [ARAIDA: Analogical Reasoning-Augmented Interactive Data Annotation](https://aclanthology.org/2024.acl-long.574/)
- [CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation](https://aclanthology.org/2023.emnlp-main.92/)
- [HyPAC: Cost-Efficient LLMs-Human Hybrid Annotation with PAC Error Guarantees](https://arxiv.org/abs/2602.02550)
- [Selective Annotation via Data Allocation: These Data Should Be Triaged to Experts for Annotation Rather Than the Model](https://aclanthology.org/2024.findings-emnlp.17/)

We gratefully acknowledge the authors for making their code available.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## License

This library is licensed under the CC-BY-NC-4.0 License.

