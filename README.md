<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Proactive Safety Alignment: Enabling Proactive Initiative In Multi-Turn Defense </h1>

<p align="center">
  <a href="https://2026.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP-2026-blue" alt="EMNLP 2026"></a>
  <a href="https://aclanthology.org/2026.findings-emnlp.2208/"><img src="https://img.shields.io/badge/Paper-EMNLP%202026-B31B1B" alt="Paper"></a>
  <a href="http://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License: Apache 2.0"></a>
</p>

<!-- Warning -->
<p align="center" style="color: red;">
    <b><em>Warning: This paper contains model outputs that may be considered offensive.</em></b>
</p>



![](framework.png)

Official implementation of **Proactive Safety Alignment (PSA)**, a framework that anticipates user intent proactively to align LLM agents' outputs with safety objectives.   PSA combines Safe User Request Prediction (SURP), which anticipates likely safe future requests, and Potentially Harmful Generation (PHG), which evaluates whether the current interaction state can support misuse.     Evaluations across multiple benchmarks demonstrate that  PSA substantially reduces attack success rates, while preserving overall utility on MT-Bench.  Our results show that PSA provides an effective and scalable paradigm for multi-turn LLM safety alignment.

**Paper**: [Proactive Safety Alignment: Enabling Proactive Initiative in Multi-Turn Defence](https://aclanthology.org/2026.findings-emnlp.2208/) (EMNLP 2026)




## How to use our project

Before running our code, 
#### Build the environment
 - Download our Apptainer image.
 - [Optional] We provide trained SURP and PHG agents or you can train them by yourself.


#### Run our code

You only need to launch your target model and specify the port in the configuration, then simply run `utils/llm_proxy.py` and send request messages to it.

To train the three components yourself, see `train/qwen_based/` for switchable SURP / PHG / COG models.

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{qin-feng-2026-proactive,
    title = "Proactive Safety Alignment: Enabling Proactive Initiative in Multi-Turn Defence",
    author = "Qin, Mengyuan  and
      Feng, Yue",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.2208/",
    pages = "47300--47311",
    ISBN = "979-8-89176-462-0",
    abstract = "LLMs are increasingly deployed as interactive agents capable of multi-turn dialogues and tool usage, but they remain vulnerable to adaptive attacks. Existing safety mechanisms are mostly reactive, only attempting to uncover pre-existing hidden malicious intent, which makes them largely ineffective against attacks when such attacks are inherently designed to conceal their intent, and causes them to often fail in multi-turn, interactive, or out-of-distribution scenarios. In addition, existing multi-turn defences are largely data-driven, which limits their generalisation ability. To address this challenge, we propose Proactive Safety Alignment (PSA), a framework that anticipates user intent proactively to align LLM agents' outputs with safety objectives. PSA combines Safe User Request Prediction (SURP), which anticipates likely safe future requests, and Potentially Harmful Generation (PHG), which evaluates whether the current interaction state can support misuse. Evaluations across multiple benchmarks demonstrate that PSA substantially reduces attack success rates, while preserving overall utility on MT-Bench. Our results show that PSA provides an effective and scalable paradigm for multi-turn LLM safety alignment. The code is available at https://github.com/qinmengyuan/PSA."
}
```
