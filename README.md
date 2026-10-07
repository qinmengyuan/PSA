<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Proactive Safety Alignment: Enabling Proactive Initiative In Multi-Turn Defense </h1>

<p align="center">
  <a href="https://2026.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP-2026-blue" alt="EMNLP 2026"></a>
  <a href="http://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License: Apache 2.0"></a>
</p>

<!-- Warning -->
<p align="center" style="color: red;">
    <b><em>Warning: This paper contains model outputs that may be considered offensive.</em></b>
</p>



![](framework.png)

Official implementation of **Proactive Safety Alignment (PSA)**, a framework that anticipates user intent proactively to align LLM agents' outputs with safety objectives.   PSA combines Safe User Request Prediction (SURP), which anticipates likely safe future requests, and Potentially Harmful Generation (PHG), which evaluates whether the current interaction state can support misuse.     Evaluations across multiple benchmarks demonstrate that  PSA substantially reduces attack success rates, while preserving overall utility on MT-Bench.  Our results show that PSA provides an effective and scalable paradigm for multi-turn LLM safety alignment.



## How to use our project

Before running our code, 
#### Build the environment
 - Download our Apptainer image.
 - [Optional] We provide trained SURP and PHG agents or you can train them by yourself.


#### Run our code

You only need to launch your target model and specify the port in the configuration, then simply run `utils/llm_proxy.py` and send request messages to it.

To train the three components yourself, see `train/qwen_based/` for switchable SURP / PHG / COG models.
