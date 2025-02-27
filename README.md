# ViFT: Do we Really Need Visual Instructions? Towards Visual Instruction-Free Fine-tuning for Large Vision-Language Models

## 😀 Overview

+ ViFT is the first instruction-free fine-tuning method with comparable performance to SOTA LVLMs.

+ We specially designed the training and inference methods for disentangling and combining natural language task-solving and visual perception abilities, to efficiently improve the multimodal capabilities of LVLMs.

+ Our ViFT is a low-cost approach for scaling data to improve LVLMs. Experimental results demonstrate the effectiveness of our approach on several benchmarks.

<p align="center">
  <img src="./assets/ViFT.png" width="75%" height="75% title="The overview of DAWN-ICL" alt="">
</p>

## 🚀 Quick Start

### Requirements

- python
- transformers
- pytorch
- accelerate
- openai==0.28.0

### Data Preparation and Training Setting

We prepare all the training datasets here:

- first_stage_captions.json contains 1m web captions sampled from laion, which is used for first-stage training(or pretraining, which can be skipped with compromised performance)

- second_stage_captions.json contains 1.7M high-quality captions collected or synthesized via Qwen2-VL-72B. We collect image from different data source, encompassing a broad range of visual domain and visual tasks.

- text_data_magpie.json and text_data_qwen_distill.json contain text-only data distilled from Qwen2.5-72B for inheriting and enhancing task-solving ability from LLM. They are mixed with second_stage_captions for second-stage training.




### Evaluation

Coming soon


We are still working on orangizing the remaining codes.

## 🌟 Results

<p align="center">
  <img src="./assets/result.png" width="75%" height="75% title="The overview of DAWN-ICL" alt="">
</p>

## Related Projects

- [Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA)
- [Bunny: A family of lightweight multimodal models](https://github.com/BAAI-DCAI/Bunny)
- [Steer LLM outputs with activation engineering](https://github.com/Mihaiii/llm_steer)

