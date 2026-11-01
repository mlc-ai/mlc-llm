<div align="center">

# MLC LLM

[![Installation](https://img.shields.io/badge/docs-latest-green)](https://llm.mlc.ai/docs/)
[![License](https://img.shields.io/badge/license-apache_2-blue)](https://github.com/mlc-ai/mlc-llm/blob/main/LICENSE)
[![Join Discoard](https://img.shields.io/badge/Join-Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/9Xpy2HGBuD)
[![Related Repository: WebLLM](https://img.shields.io/badge/Related_Repo-WebLLM-fafbfc?logo=github)](https://github.com/mlc-ai/web-llm/)

**基于机器学习编译技术的通用大语言模型（LLM）部署引擎**

[快速开始](https://llm.mlc.ai/docs/get_started/quick_start) | [官方文档](https://llm.mlc.ai/docs) | [技术博客](https://blog.mlc.ai/)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

</div>

## 关于本项目

MLC LLM 是一款面向大语言模型（LLM）的机器学习编译器与高性能部署引擎。本项目的使命是让每个人都能够在各种平台上原生开发、优化并部署 AI 模型。

<div align="center">
<table style="width:100%">
  <thead>
    <tr>
      <th style="width:15%"> </th>
      <th style="width:20%">AMD GPU</th>
      <th style="width:20%">NVIDIA GPU</th>
      <th style="width:20%">Apple GPU</th>
      <th style="width:24%">Intel GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linux / Win</td>
      <td>✅ Vulkan, ROCm</td>
      <td>✅ Vulkan, CUDA</td>
      <td>N/A</td>
      <td>✅ Vulkan</td>
    </tr>
    <tr>
      <td>macOS</td>
      <td>✅ Metal (dGPU)</td>
      <td>N/A</td>
      <td>✅ Metal</td>
      <td>✅ Metal (iGPU)</td>
    </tr>
    <tr>
      <td>Web 浏览器</td>
      <td colspan=4>✅ WebGPU 与 WASM</td>
    </tr>
    <tr>
      <td>iOS / iPadOS</td>
      <td colspan=4>✅ Apple A 系列 GPU 上的 Metal</td>
    </tr>
    <tr>
      <td>Android</td>
      <td colspan=2>✅ Adreno GPU 上的 OpenCL</td>
      <td colspan=2>✅ Mali GPU 上的 OpenCL</td>
    </tr>
  </tbody>
</table>
</div>

MLC LLM 将模型编译并运行在 **MLCEngine** 之上——这是一个横跨上述所有平台的统一高性能 LLM 推理引擎。MLCEngine 提供与 OpenAI 兼容的 API，可通过 REST 服务器、Python、JavaScript、iOS 和 Android 直接调用，所有平台均由我们与社区持续改进的同一套底层引擎与编译器提供支持。

## 快速开始

请访问我们的[官方文档](https://llm.mlc.ai/docs/)以开始使用 MLC LLM：
- [安装指南](https://llm.mlc.ai/docs/install/mlc_llm)
- [快速开始](https://llm.mlc.ai/docs/get_started/quick_start)
- [项目简介](https://llm.mlc.ai/docs/get_started/introduction)

## 引用说明（Citation）

如果您觉得本项目对您的研究或工作有所帮助，请考虑引用我们的项目：

```bibtex
@software{mlc-llm,
    author = {{MLC team}},
    title = {{MLC-LLM}},
    url = {https://github.com/mlc-ai/mlc-llm},
    year = {2023-2025}
}
```

MLC LLM 所依托的核心底层技术包括：

<details>
  <summary>参考文献（点击展开）</summary>

  ```bibtex
  @inproceedings{tensorir,
      author = {Feng, Siyuan and Hou, Bohan and Jin, Hongyi and Lin, Wuwei and Shao, Junru and Lai, Ruihang and Ye, Zihao and Zheng, Lianmin and Yu, Cody Hao and Yu, Yong and Chen, Tianqi},
      title = {TensorIR: An Abstraction for Automatic Tensorized Program Optimization},
      year = {2023},
      isbn = {9781450399166},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3575693.3576933},
      doi = {10.1145/3575693.3576933},
      booktitle = {Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
      pages = {804–817},
      numpages = {14},
      keywords = {Tensor Computation, Machine Learning Compiler, Deep Neural Network},
      location = {Vancouver, BC, Canada},
      series = {ASPLOS 2023}
  }

  @inproceedings{metaschedule,
      author = {Shao, Junru and Zhou, Xiyou and Feng, Siyuan and Hou, Bohan and Lai, Ruihang and Jin, Hongyi and Lin, Wuwei and Masuda, Masahiro and Yu, Cody Hao and Chen, Tianqi},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
      pages = {35783--35796},
      publisher = {Curran Associates, Inc.},
      title = {Tensor Program Optimization with Probabilistic Programs},
      url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf},
      volume = {35},
      year = {2022}
  }

  @inproceedings{tvm,
      author = {Tianqi Chen and Thierry Moreau and Ziheng Jiang and Lianmin Zheng and Eddie Yan and Haichen Shen and Meghan Cowan and Leyuan Wang and Yuwei Hu and Luis Ceze and Carlos Guestrin and Arvind Krishnamurthy},
      title = {{TVM}: An Automated {End-to-End} Optimizing Compiler for Deep Learning},
      booktitle = {13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)},
      year = {2018},
      isbn = {978-1-939133-08-3},
      address = {Carlsbad, CA},
      pages = {578--594},
      url = {https://www.usenix.org/conference/osdi18/presentation/chen},
      publisher = {USENIX Association},
      month = oct,
  }
  ```
</details>

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月26日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
