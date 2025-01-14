<div align="center">

# MLC LLM

[![Installation](https://img.shields.io/badge/docs-latest-green)](https://llm.mlc.ai/docs/)
[![License](https://img.shields.io/badge/license-apache_2-blue)](https://github.com/mlc-ai/mlc-llm/blob/main/LICENSE)
[![Join Discoard](https://img.shields.io/badge/Join-Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/9Xpy2HGBuD)
[![Related Repository: WebLLM](https://img.shields.io/badge/Related_Repo-WebLLM-fafbfc?logo=github)](https://github.com/mlc-ai/web-llm/)

**Universal LLM Deployment Engine with ML Compilation**

[Get Started](https://llm.mlc.ai/docs/get_started/quick_start) | [Documentation](https://llm.mlc.ai/docs) | [Blog](https://blog.mlc.ai/)

</div>

## About

MLC LLM is a machine learning compiler and high-performance deployment engine for large language models.  The mission of this project is to enable everyone to develop, optimize, and deploy AI models natively on everyone's platforms. 

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
      <td>Web Browser</td>
      <td colspan=4>✅ WebGPU and WASM </td>
    </tr>
    <tr>
      <td>iOS / iPadOS</td>
      <td colspan=4>✅ Metal on Apple A-series GPU</td>
    </tr>
    <tr>
      <td>Android</td>
      <td colspan=2>✅ OpenCL on Adreno GPU</td>
      <td colspan=2>✅ OpenCL on Mali GPU</td>
    </tr>
  </tbody>
</table>
</div>

MLC LLM compiles and runs code on MLCEngine -- a unified high-performance LLM inference engine across the above platforms. MLCEngine provides OpenAI-compatible API available through REST server, python, javascript, iOS, Android, all backed by the same engine and compiler that we keep improving with the community.

## Get Started

Please visit our [documentation](https://llm.mlc.ai/docs/) to get started with MLC LLM.
- [Installation](https://llm.mlc.ai/docs/install/mlc_llm)
- [Quick start](https://llm.mlc.ai/docs/get_started/quick_start)
- [Introduction](https://llm.mlc.ai/docs/get_started/introduction)

## Citation

Please consider citing our project if you find it useful:

```bibtex
@software{mlc-llm,
    author = {{MLC team}},
    title = {{MLC-LLM}},
    url = {https://github.com/mlc-ai/mlc-llm},
    year = {2023-2025}
}
```

The underlying techniques of MLC LLM include:

<details>
  <summary>References (Click to expand)</summary>

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
