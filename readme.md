# LLM 推理服务吞吐测试

本仓库包含了测试常见推理服务的吞吐的代码，以及相关的测试结果。

## 硬件

阿里云 GPU 实例， 具体型号为 [ecs.gn7i-c32g1.8xlarge](https://help.aliyun.com/zh/ecs/user-guide/overview-of-instance-families#title-0xr-tb2-8ac)。

* CPU: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz * 32 vCPU
* RAM: 188G
* GPU: [NVIDIA A10](https://www.techpowerup.com/gpu-specs/a10-pcie.c3793)

## 测试的模型

Llama 2 系列：
* TheBloke/Llama-2-7B-Chat-AWQ
* TheBloke/Llama-2-7B-Chat-GPTQ
* meta-llama/Llama-2-7b-chat-hf

Qwen 系列：
* Qwen/Qwen-7B-Chat
* Qwen/Qwen-7B-Chat-Int4
* TheBloke/Qwen-7B-Chat-AWQ

Mistral 系列:
* WIP

## 参测推理服务

*   [vllm](https://github.com/vllm-project/vllm)
    支持 Qwen 系列模型的推理，支持 AWQ 量化方式。
*   [vllm-gptq](https://github.com/chu-tianxiang/vllm-gptq/)
    为 vllm 添加了 GPTQ 支持，目前采用了 exllamav2 的 gptq kernel
*   [text-generation-inference](https://github.com/huggingface/text-generation-inference)
    没有千问的支持

## 压测方法

`benchmark.py` 为主要的压测脚本实现，实现了一个 naive 的 asyncio + ProcessPoolExecutor 的压测框架。

在发送请求时，目前基本为不做等待的直接并行发送请求，这可能无法利用好 PagedAttention 的节约显存的特性。在解读结果时可能需要读者注意。

对于不同的模型，Prompt 有一些调整，基本为让模型输出 0 ~ 100 的数字作为 benchmark。

`results` 文件夹下包含了脚本输出的原始的测试结果数据，可以利用 draw.ipynb 进行绘图。

## 结果

### Llama2 7B

![](images/llama2-7b-throughput.png)
![](images/llama2-7b-first-token-lat.png)
![](images/llama2-7b-avglat.png)

### Qwen 7B

![](images/qwen-7b-token_per_s.png)
![](images/qwen-7b-avg_first_token_latency.png)
![](images/qwen-7b-avg_token_latency.png)