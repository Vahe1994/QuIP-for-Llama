# QuIP: Quantization with Incoherence Processing

This repository contains code for the paper [**QuIP: 2-Bit Quantization of Large Language Models with Guarantees**](https://arxiv.org/pdf/2307.13304.pdf). 

**TLDR:** Our proposed incoherence processing enables quantization of large language models down to 2 bits.
Please see our paper for full details.

The code is built on top of [QuIP](https://github.com/jerry-chee/QuIP), [GPTQ](https://github.com/IST-DASLab/gptq), and [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa). The current code includes the following: 

## Language Generation

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/llama-2-7b-hf c4
# Run a quantization method with Incoherence Processing
CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/llama-2-7b-hf c4 --wbits 4 --quant <quantmethod> --incoh_processing --save <savename>
# Run a quantization method with baseline processing
CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/llama-2-7b-hf c4 --wbits 4 --quant gptq --pre_gptqH --save <savename>
````

Quantization methods include:
- `ldlq`: runs the LDLQ rounding algorithm (we show its equivalence to GPTQ, providing a novel theoretical analysis)
- `ldlqRG`: runs the LDLQ_RG algorithm with additional hessian-based hessian reordering, and further greedy updates, with `--npasses` controlling the number of passes over the weights
- `gptq`: runs GPTQ algorithm as implemented by its authors
- `allbal`: algorithm to run greedy updates by themselves, with `--npasses` the argument controlling the number of passes over the weights
- `ldlbal_admm`: alternative algorithm which constraints the rounded weights to be sufficiently close to their original, giving a better theoretical bound.

The `--incoh_processing` argument is a meta argument which sets the following flags `--pre_gptqH --pre_rescale --pre_proj --qfn b`. 
For more control into the pre and post processing, these arguments can be set individually.

To run other Llama models replace `llama-2-7b-hf` with the other size variants.
On larger models, a low compute-to-memory-access ratio can slow down the quantization algorithms. 
We implement a lazy batch update to te weight matrix specified by `--lazy_batch`.
This argument works with the quantization methods {ldlq, ldlqRG, allbal}.
Note GPTQ already implements this, and is where we got the idea from.


## GPTQ and LDLQ Equivalence
Run the following script to empirically verify that the output of GPTQ's implemenation and our implemenation of LDLQ are identical: `python gptq_ldlq_equiv.py`.
Note GPTQ's implementation requires running on a GPU.

## GTPQ/LDLQ Finite Grid Counterexample
Run `python gptq_counter.py` to compute the proxy loss of our W,H counterexample. 

## Computing Proxy Loss
In a similar manner to `llama.py`, run `llama_saveH.py` to save the H matrices resulting from the specified model and quantization method.
Then, run `llama_proxy.py` to compute the proxy loss for a specified quantization method. 
```
CUDA_VISIBLE_DEVICES=0 python llama_proxy.py c4 --wbits 4 --quant <quant_method>
```

## H Summary
Run the following script to compute summary statistics of a folder `<dirname>` of H matrices, output from running `llama_saveH.py`. 
```
python compute_Hsummary.py --dirname <> --savename <> 
```
