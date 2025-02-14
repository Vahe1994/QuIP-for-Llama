import time

import torch
import torch.nn as nn

from gptq import *
from bal import Balance
from near import Nearest
from modelutils import *
from quant import *

from tqdm import tqdm

from llama import get_llama


@torch.no_grad()
def llama_sequential_proxy(model, dev, args, proxy_layers, load_H):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    print('Ready.')

    quantizers = {}
    errors, Hmags, times = [], [], []
    for i in tqdm(proxy_layers):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        quant_method = {}
        # Initialize Quant Method
        for name in subset:
            if args.quant == 'gptq':
                quant_method[name] = GPTQ(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant == 'nearest':
                quant_method[name] = Nearest(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant == 'gptq_updown':
                quant_method[name] = GPTQ_UD(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant in ['bitbal','parbal','allbal','ldlbal']:
                quant_method[name] = Balance(subset[name])
                quant_method[name].configure(
                                    args.quant,
                                    args.wbits, 
                                    args.npasses,
                                    unbiased=args.unbiased)
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)

        # Load H & Quantize
        for name in subset:
            fname = f'{load_H}/H_model.layers.{i}.{name}.pt'
            del quant_method[name].H
            quant_method[name].H = torch.load(fname, 
                    map_location=quant_method[name].layer.weight.device).to(torch.float32)

            quant_method[name].preproc(
                                preproc_gptqH=False, percdamp=False,
                                preproc_rescale=False,
                                preproc_proj=False, preproc_proj_extra=0)
            if args.quant == 'gptq':
                quant_method[name].fasterquant(groupsize=args.groupsize)
                quantizers['model.layers.%d.%s' %
                           (i, name)] = quant_method[name].quantizer
            if args.quant == 'gptq_updown':
                quant_method[name].fasterquant_updown(groupsize=args.groupsize)
            elif args.quant in ['bitbal','parbal','allbal','ldlbal']:
                quant_method[name].fasterquant()
            elif args.quant == 'nearest':
                quant_method[name].fasterquant()

            errors.append(quant_method[name].error)
            times.append(quant_method[name].time)
            Hmags.append(quant_method[name].Hmag)
            quant_method[name].free()


        layers[i] = layer.cpu()
        del layer
        del quant_method
        torch.cuda.empty_cache()

        # inps, outs = outs, inps

    model.config.use_cache = use_cache
    # print("errors")
    # print(errors)
    # print("Hmags")
    # print(Hmags)
    print(f'Total quant time: {sum(times):.2f}s')

    return quantizers, errors



if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset',
                        type=str,
                        choices=['wikitext2', 'ptb', 'c4'],
                        help='Where to extract calibration data from.')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed for sampling the calibration data.')
    parser.add_argument('--quant',
                        choices=['bitbal', 'parbal', 'allbal', 'ldlbal', 'nearest', 'gptq', 'gptq_updown'],
                        default='nearest',
                        help='Which quantization method to use.')
    parser.add_argument(
        '--wbits',
        type=int,
        default=16,
        choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument(
        '--npasses',
        type=int,
        default=1,
        help='number passes to repeat balance loop over 1-d.')
    parser.add_argument(
        '--groupsize',
        type=int,
        default=-1,
        help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--qfn',
                        type=str,
                        default='a',
                        help='qfn: a is default, b is sym incoherent based')
    parser.add_argument(
        '--unbiased',
        action='store_true',
        help='unbiased')
    # parser.add_argument('--load_H',
    #                     type=str,
    #                     default='',
    #                     help='Load quantized model.')

    args = parser.parse_args()

    toterr, totlay, tot_time = 0, 0, 0

    dict_proxy_layers = {
        "llama-7b": [32],
        "llama-2-7b-hf": [32],
        "llama-13b": [40],
        "llama-2-13b-hf": [40],
        "llama-30b": [60],
        "llama-65b": [80]
    }

    for argmodel in ["llama-2-7b-hf", "llama-2-13b-hf"]:
        for H_method in ["nearest", "gptq"]:
            print("H_method: {H_method}")
            # load_H = f"slurm/H_run2/{argmodel}_{H_method}_W4_preproc1"
            print("WARNING: need to specify load_H path")
            load_H = ""
            print(f"load_H: {load_H}")
            model = get_llama(f"meta-llama/{argmodel}")
            model.eval()
            tick = time.time()
            quantizers, errors llama_sequential_proxy(
                model, DEV, args, dict_proxy_layers[argmodel], load_H)
            tot_time += time.time() - tick
            print(f"Specific proxy (w^T H w) error: {sum(errors)}, len:{len(errors)}")
            toterr += sum(errors)
            totlay += len(errors)
            del model, quantizers, errors
    
    print("")
    print("------------------------------------------------")
    print(f'Total quant time elapsed: {tot_time:.2f}s')
    print(f'Proxy Summary: Qmethod:{args.quant}, Unbiased: {args.unbiased}, W:{args.wbits}, NPass:{args.npasses}')
    print(f"Avg proxy (w^T H w) error: {toterr / totlay} ({toterr} / {totlay})")
    print("------------------------------------------------")
    print("")
