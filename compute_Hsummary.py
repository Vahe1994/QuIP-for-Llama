import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle

def Hsummary(H, percdamp=0.01):
    assert H.shape[0] == H.shape[1]
    n = H.shape[0]
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(n)
    H[diag, diag] += damp
    L = torch.linalg.cholesky(H)
    D = torch.diag(L).square()
    a = D.sum() / H.trace()
    k00 = torch.linalg.matrix_rank(H) / n
    k01 = torch.linalg.matrix_rank(H, rtol=0.01) / n
    _, Q = torch.linalg.eigh(H)
    mu = torch.linalg.matrix_norm(Q) * np.sqrt(n)
    return a, k00, k01, mu

def collect(dirname, savename):
    a_ls, k00_ls, k01_ls, mu_ls = [], [], [], []
    for fname in tqdm(glob.glob(dirname+'/*.pt')):
        H = torch.load(fname)
        print(f"{fname}, H.shape: {H.shape}")
        a, k00, k01, mu = Hsummary(H)
        a_ls.append(a)
        k00_ls.append(k00)
        k01_ls.append(k01)
        mu_ls.append(mu)
    a_ls = np.array(a_ls)
    k00_ls = np.array(k00_ls)
    k01_ls = np.array(k01_ls)
    mu_ls = np.array(mu_ls)
    print(f"tr(D) / tr(H): {np.mean(a_ls)} (+/- {np.std(a_ls)})")
    print(f"matrix rank rtol=0.00: {np.mean(k00_ls)} (+/- {np.std(k00_ls)})")
    print(f"matrix rank rtol=0.01: {np.mean(k01_ls)} (+/- {np.std(k01_ls)})")
    print(f"incoherency mu: {np.mean(mu_ls)} (+/- {np.std(mu_ls)})")
    with open(savename, 'wb') as f:
        pickle.dump({
            'trDtrH': a_ls,
            'rank_rtol0': k00_ls,
            'rank_rtol01': k01_ls,
            'incoh_mu': mu_ls
        }, f)

p1 = [
    "slurm/H_run2/llama-2-7b-hf_gptq_W4_preproc1",
    "slurm/H_run2/llama-2-13b-hf_gptq_W4_preproc1",
    "slurm/H_run2/llama-7b_gptq_W4_preproc1",
    "slurm/H_run2/llama-13b_gptq_W4_preproc1",
    "slurm/H_run2/llama-30b_gptq_W4_preproc1",
    "slurm/H_run2/llama-65b_gptq_W4_preproc1",
]
p2 = [
    "slurm/H_run2/llama-2-7b-hf_gptq_W4_preproc2",
    "slurm/H_run2/llama-2-13b-hf_gptq_W4_preproc2",
    "slurm/H_run2/llama-7b_gptq_W4_preproc2",
    "slurm/H_run2/llama-13b_gptq_W4_preproc2",
    "slurm/H_run2/llama-30b_gptq_W4_preproc2",
    "slurm/H_run2/llama-65b_gptq_W4_preproc2",
]

def save_spectrum(fname, savename):
    """ slurm/Hspectrum/...
    """
    H = torch.load(fname)
    n = H.shape[0]
    percdamp = 0.01
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(n)
    H[diag, diag] += damp 
    L = torch.linalg.eigvalsh(H).numpy()
    L = pd.DataFrame(L)
    L.to_csv(savename)

def kick_spectrum():
    save_spectrum(
        "slurm/H_run2/llama-7b_gptq_W4_preproc1/H_model.layers.16.self_attn.k_proj.pt",
        "slurm/Hspectrum/llama-7b_16kproj_preproc1.csv"
        )
    save_spectrum(
        "slurm/H_run2/llama-7b_gptq_W4_preproc1/H_model.layers.30.fc1.pt",
        "slurm/Hspectrum/llama-7b_30post_attention_layernorm_preproc1.csv"
        )
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname',
                        type=str)
    parser.add_argument('--savename',
                        type=str)
    args = parser.parse_args()

    collect(args.dirname, args.savename)