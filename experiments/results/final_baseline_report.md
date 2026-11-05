# RACS2 Final Baseline Report

- Generated: 2026-11-05T09:46:36Z
- Status: **fail**
- System: agentsemstecom
- Role: evaluation
- Approximate: False

## Baseline recovery quality

| class | SNR dB | n | MSE | PSNR | SSIM | LPIPS |
| :-- | --: | --: | --: | --: | --: | --: |
| common | 5 | 2 | 0.015212 | 18.3035 | 0.4216 | 0.4586 |
| common | 10 | 2 | 0.008689 | 20.7232 | 0.5365 | 0.2971 |
| common | 15 | 2 | 0.007283 | 21.5328 | 0.5775 | 0.2465 |
| common | 20 | 2 | 0.007394 | 21.4470 | 0.5771 | 0.2384 |
| facial | 5 | 2 | 0.005605 | 22.5574 | 0.5991 | 0.2335 |
| facial | 10 | 2 | 0.003524 | 24.5347 | 0.6866 | 0.1580 |
| facial | 15 | 2 | 0.003246 | 24.8867 | 0.7058 | 0.1469 |
| facial | 20 | 2 | 0.003096 | 25.0972 | 0.7150 | 0.1403 |
| style | 5 | 2 | 0.010398 | 19.9293 | 0.5430 | 0.3448 |
| style | 10 | 2 | 0.005494 | 22.6752 | 0.6530 | 0.2192 |
| style | 15 | 2 | 0.004760 | 23.2770 | 0.6833 | 0.1950 |
| style | 20 | 2 | 0.004590 | 23.4343 | 0.6918 | 0.1784 |

## Token-condition security

| condition | PSNR dB |
| :-- | --: |
| correct | 22.3665 |
| wrong | 13.9598 |
| no_token | 14.0246 |

- correct minus wrong: 8.4067 dB
- correct minus no_token: 8.3419 dB

## XuNet steganalysis

- Test accuracy: 0.5000 (n=8)
- Chance rate: 0.5000
- Paper anchor: {'accuracy': 0.5091, 'label': 'approximate'}
- Checkpoint: D:\racs2\experiments\steganalysis\xunet_baseline_0000002a.pt

## Verification gates

- Phase 15: pass=False gate_a=False gate_b=False summary={'passed': 11, 'total': 12, 'gate_a_misses': 8}
- Phase 16: pass=False summary={'passed': 12, 'total': 17}

Gate A compares calibration cells to the paper's Table II anchors with tolerances. Gate B is the authoritative structural gate. Every Gate A miss is recorded in docs/DEVIATIONS.md.

## Provenance

- split_version: 397637a71130e0d21ca52bfd6e6798efdbb868248739c4ee3f57c71315ba4202
- tokens_version: 201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1
- config_hash: f4c93891b8fab426c72d5cd53a2bac6cfaec838a7743b037bcf55507b14fd935
- code_commit: cd45d0a756fcdcc76f548d314f087ee611600357
- baseline_created_utc: 2026-10-29T10:20:06Z
- models: {'reference_backend': 'picx_real', 'reference': 'models/picx-real', 'controlnet': 'lllyasviel/control_v11p_sd15_seg', 'facial_controlnet': 'lllyasviel/control_v11p_sd15_openpose', 'kpub': 'local_llm', 'xfeat': {'common': 'oneformer', 'facial': 'openpose', 'style': 'none'}, 'edict': {'scheduler': 'ddim', 'steps': 20, 'mixing': 0.93, 'image_size': 256, 'latent_size': 32, 'guidance_scale': 1.0, 'dtype': 'float16', 'latent_dtype': 'float32', 'sweep_steps': [10, 20, 50], 'sweep_mixing': [0.55, 0.7, 0.93, 0.99], 'num_random_latents': 20, 'num_image_latents': 20, 'results_path': 'experiments/results/phase10_edict.json', 'grid_path': 'experiments/results/phase10_edict_sweep.png', 'max_rel_error': 0.005}, 'ipadapter': {'enabled': True, 'weight_name': 'ip-adapter-plus_sd15.safetensors', 'scale': 1.0, 'weight_sha256': 'a1c250be40455cc61a43da1201ec3f1edaea71214865fb47f57927e06cbe4996'}, 'codec': 'swinjscc', 'codec_sha256': '7a45d4fa6a5b4210909e4d529b5b82ce3f1270cb0021adafe8c5f7318d164030', 'enhancement': 'enabled', 'enhancement_backend': 'codeformer'}
- determinism: {'sample': 2, 'reference_equal': True, 'stego_equal': True, 'recovered_equal': True}
- baseline_table_sha256: 987032a003bd599489fbd466247902e719dfd4ccf9e6323b26a99ef75e401bac

## Artifacts

- experiments/results/baseline_table.json
- experiments/results/baseline_manifest.json
- experiments/results/baseline_security.json
- experiments/results/baseline_steganalysis.json
- experiments/results/final_baseline_report.json
- experiments/results/final_baseline_table.csv
