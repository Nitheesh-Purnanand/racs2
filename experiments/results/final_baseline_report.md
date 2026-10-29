# RACS2 Final Baseline Report

- Generated: 2026-10-29T07:19:21Z
- Status: **fail**
- System: agentsemstecom
- Role: evaluation
- Approximate: False

## Baseline recovery quality

| class | SNR dB | n | MSE | PSNR | SSIM | LPIPS |
| :-- | --: | --: | --: | --: | --: | --: |
| common | 5 | 22 | 0.049567 | 13.1420 | 0.2463 | 0.6829 |
| common | 10 | 22 | 0.028549 | 15.6511 | 0.3517 | 0.5742 |
| common | 15 | 22 | 0.018251 | 17.7753 | 0.4565 | 0.4476 |
| common | 20 | 22 | 0.013714 | 19.1920 | 0.5302 | 0.3527 |
| facial | 5 | 22 | 0.038392 | 14.2310 | 0.2803 | 0.6493 |
| facial | 10 | 22 | 0.019245 | 17.3492 | 0.4202 | 0.5323 |
| facial | 15 | 22 | 0.010630 | 20.0407 | 0.5495 | 0.3968 |
| facial | 20 | 22 | 0.006610 | 22.2311 | 0.6449 | 0.2740 |
| style | 5 | 22 | 0.044858 | 13.6004 | 0.2539 | 0.6377 |
| style | 10 | 22 | 0.023435 | 16.5308 | 0.3736 | 0.4960 |
| style | 15 | 22 | 0.015492 | 18.4638 | 0.4593 | 0.3752 |
| style | 20 | 22 | 0.011815 | 19.7778 | 0.5178 | 0.2887 |

## Token-condition security

| condition | PSNR dB |
| :-- | --: |
| correct | 17.3321 |
| wrong | 13.0144 |
| no_token | 13.0361 |

- correct minus wrong: 4.3177 dB
- correct minus no_token: 4.2960 dB

## XuNet steganalysis

- Test accuracy: 0.4625 (n=80)
- Chance rate: 0.5000
- Paper anchor: {'accuracy': 0.5091, 'label': 'approximate'}
- Checkpoint: D:\racs2\experiments\steganalysis\xunet_baseline_0000002a.pt

## Verification gates

- Phase 15: pass=True gate_a=False gate_b=True summary={'passed': 12, 'total': 12, 'gate_a_misses': 8}
- Phase 16: pass=False summary={'passed': 16, 'total': 17}

Gate A compares calibration cells to the paper's Table II anchors with tolerances. Gate B is the authoritative structural gate. Every Gate A miss is recorded in docs/DEVIATIONS.md.

## Provenance

- split_version: 397637a71130e0d21ca52bfd6e6798efdbb868248739c4ee3f57c71315ba4202
- tokens_version: 201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1
- config_hash: da93514ef6f31a949263214acd9c9f8c024c1af7ccf670ac15cf1e307ebe5874
- code_commit: cd45d0a756fcdcc76f548d314f087ee611600357
- baseline_created_utc: 2026-10-28T21:32:55Z
- models: {'reference': 'stable-diffusion-v1-5/stable-diffusion-v1-5', 'controlnet': 'lllyasviel/control_v11p_sd15_seg', 'edict': {'scheduler': 'ddim', 'steps': 20, 'mixing': 0.93, 'image_size': 256, 'latent_size': 32, 'guidance_scale': 1.0, 'dtype': 'float16', 'latent_dtype': 'float32', 'sweep_steps': [10, 20, 50], 'sweep_mixing': [0.5, 0.7, 0.93, 0.99], 'num_random_latents': 20, 'num_image_latents': 20, 'results_path': 'experiments/results/phase10_edict.json', 'grid_path': 'experiments/results/phase10_edict_sweep.png', 'max_rel_error': 0.005}, 'codec': 'sd_vae_substitute', 'enhancement': 'disabled'}
- determinism: {'sample': 6, 'reference_equal': True, 'stego_equal': True, 'recovered_equal': True}
- baseline_table_sha256: 2d5cc07ad006650d3ac7bbd70a264536c74a571ada32c8a93f3a24ae49b54d92

## Artifacts

- experiments/results/baseline_table.json
- experiments/results/baseline_manifest.json
- experiments/results/baseline_security.json
- experiments/results/baseline_steganalysis.json
- experiments/results/final_baseline_report.json
- experiments/results/final_baseline_table.csv
