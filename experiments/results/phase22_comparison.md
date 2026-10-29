# Phase 22 — AgentSemSteCom vs RACS-Base

Same evaluation cases, tokens, codec, SNRs and downstream code; only Xref differs.

| system | class | SNR | n | PSNR | SSIM | LPIPS | ref latency s | total s | ref VRAM GB |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| agentsemstecom | common | 5 | 2 | 18.30 | 0.422 | 0.459 | 4.201 | 15.2 | 2.84 |
| agentsemstecom | common | 10 | 2 | 20.72 | 0.537 | 0.297 | 4.201 | 14.4 | 2.84 |
| agentsemstecom | common | 15 | 2 | 21.53 | 0.577 | 0.247 | 4.201 | 14.2 | 2.84 |
| agentsemstecom | common | 20 | 2 | 21.45 | 0.577 | 0.238 | 4.201 | 14.1 | 2.84 |
| agentsemstecom | facial | 5 | 2 | 22.56 | 0.599 | 0.234 | 4.204 | 16.4 | 2.84 |
| agentsemstecom | facial | 10 | 2 | 24.53 | 0.687 | 0.158 | 4.204 | 16.3 | 2.84 |
| agentsemstecom | facial | 15 | 2 | 24.89 | 0.706 | 0.147 | 4.204 | 16.2 | 2.84 |
| agentsemstecom | facial | 20 | 2 | 25.10 | 0.715 | 0.140 | 4.204 | 16.3 | 2.84 |
| agentsemstecom | style | 5 | 2 | 19.93 | 0.543 | 0.345 | 3.343 | 11.9 | 2.16 |
| agentsemstecom | style | 10 | 2 | 22.68 | 0.653 | 0.219 | 3.343 | 11.9 | 2.16 |
| agentsemstecom | style | 15 | 2 | 23.28 | 0.683 | 0.195 | 3.343 | 11.9 | 2.16 |
| agentsemstecom | style | 20 | 2 | 23.43 | 0.692 | 0.178 | 3.343 | 11.9 | 2.16 |
| racs_base | common | 5 | 2 | 18.45 | 0.427 | 0.437 | 0.078 | 14.7 | 0.00 |
| racs_base | common | 10 | 2 | 20.86 | 0.544 | 0.285 | 0.078 | 14.2 | 0.00 |
| racs_base | common | 15 | 2 | 21.57 | 0.582 | 0.236 | 0.078 | 14.1 | 0.00 |
| racs_base | common | 20 | 2 | 21.55 | 0.584 | 0.229 | 0.078 | 14.2 | 0.00 |
| racs_base | facial | 5 | 2 | 23.19 | 0.623 | 0.217 | 0.037 | 14.7 | 0.00 |
| racs_base | facial | 10 | 2 | 25.07 | 0.708 | 0.143 | 0.037 | 14.8 | 0.00 |
| racs_base | facial | 15 | 2 | 25.53 | 0.727 | 0.129 | 0.037 | 14.8 | 0.00 |
| racs_base | facial | 20 | 2 | 25.60 | 0.730 | 0.132 | 0.037 | 14.7 | 0.00 |
| racs_base | style | 5 | 2 | 19.88 | 0.543 | 0.349 | 0.028 | 11.4 | 0.00 |
| racs_base | style | 10 | 2 | 22.83 | 0.663 | 0.210 | 0.028 | 11.4 | 0.00 |
| racs_base | style | 15 | 2 | 23.41 | 0.693 | 0.185 | 0.028 | 11.4 | 0.00 |
| racs_base | style | 20 | 2 | 23.45 | 0.699 | 0.173 | 0.028 | 11.4 | 0.00 |

## Security (10 dB, mean PSNR)

| system | token mode | PSNR |
| :-- | :-- | --: |
| agentsemstecom | correct | 22.64 |
| agentsemstecom | no_token | 14.09 |
| agentsemstecom | wrong | 13.97 |
| racs_base | correct | 22.92 |
| racs_base | no_token | 14.08 |
| racs_base | wrong | 13.98 |
| racs_base | wrong_with_wrong_ref | 13.98 |

## Conclusion (measured)

- Reference acquisition: 3.92 s (diffusion) vs 0.048 s (retrieval), 82x faster; retrieval needs no GPU memory for Xref (baseline peak 2.61 GB).
- Recovery PSNR: baseline 22.37 dB vs RACS 22.62 dB (paired mean delta 0.25 dB over 24 cases; RACS better).
- SSIM 0.616 vs 0.627; LPIPS 0.238 vs 0.227 (lower is better).
- Security gap (correct - wrong token, 10 dB): baseline 8.67 dB, RACS 8.94 dB.
- Subset run: 2 evaluation image(s) per class; treat as indicative, not final.

Paper anchors (approximate, not measured here): security gap 6.46 dB, XuNet 0.5091.
