Turn one photo into a 3D object you can spin around, sides and back included, entirely on your phone.

Idol (speridlabs) now runs on the phone with LiteRT: the image encoders, the 5-step turbo denoiser and the Gaussian decoder on the GPU, the author's background remover on the CPU. One photo becomes 262,144 Gaussians in 303 s on a Galaxy S26 (the first photo after start, on a cool phone). Checked against the author's PyTorch pipeline: renders from four sides agree at a PSNR of 41.4–46.6 dB.

Graphs, a Python runner and an Android sample: https://huggingface.co/litert-community/Idol

---

Clip (attach): ~/Downloads/showcase-video/idol/idol_s26.mp4
