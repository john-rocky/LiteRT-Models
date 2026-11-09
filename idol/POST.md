Turn one photo into a 3D object you can spin around, sides and back included, on a phone.

Idol (speridlabs) now runs on the phone with LiteRT: the image encoders, the 5-step turbo denoiser and the Gaussian
decoder on the GPU, the author's background remover on the CPU. One photo becomes 262,144 Gaussians in {S26_TIME} on a
Galaxy S26. Checked against the author's PyTorch pipeline: renders from four sides agree at {PSNR_RANGE} dB.

Graphs, a Python runner and an Android sample: https://huggingface.co/litert-community/Idol
