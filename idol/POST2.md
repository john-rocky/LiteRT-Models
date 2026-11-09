Turn one photo into a 3D object you can spin around, sides and back included, entirely on your phone.

On a Galaxy S26, Idol (speridlabs) on LiteRT now makes a splat of 65,536 Gaussians in 158 s with its new default form, or of 262,144 Gaussians in 267 s with the first release's form, which stays in the repo (the first photo after start, on a cool phone). Checked against the author's PyTorch pipeline, renders from four sides agree at a PSNR of 32.3–34.4 dB for the default and 41.4–46.6 dB for the larger form.

Graphs, a Python runner and an Android sample: https://huggingface.co/litert-community/Idol

---

Clip (attach, unchanged from the day-0 post): ~/Downloads/showcase-video/idol/idol_s26.mp4 (sha256 df5c5a23…)

---

JA:

写真 1 枚から、側面も裏側も回して見られる 3D の物体を、スマホの中だけで作れます。

Galaxy S26 で、LiteRT 版の Idol(speridlabs)は新しい既定の形なら 65,536 個の Gaussian の splat を 158 秒で、repo に残した最初の公開の形なら 262,144 個を 267 秒で作ります(起動後の最初の 1 枚、冷えた端末)。作者の PyTorch の pipeline と比べると、4 方向から描いた画像の一致は PSNR で既定の形が 32.3–34.4 dB、大きい形が 41.4–46.6 dB です。

graph、Python の runner、Android の sample: https://huggingface.co/litert-community/Idol
