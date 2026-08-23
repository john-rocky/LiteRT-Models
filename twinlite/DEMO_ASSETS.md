# Demo asset provenance (TwinLiteNet)

The demo clip is **Pexels License** — free for commercial and non-commercial use,
**no attribution required**, modification allowed (our segmentation overlay is a
derivative). The file itself is not committed; `**/assets/*.mp4` is gitignored.

| file | source | author |
|---|---|---|
| `app/src/main/assets/demo_road.mp4` | https://www.pexels.com/video/cars-traveling-on-expressway-5382495/ | K |

Verified live via the Pexels API on 2026-08-23. Constraints we stay inside: no selling
of unaltered copies, no implication that anyone depicted endorses anything, neutral
framing. Both build flavors read the same file, so the GPU and NPU runs see identical
frames and the comparison does not depend on where a camera was pointed.

To fetch it again:

```bash
curl -s -H "Authorization: $PEXELS_API_KEY" \
  "https://api.pexels.com/videos/videos/5382495" |
  python3 -c "import json,sys; d=json.load(sys.stdin); \
    print([f['link'] for f in d['video_files'] if f.get('height')==720][0])" |
  xargs curl -sL -o twinlite/app/src/main/assets/demo_road.mp4
```
