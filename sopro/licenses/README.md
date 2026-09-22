# Component licenses

| Component | License | Retained text or source |
| --- | --- | --- |
| Sopro model and synthesis package | Apache-2.0 | [Sopro license](Sopro-APACHE-2.0.txt), [source](https://github.com/samuel-vitorino/sopro) |
| SentencePiece reference behavior, version 0.2.2 | Apache-2.0 | [SentencePiece license](SentencePiece-APACHE-2.0.txt), [tagged source](https://github.com/google/sentencepiece/tree/v0.2.2) |
| LiteRT 2.2.0, AndroidX and Compose Material 1 | Apache-2.0 | [Apache text](Android-components-APACHE-2.0.txt); artifact coordinates and versions are pinned in `gradle/libs.versions.toml` |
| Kotlin and Gradle wrapper | Apache-2.0 | [Apache text](Android-components-APACHE-2.0.txt); existing source license headers remain intact |
| Default demo voice | CC0-1.0 | [Attribution and legal-code link](Demo-voice-CC0.md) |

The reference Python environment supports conversion and testing and is not embedded in the Android application.
Model and audio assets are fetched separately; their license declarations do not change the license of this source module.
