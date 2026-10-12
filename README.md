# IWSLT 2025 JU-CSE-NLP Cascaded Speech Translation

Code release for *JU-CSE-NLP’s Cascaded Speech to Text Translation Systems for IWSLT 2025 in Indic Track*.

| Folder | Direction |
|---|---|
| [`Indic-to-English/`](Indic-to-English/) | IndicConformer ASR → IndicTrans2 MT (Bengali / Hindi / Tamil → English) |
| [`English-to-Indic/`](English-to-Indic/) | Whisper ASR → NLLB MT (English ) → (Bengali / Hindi / Tamil) |

Use each subdirectory as its own package root for install, training, and inference.


# Citation

Dhar, Lahiri, Mondal, and Bandyopadhyay. *JU-CSE-NLP’s Cascaded Speech to Text Translation Systems for IWSLT 2025 in Indic Track.* In *Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)*, pages 201–205. ACL.


## BibTeX

```bibtex
@inproceedings{dhar-etal-2025-ju,
  title = {{JU-CSE-NLP}'s Cascaded Speech to Text Translation Systems for {IWSLT} 2025 in Indic Track},
  author = {Dhar, Debjit and Lahiri, Soham and Mondal, Tapabrata and Bandyopadhyay, Sivaji},
  booktitle = {Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)},
  pages = {201--205},
  year = {2025},
  publisher = {Association for Computational Linguistics}
}
```