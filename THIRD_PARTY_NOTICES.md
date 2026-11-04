# Third-party notices

Original code and authored course material are MIT-licensed. External papers, websites, model weights, datasets and software retain their own licenses; linking them does not relicense them.

## Bundled components

- **KaTeX 0.19.1**: MIT. License and release-integrity metadata are included in `src/training_lab/web/vendor/katex/` and copied into the static preview's vendor folder. Upstream: https://github.com/KaTeX/KaTeX
- **Authored miniature text fixture**: included for inspecting decoder training mechanics, not a corpus for general language learning.

## Installed dependencies

PyTorch, scikit-learn, NumPy, TensorBoard, joblib, threadpoolctl and the optional Hugging Face/PEFT/Arrow packages are installed separately. Their distributions include their respective licenses. They are not vendored into this release. Consult each installed package before redistributing a binary build.

## Optional, separately downloaded assets

- **SmolLM2-360M-Instruct**, Apache-2.0: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct . Pinned revision is in `assets.py`. The repository excludes its weights and tokenizer assets.
- **Everyday Conversations**, Apache-2.0: https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k . Synthetic data; may overlap the model's original training. Excluded from the release.
- **Fashion-MNIST**: source and terms https://github.com/zalandoresearch/fashion-mnist . Downloaded only through explicit preparation; excluded from the release.
- **Wine Quality**: source, citation and license https://archive.ics.uci.edu/dataset/186/wine+quality . Downloaded only through explicit preparation; excluded from the release.
- **Iris, Wine and digits**: supplied by scikit-learn's dataset loaders; original source/citation details remain in the loader documentation and data catalog. They are not separately repackaged here.

Course source links and references remain adjacent to their explanations. Review upstream asset terms if you redistribute downloaded data or weights in your own release.
