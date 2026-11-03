# Subject-consistent image editing

Use multiple views of real Amazon Berkeley Objects products to compare text-only, IP-Adapter reference conditioning, and spatially masked reference conditioning. Install CUDA PyTorch 2.4.1 and requirements.txt in a Python 3.11 environment. Run `python run.py demo` for a single object, `python run.py reproduce` for both colors on four subjects, and `python run.py evaluate` to rebuild scores.

`src/data.py` reads the pinned publicly available photographs and extracts a white-background foreground. `src/experiment.py` uses a pretrained SD1.5 inpainting pipeline and the official SD1.5 IP-Adapter. The text control sets its scale to zero. The other two modes keep the same reference strength, target, seed and steps. The regional treatment restricts reference attention to the foreground. Two products are held out for evaluation. Reported dimensions are edit-mask-exterior RGB error, CLIP vision reference cosine and requested background hue coverage. The cosine uses the conditioning encoder and must be interpreted with the independent pixel and hue checks; it is not independent human identity recognition. A catalog image of two bottles is excluded because the mask extractor assumes one product.

The contact sheet in examples/source-images.jpg shows actual catalog inputs. Experiment completion and dependencies are recorded in HANDOFF.md.

