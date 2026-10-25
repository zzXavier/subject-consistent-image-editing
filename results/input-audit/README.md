# Reference-photo audit

The first run selected each listing's first catalog reference. Reviewing the input photos showed that two depicted a different packaging unit or a person. The replacement images were selected for showing the exterior of the same product before examining the previous held-out scores. All three methods use the new reference for each affected subject.

| Subject | Split | Prior reference | Final reference | Reason |
| --- | --- | --- | --- | --- |
| B07FFHHG1Q lemonade | development | `1c/1c061a66.jpg` | `b7/b7a8d83d.jpg` | The first image shows an inner sachet. The second shows the original tub from another angle. |
| B0752GQ94S shampoo | test | `c7/c76e6501.jpg` | `9c/9c580902.jpg` | A person dominates the first image. The second shows the rear of the shampoo bottle. |

The original reference SHA-256 digests are `5a940b8e18e54087ac46224c6d0e1baf27851cd3d565a23ecd4e2e4cca621847` and `5aa02594052cbbdc7b7298b3d656abe2f4de0c2ed5884e2deb67cd249621c46b`, respectively. The final IDs and digests are in `configs/products.json`. The catalog files are included in `examples/abo/` under their original names. [ABO listing shard](https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_0.json.gz) · [image ID mapping](https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz).

The archived per-case values are in `old-scores.csv`. Both colors and all three methods were rerun for the two changed subjects. We checked that the processed source and reference pixels matched exactly before carrying forward the four unaffected cases. Gallery retrieval scores were recomputed over the complete revised gallery. A fresh full run executes all eight cases under the final input configuration.
