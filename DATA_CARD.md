# Data card: one real benchmark, one original fixture

## MovieLens 100K: downloaded locally, not redistributed

The real experiment uses ratings collected by GroupLens Research at the University of Minnesota: 100,000 ratings on a 1-5 scale from 943 users across 1,682 movies. The source observations span September 1997 to April 1998. One row is one user-item rating with a Unix-second timestamp. This experiment reads only `u.data`; it does not load age, gender, occupation, ZIP codes, movie titles or external URLs.

Source and terms, checked November 4, 2026:

- [Official dataset page](https://grouplens.org/datasets/movielens/100k/)
- [Original usage license and schema](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
- Archive: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
- SHA-256 observed for that archive: `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`

The original terms permit research use with attribution, prohibit implying GroupLens endorsement, prohibit redistribution without separate permission, and require separate permission for commercial or revenue-bearing use. This repository is an independent research/education portfolio. The source-code license does not relicense the dataset. `data/raw/` is ignored; the downloader extracts only `u.data` from the validated archive. Do not commit the archive or raw ratings. The published per-user artifact contains only neutral row numbers and derived evaluation measures, not source user identifiers or rating rows.

Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. [doi:10.1145/2827872](https://doi.org/10.1145/2827872).

## What the dataset does not tell us

- Rating timestamps are observation times; they are not a complete exposure or impression log.
- Unrated movies are not verified dislikes. BPR negative sampling is a modeling assumption.
- This old, filtered, self-selected population is not representative of a modern product's users.
- Test-only users/items cannot receive factors learned only from training. Coverage is reported explicitly.
- Offline ranking measures do not establish online CTR, retention, revenue, or causal lift.

## Original synthetic fixture

`data/fixture.tsv` is an original, deterministic set of invented user-item ratings for exercising loading and splitting without a network connection. It is not sampled, transformed, or copied from MovieLens. It is covered by this project's source-code license. Fixture results are not the published research benchmark.

Recreate it with:

```python
from pathlib import Path
rows = [(u, i, 5 if (u + i) % 3 else 2, (i * 7 + u * 13) % 97 + 1)
        for u in range(1, 7) for i in range(1, 13)]
Path("data/fixture.tsv").write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
```
