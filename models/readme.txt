These models were trained on a compilation of the SemEval-2017 Task A dataset, and tweets scraped during OCtober 2020 that contained :) or :(.

These models perform well on that dataset, with vector features as high as 40,000.

These models did not generlize overly well to the #bcpoli dataset. My suspicion is that this is in part due to the lack of ASCII emotions in the new data, which I processed into "emopos" and "emoneg", before training.
