# MP3 Reflection

## Grounding & abstention
Where did your assistant abstain correctly? Did it ever answer something it
shouldn't have (or refuse something it could have answered)? What threshold did
you settle on, and why?

*The assistant correctly abstained on all 3 should-abstain questions: abstention accuracy was 1.0, with a hallucination rate of 0.0. However, it also abstained on 2 answerable questions. The file’s default threshold was 0.3 but changing its value didn't impact the output as score was already greter than threshold. When I increased the number of chunks in answer fn then it was able to answer one of the abstained answerable question but answer was still abstained for first question.*


## Citations
Did the model ever cite a chunk that didn't support the claim, or invent a marker?
What did `validate_citations` catch?

*The 5 answerable questions the assistant answered were grounded and cited; 1 answerable question was abstained on with k=5. The table doesn’t show the citation markers themselves, so it can’t establish whether the model invented one. validate_citations flags markers whose numbers are outside the available source range; it does not verify that a cited excerpt semantically supports a claim.*

## Reading your eval table
What do your four numbers say? Which is weakest, and what would you change?

*Retrieval hit rate: 1.0 — the expected source was retrieved for all 6 answerable questions.
Grounded & cited rate: 0.833 — 5 of 6 answerable questions received a grounded, cited response.
Abstention accuracy: 1.0 — all 3 should-abstain questions were refused.
Hallucination rate: 0.0 — none of those 3 questions received an answer.
The weakest result is the grounded & cited rate. The one answerable question that was refused account for the gap despite perfect retrieval. I’d inspect their scores and context first, then tune the threshold only if the retrieval results support doing so.*

## One surprise
Something from the build that genuinely surprised you.
```
*Retrieval found the expected source for every answerable question, yet the assistant still abstained on one of them. A retrieval hit alone doesn’t guarantee the answer stage will respond.*


Deployed to https://github.com/himanshudwi31/mp1/tree/main/Week12_GradedMiniProject