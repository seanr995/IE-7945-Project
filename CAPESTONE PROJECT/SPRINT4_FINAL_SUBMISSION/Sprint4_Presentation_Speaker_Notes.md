# Sprint 4 Presentation - Speaker Notes

All figures are real values from `sprint4_results/` (see `Sprint4_Presentation_Key_Metrics.md`). Human review is complete (80/80, values from human_review_template.csv); it is a team review, not ground truth.

## Slide 1 - Sprint 4: From 80 Leaf Topics to a Hierarchical Taxonomy

Open with the one-line story. We did not re-cluster anything: the 80 leaf topics from Sprint 3 stay exactly as they were. Sprint 4 adds names, two parent levels, and an outside check against O*NET. The final structure is 8 work domains over 20 activity families over 80 leaves. Point out the colour code: teal is deterministic computation, purple is Gemini, amber is O*NET, red is human review, which is complete (80 of 80 reviewed).

## Slide 2 - What Entered Sprint 4

Stress that Sprint 4 starts from Sprint 3's final model: Sentence-T5 embeddings clustered with Leiden at 80 topics. We froze it so that anything we change later can be attributed to naming or hierarchy, not to re-clustering. We verified byte-for-byte that all 3,015 task-to-leaf assignments are identical in the Sprint 4 output. Say the key sentence clearly: O*NET did not create our topics. It is an outside reference used only to evaluate.

## Slide 3 - Naming the 80 Leaf Topics

Walk the pipeline left to right. For each leaf we computed the centroid, picked the 10 core tasks closest to it, the 5 boundary tasks furthest from it, and c-TF-IDF keywords. Gemini then got a compact slice of that evidence (7 core, 3 boundary, 8 keywords) and returned a structured label. Python validated every record against the schema and word limits; Gemini never wrote a file. The three examples are real rows from the results. One topic, LT_048, was flagged by Gemini as mixed, which is why review matters.

## Slide 4 - Building the Hierarchy

Level-3 leaves roll up into Level-2 activity families and Level-1 work domains, both cut from one dendrogram so they are nested. We tested average linkage and Ward linkage on the 80 leaf centroids, 9 candidates each, with identical gates and the same score. No average-linkage candidate passed the gates: at 20 and 8, one domain swallowed 87 percent of all tasks. All nine Ward candidates passed; Ward 20/8 had the highest score of 0.903. So the selection was evidence-based, not hand-picked. Then one Gemini critic round reviewed the structure and moved a single leaf. Use the small real example on the right: Strategic Planning and Financial Management with its two families.

## Slide 5 - O*NET Evaluation

Explain the order of operations. For every leaf centroid we computed cosine similarity to all 18,838 O*NET task statements and took the exact top 10. That is plain linear algebra. Only after that did Gemini see the top candidates and judge the semantic match. Gemini did not select the nearest O*NET tasks. Metrics: coverage is 0.975 because 78 of 80 topics matched meaningfully. Purity, 0.1935, is concept agreement, not accuracy: it is the share of tasks in a topic whose nearest O*NET task falls in the dominant O*NET concept of that topic. It is low in absolute terms because members of one topic map to many different O*NET activities. Top-10 nearest-neighbour agreement is 0.6225. Gemini judged 58 strong, 20 partial and 2 no-match topics.

## Slide 6 - Beyond Basic Sprint 4: Robustness and Coverage Gaps

A: We perturbed the hierarchy five times, with seeds 101 to 505, each time keeping a random 80 percent of the tasks inside every leaf, recomputing centroids and re-clustering. Persistence is how often a leaf kept its parent. Mean persistence is 0.715 at Level 1 and 0.7925 at Level 2, ARI against the final hierarchy 0.55 and 0.65. That is moderate, not perfect, and we report it honestly. B: The gap analysis uses quantile cut-offs from our own data. Possible emerging work means job-posting evidence has limited alignment with O*NET; it does not prove O*NET is outdated. C: 80 evidence cards, one per topic, are what reviewers will read.

## Slide 7 - Human Validation

Human review is complete: all 80 topics were judged on five questions and the CSV has no blanks. Labels: 77 accepted, 3 need a rename (LT_015, LT_071, LT_076); suggested names are reported, not applied. Representative tasks: 74 topics coherent, 6 mixed. Boundary tasks: 28 topics have boundary tasks that do not fit, which is the weakest area. Over-merged: 5 topics (LT_015, LT_039, LT_071, LT_076, LT_078); these are grab-bags glued by a word like lead or drive. Under-merged: 12 topics (LT_004, LT_009, LT_010, LT_021, LT_026, LT_027, LT_032, LT_034, LT_042, LT_052, LT_055, LT_061); they form near-duplicate pairs such as LT_004 and LT_061, LT_026 and LT_042, LT_021 and LT_052. We did not change any memberships or names: merges and renames are future work. This is a review by our team, not ground truth.

## Slide 8 - Sprint 4 Outcome / Sprint 5 Handoff

Summarise the deliverables: 80 labelled leaves, 20 families, 8 domains, 3,015 frozen task memberships, an O*NET evaluation, a stability test and 80 evidence cards. Gemini made 10 real calls with no repairs or retries, and Sprint 4 made no Claude calls. Close with the chain at the bottom and the caveat: this is an interpretable, human-auditable structure, not ground truth. Human review is complete: it accepted 77 of 80 labels and 74 of 80 topics as coherent, but found 5 over-merged and 12 under-merged topics. Sprint 5 should apply the 3 renames and decide on merging the near-duplicate pairs.

