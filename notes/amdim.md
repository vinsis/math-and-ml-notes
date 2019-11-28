## Learning Representations by Maximizing Mutual Information Across Views

This is a build up on top of deep infomax:

> Our model, which we call Augmented Multiscale DIM (AMDIM), extends the local version of DeepInfoMax introduced by Hjelm et al. [2019] in several ways. First, we maximize mutual information between features extracted from independently-augmented copies of each image, rather than between features extracted from a single, unaugmented copy of each image. Second, we maximize mutual information __between multiple feature scales simultaneously, rather than between a single global and local scale__. Third, we use a more powerful encoder architecture. Finally, we introduce mixture-based representations. We now describe local DIM and the components added by our new model.

* Global features are replaced with _antecedent features_. Local features are replaced with _consequent features_. We want to predict consequent features conditioned on antedecent features.

> Intuitively, the task of the antecedent feature is to pick its true consequent out of a large bag of distractors.
