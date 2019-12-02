### Evaluating the Unsupervised Learning of Disentangled Representations

Disentangled representations are representations where `models capture the independent features of a given scene in such a way that if one feature changes, the others remain unaffected.`

### Key points
* They present a theorem which states that
> Unsupervised learning of disentangled representations is impossible without inductive biases on both the data set and the models (i.e., one has to make assumptions about the data set and incorporate those assumptions into the model)

* For the considered models and data sets, we cannot validate the assumption that disentanglement is useful for downstream tasks, e.g., that with disentangled representations it is possible to learn with fewer labeled observations.

They also released a [library](https://github.com/google-research/disentanglement_lib) and released [lots of pretrained models](https://github.com/google-research/disentanglement_lib#pretrained-disentanglement_lib-modules).
