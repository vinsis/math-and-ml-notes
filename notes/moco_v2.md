> We verify the effectiveness of two of SimCLR’s design improvements by implementing them in the MoCo framework. With simple modifications to MoCo — namely,  `using  an  MLP  projection  head`  and  `more  data augmentation` — we establish stronger baselines that outperform SimCLR and do not require large training batches.

### MLP head
> Using the default `τ=0.07`,  pre-training with the MLP head improves from 60.6% to 62.9%; switching to the optimal value for MLP (0.2), the accuracy increases to 66.2%.

### Augmentation
>  The extra augmentation alone (i.e. no MLP) improves the MoCo baseline on ImageNet by 2.8% to 63.4%.