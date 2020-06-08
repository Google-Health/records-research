# Model Uncertainty for Electronic Health Records

This repository contains code for the following two papers:

* ["Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors"](https://arxiv.org/abs/2005.07186) (Dusenberry et al., 2020b)
* ["Analyzing the Role of Model Uncertainty for Electronic Health
  Records"](https://arxiv.org/abs/1906.03842) (Dusenberry et al., 2020)

The code currently contains the layers, models, and train/eval/predict script
used in the papers. NOTE: The code provided here is *not currently executable* due
to reliance on internal Google utilities. In particular, the data processing
pipeline is not included, but has been described in detail in [Rajkomar et al.,
2018](https://arxiv.org/abs/1801.07860).

This is not an officially supported Google product.

## References

> Michael W. Dusenberry\*, Ghassen Jerfel\*, Yeming Wen, Yian Ma, Jasper Snoek,
> Katherine Heller, Balaji Lakshminarayanan, Dustin Tran. [Efficient and
> Scalable Bayesian Neural Nets with Rank-1
> Factors](https://arxiv.org/abs/2005.07186). In Proc. of International
> Conference on Machine Learning (ICML) 2020.

> Michael W. Dusenberry, Dustin Tran, Edward Choi, Jonas Kemp, Jeremy Nixon,
> Ghassen Jerfel, Katherine Heller, and Andrew M. Dai. [Analyzing the Role of
> Model Uncertainty for Electronic Health
  Records](https://arxiv.org/abs/1906.03842) In Proc. of ACM Conference
> on Health, Inference, and Learning (ACM CHIL), 2020.

```none

@article{dusenberry2020rank1,
  author = {Dusenberry, Michael W. and Jerfel, Ghassen and Wen, Yeming and Ma, Yi-an and Snoek, Jasper and Heller, Katherine and Lakshminarayanan, Balaji and Tran, Dustin},
  title = {Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors},
  booktitle = {Proc. of the International Conference on Machine Learning (ICML)},
  year = {2020},
  url = {http://arxiv.org/abs/2005.07186},
}

@inproceedings{dusenberry2020analyzing,
  author = {Dusenberry, Michael W. and Tran, Dustin and Choi, Edward and Kemp, Jonas and Nixon, Jeremy and Jerfel, Ghassen and Heller, Katherine and Dai, Andrew M.},
  title = {Analyzing the role of model uncertainty for electronic health records},
  booktitle = {Proc. of the ACM Conference on Health, Inference, and Learning (ACM CHIL)},
  year = {2020},
  doi = {10.1145/3368555.3384457},
  url = {http://arxiv.org/abs/1906.03842},
}
```
