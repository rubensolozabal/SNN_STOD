# Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks (<a href="https://openreview.net/forum?id=udTDFAshNM&noteId=udTDFAshNM">ICLR 2026</a>)


**Desong Zhang, Jia Hu, Geyong Min**  
  University of Exeter  

### Quick Start with
``` 
python train.py -model 'vgg11' -dataset 'cifar10' -p 8 -gor_lambda 0.05
```
### Configuration

Hyperparameters and dataset settings can be configured in `./utils/config.py`

### Paper Abstract

Spiking Neural Networks (SNNs) have emerged as an efficient neuromorphic computing paradigm, offering low energy consumption and strong representational capacity through binary spike-based information processing. However, their performance is heavily shaped by the input encoding method. While direct encoding has gained traction for its efficiency and accuracy, it proves less robust than traditional rate encoding. To illuminate this issue, we introduce Gradient Temporal Collinearity (GTC), a principled measure that quantifies the directional alignment of gradient components across time steps, and we show—both empirically and theoretically—that elevated GTC in direct encoding undermines robustness. Guided by this insight, we propose Structured Temporal Orthogonal Decorrelation (STOD), which integrates parametric orthogonal kernels with structured constraints into the input layer of direct encoding to diversify temporal features and effectively reduce GTC. Extensive experiments on visual classification benchmarks, show that STOD consistently outperforms state-of-the-art methods in robustness, highlighting its potential to drive SNNs toward safer and more reliable deployment.

### Acknowledgements

This implementation is based on the <a href="https://github.com/fangwei123456/spikingjelly">SpikingJelly</a> framework. We sincerely thank the authors for making their code publicly available. 
