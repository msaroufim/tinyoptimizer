This experiment so far is kind of a dud SGD converges the fastest with the least amount of memory while Adafactor has the worst convergence properties and frequently gets stuck in local minima and that isn't affected by any hyperparam tuning that I'm doing


## Vanilla PyTorch implementation on synthetic data

The best recipe is this which beats ADAM `optimizer = Adafactor(model_copy.parameters(),  scale_parameter=True, relative_step=True, warmup_init=False)`

* Adafactor does not need you to set a learning rate it can do so itself using scale_parameters and relative step as a function of the number of steps. Warmup init also but that just seems to make the results worse for me
* Epislon is only there for numerical reasons not important
* clip threshold seemed to have no impact on my toy model nooping this would be setting it to `sys.maxint`
* for decay rate and weight decay setting them to values close to 0 will prevent underfitting

## Built in hf implementation on wikitext

In this case it seems like the HF implementation converges very similarly. For this experiment we use 10% of wikitext 

## Memory profiles

Adafactor <img width="1727" alt="Screenshot 2024-01-18 at 10 26 41 AM" src="https://github.com/msaroufim/tinyoptimizer/assets/3282513/b7ba6f64-549e-4205-8ed0-c5a6b6a58c81">

Adam <img width="1720" alt="Screenshot 2024-01-18 at 10 27 54 AM" src="https://github.com/msaroufim/tinyoptimizer/assets/3282513/f6882516-215f-4ec6-888c-4c72ef4fe0c6">
