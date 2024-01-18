This experiment so far is kind of a dud SGD converges the fastest with the least amount of memory while Adafactor has the worst convergence properties and frequently gets stuck in local minima and that isn't affected by any hyperparam tuning that I'm doing


The best recipe is this which beats ADAM `optimizer = Adafactor(model_copy.parameters(),  scale_parameter=True, relative_step=True, warmup_init=False)`

* Adafactor does not need you to set a learning rate it can do so itself using scale_parameters and relative step as a function of the number of steps. Warmup init also but that just seems to make the results worse for me
* Epislon is only there for numerical reasons not important
* clip threshold seemed to have no impact on my toy model nooping this would be setting it to `sys.maxint`
* for decay rate and weight decay setting them to values close to 0 will prevent underfitting
