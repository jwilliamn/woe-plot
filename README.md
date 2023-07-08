# The Weight of Evidence (WOE) plot 

This function is an alternative to the `woebin_plot` function from [scorecardpy](https://github.com/ShichenXie/scorecardpy/tree/master) which does not have the option to plot on matplotlib axes.

In this function I added such functionality that plots directly to matplotlib axes, which is important(at least for me) to generate neatly reports for all features desired that are in your data.


The axes can be passed as:
```python
woebin_plot2(bins, ax=axes[1, 1])
```
