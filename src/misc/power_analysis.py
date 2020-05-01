# Power = 1 - Type II Error
# Pr(True Positive) = 1 - Pr(False Negative)

# estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower
# parameters for power analysis
effect = 0.8
alpha = 0.05 # sensitivity
power = 0.2 # specifity
ratio = 0.01
# perform power analysis
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=ratio, alpha=alpha)
print('Sample Size: %.3f' % result)