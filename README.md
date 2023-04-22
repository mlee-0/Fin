# Fin
2D transient thermal response predictions in fins using convolutional neural networks.

Variable parameters:
* Fin thickness
* Fin taper ratio
* Temperature boundary condition
* Convection coefficient

Input images (3×40×80):
* Channel 1: Fin shape
* Channel 2: Convection coefficient
* Channel 3: Temperature boundary condition

Output images (10×8×32 or 1×8×32):
* 2D temperature distribution over 10 instances in time (10 channels), or
* 2D thermal gradient distribution over 10 instances in time (10 channels), or
* 2D thermal stress distribution at final time only (1 channel)