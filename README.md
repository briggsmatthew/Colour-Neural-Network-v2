# Colour Neural Network Version 2

A basic neural network made in Python with numpy. It attempts to identify a colour
from an RGB value using the `interact` function. 

The file `colour_neural_net_v2.py` provides the `ColourNetwork` class which, 
after training (ie. using the `train_from_file` function) allows the user to 
`interact` with the network. In doing so, you provide the ColourNetwork with an
RGB list (ie. `[255, 0, 0]` will be pure red) and it attempts to guess what the
colour is named.

Below is an example of usage:
```python
>>> a = ColourNetwork()
>>> a.train_from_file("training1.txt")
>>> a.interact([33, 33, 200]) # A nice blue
Red: 0.0%
Orange: 0.0%
Yellow: 0.0%
Green: 0.0%
Blue: 100.0%
Indigo: 0.0%
Violet: 0.0%
Black: 0.0%
White: 0.0%
Best idea: Blue
```