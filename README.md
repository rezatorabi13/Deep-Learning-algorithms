
## Deep learning algorithms
In this toturial, I introduce some deep learning algorithms and provive codes for building and training neural networks from the scratch based on those algorithms.

### 1- Backpropagation algorithm
Backpropagation algorithm is probably the most fundamental building block in a neural network. The algorithm is used to effectively train a neural network through a method called chain rule. In simple terms, after each forward pass through a network, backpropagation performs a backward pass while adjusting the model’s parameters (weights and biases). In this tutorial we develope a backpropagation algorithms from the first scratch. To see a backpropagation algorithm and how it works, run <code>Backpro.py<code>. 

### 2- Recirculation algorithm
The backpropagation algorithm is often thought to be biologically implausible in the brain due to its nonlocal nature. In fact, there is no evidence for back-propagation of error in the brain. Recirculation algorithms introduced by Hinton & McClelland that calculates error locally. Therfore, it is a more biologically plausible algorithm. For more information about this algorithm read **Recirculation.pdf**. To see the algorithm and how it works, run <code>Recirculation.py<code>.
