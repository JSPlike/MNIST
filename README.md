# MNIST
## H(x) = W(x) + b

```
    x = [1, 2, 3]
    y = [1, 2, 3]
    
    # 값이 하나인 1차원 배열을 만든다
    W = tf.Variable(tf.random_normal([1]), name = 'weight')
    b = tf.Variable(tf.ramom_normal([1]), name = 'bias')
    
    hypothesis = x * W + b
```
cost(W, b) = 

$$ \frac{1}{m}\sum_{i=1}^{m} (H(x^i) - y^i)^2 $$

```
 cost = reduce_mean(tf.square(hypothesis - y))
```
