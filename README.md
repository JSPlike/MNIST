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
    #reduce_mean : 어떠한 텐서가 주어지면 그 텐서의 평균값을 구한다
    ex)....
    
    t = [1, 2, 3, 4]
    tf.reduce_mean(t) = 2.5
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))
 
```

보통은 placeholder를 사용하여 위의 코드를 새롭게 표현할 수 있다.

```
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    
    cost = tf.reduce_mean(tf.square(hypothesis-y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    
    train = optimizer.minimize(cost)
    sess = tf.session()
    
    sess.run(tf.global_variables_initializer())
    
    
    #trin에 나오는 값은 별로 중요하지 않기에 "_"로 채워놓았다.
    
    for step in range(2001):
    
        #임시 변수에 훈련시킨 값들을 담는다
    
        cost_val, W_val, b_val, _ = \
            sess.run(cost, W, b, train),
                feed_dict = {x: [1, 2, 3], y: [1, 2, 3]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
```
