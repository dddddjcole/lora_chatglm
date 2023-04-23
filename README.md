# LORA: LOW-RANK ADAPTATION 论文理解



##### 总体作用： Lora在大的语言模型上，可以减少训练的参数，对于模型的推理(预测)没有延迟，快速微调模型以适应下游任务



### 思想：以Adam优化器来更新参数为例  [Adam论文地址](https://arxiv.org/pdf/1412.6980.pdf)

```python
#Adam 算法逻辑，具体实现原理请参考Adam论文 https://arxiv.org/pdf/1412.6980.pdf
m = beta1*m + (1‐beta1)*dx
v = beta2*v + (1‐beta2)*(dx**2)
x += ‐ learning_rate * m / (np.sqrt(v) + eps)
```

在Adam优化器中需要维护一个momentum(**m**)和一个variance(**v**)来更新参数，在使用Nvidia卡做运算的时候，通常会使用混合精度的方式。对于每一个参数**w**来说采用FP32来存贮，在前向传播中会把**w**复制一份并转为FP16以提高计算速度（神经网络中输入**x**和输出**y**均采用FP16）。综上，对于一个参数而言我们需要更新我们需要16个字节，四个字节**w**原始值，两个字节**w**复制，四个字节**m**，四个字节**v**，和两个字节的梯度。

![Image text](https://github.com/dddddjcole/lora_chatglm/raw/main/image/image-3.png)

这时我们发现一个问题，我们保存一个模型的参数的时候，每个参数只需要四个字节，但是当我们更新的时候却需要维护16个字节的内存。**当我们想要对模型的整体进行微调的时候，就不得不加载过多的参数，为了解决这一问题Lora诞生了。**



### 具体实现：![Image text](https://github.com/dddddjcole/lora_chatglm/blob/main/image/image-1.png)

根据上述内容得知，主要**矛盾点**在于想对模型整体进行微调又不想加载过多的参数，解决办法是构造一个和原来一样维度的矩阵

△W，这个矩阵上的值就是要对整个模型优化的增量。
![Image text](https://github.com/dddddjcole/lora_chatglm/blob/main/image/1.jpg)


**如果我们仅仅构造了一个△W，参数所占的内存并没有得到改善，因为两个矩阵的维度并没有发生变化**

这时构造了一个B矩阵和A矩阵，其中B矩阵时(r x d) ，A矩阵是（d x r）

![Image text](https://github.com/dddddjcole/lora_chatglm/blob/main/image/image-2.png)
![Image text](https://github.com/dddddjcole/lora_chatglm/blob/main/image/2.jpg)
为什么要这样做呢，因为对△W来说我们们仅仅拿来做微调，就算不是做微调，就原始的参数矩阵W而言，也很大概率不是满秩矩阵，所以相当于把△W做了一个矩阵分解，其中r是超参数。对于r的选取，如果模型偏差很大，可以根据自己的计算资源尽量调大r，如果模型差别较小，可以把r调小。r的取值决定了我们要更新的参数维度。

**参数更新上**
![Image text](https://github.com/dddddjcole/lora_chatglm/blob/main/image/3.jpg)
拿真实的H-预测h得到损失，计算梯度，更新参数。

#### 总结：为什么说减少了训练参数？因为采用了两个低秩矩阵，来对整个W进行“局部”更新。为什么说不影响推理（预测）？因为我们只对模型的全部参数进行了调整，并没有增加模型的层，也没有改变模型的结构，Lora仅仅在训练的时候起作用。
