# 自动微分

我们实现的是reverse模式的自动微分，reverse模式下我们从后往前进行计算。另外我们使用的是静态图，需要先使用占位符构建表达式，得到静态计算图，然后填充数据，根据静态计算图执行计算。


```python
# 创建两个占位符变量，在计算图中充当叶子节点
x2 = ad.Variable(name = "x2")
x3 = ad.Variable(name = "x3")
# 构建一个静态计算图
y = x2 + x3
```
**构建计算梯度的计算图**
为了实现自动微分，我们需要构建反向传播的计算图。
```python
# 构建计算梯度的计算图，求y对x2和x3的梯度
grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
```

构建梯度计算图如下：
```python
def gradients(output_node, node_list):
    # a map from node to the gradient of that node
    node_to_output_grad = {}
	# 输出节点对自身的梯度为1
    node_to_output_grad[output_node] = oneslike_op(output_node)
    # 拓扑排序并将结果倒置，因为我们的梯度计算是从后往前计算
	# 比如y = x1 + x1*x2
	# [y, x1*x2, x2, x1]
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    # 遍历节点
    for node in reverse_topo_order:
        # 根据链式法则，对node的输入求梯度
        grads = node.op.gradient(node, node_to_output_grad[node])
        for i, n in enumerate(node.inputs):
            if n in node_to_output_grad:
                node_to_output_grad[n] += grads[i]
            else:
                node_to_output_grad[n] = grads[i]
    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list
```

**执行计算**
执行计算前，我们需要创建一个执行器，确定需要计算哪些节点的值。
```python
# 创建一个执行器
executor = ad.Executor([y, grad_x2, grad_x3])
x2_val = 2 * np.ones(3)
x3_val = 3 * np.ones(3)
# 填充数据，执行计算
y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})
```
前向计算：执行计算前，先对计算图进行拓扑排序，确定需要计算的节点所依赖的节点，比如计算表达式 y = x1 + x1*x2，拓扑排序之后需要计算的节点依次为：[x1, x2, x1*x2, y]，然后我们遍历这个列表计算每个节点的值：
* 对于叶子节点：直接使用填充数据
* 对于非叶子节点的计算：使用前驱节点的计算结果进行计算，比如计算x1*x2，我们需要使用节点x1和x2的值。