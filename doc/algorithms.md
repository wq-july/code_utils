# 算法思路记录

## 点云地图

介绍: 可以存储点云的数据结构，同时可以快速的实现增删查的方法，在slam中比较适合的方法是kdtree，Voxel体素方法。

### kdtree


### voxel



## 非线性优化方法

1. 介绍：非线性优化方法，是slam中后端优化的核心，配合回环检测效果更好，目前开源库有ceres, g2o, gtsam。非线性优化方法需要了解底层的
优化算法，主要有GaussNewton（GN）, Levenberg-Marquadt（LM），DOG等，在slam中主要用来求解非线性最小二乘问题，需要深入掌握底层原理；

2. 基本问题：
  $$ F(x)=\min _{x} \frac{1}{2}\|f(x)\|_{2}^{2} $$
    
  - 其中$x$可能为多维元素，比如slam中的状态量。



### GaussNewton (GN，高斯牛顿法)

1. 推导
  - 将函数$f(x)$进行一阶泰勒展开（而不是对整个误差函数进行展开）：

    $$ f(x + \Delta x) \approx f(x) + J(x) \Delta x$$

  - 这里 $J(x)$ 为 $f(x)$ 关于 x 的导数，实际上是一个 $m × n$ 的矩阵，也是一个雅可比矩阵。根据前面的框架，当前的目标是为了寻找下降矢量 $∆x$，使得 $∥f (x + ∆x)∥2$ 达到最小。为了求 $∆x$，我们构建 一个线性的最小二乘问题：

    $$
    \frac{1}{2} \|f(x) + J(x)\Delta x\|^2 = \frac{1}{2} (f(x) + J(x)\Delta x)^T (f(x) + J(x)\Delta x) \\
    = \frac{1}{2} \left( \|f(x)\|_2^2 + 2 f(x)^T J(x) \Delta x + \Delta x^T J(x)^T J(x) \Delta x \right).
    $$
 
  - 求上式关于 $∆x$ 的导数，并令其为零：
   
    $$ 2 J(x)^T f(x) + 2 J(x)^T J(x) \Delta x = 0.$$
    $$ J(x)^T J(x) \Delta x = -J(x)^T f(x). $$

  - 我们要求解的变量是 $∆x$，这是一个线性方程组，我们称它为增量方程或高斯牛顿方程 (Gauss Newton equations) 或者正规方程 (Normal equations)。我们把左边的系数定义为 $H$，右边定义为 $g$，那么上式变为:
  
    $$H \Delta x = g.$$
    $$ H = J^TJ $$


2. 优缺点总结

        对比牛顿法可见，高斯牛顿法用 J矩阵的转置乘以J矩阵作为牛顿法中二阶 H 矩阵的近似，从而省略了计算 H 的过程。求解增量方程是整个优化问题的核心所在。原则上，它要求近似的矩阵H是可逆的（而且是正定的），而实际计算中得到的JTJ却是半正定的。也就是使用高斯牛顿法会出现JTJ为奇异或者病态情况，此时增量的稳定性较差，导致算法不收敛。即使H非奇异也非病态，如果求得的Δx非常大,也会导致我们采用的局部近似不够正确，这样以来可能不能保证收敛，哪怕是还有可能让目标函数更大。即使高斯牛顿法具有它的缺点，但是很多非线性优化可以看作是高斯牛顿法的一个变种，这些算法结合了高斯牛顿法的优点并修正其缺点。例如LM算法，尽管它的收敛速度可能比高斯牛顿法更慢，但是该方法健壮性更强，也被称为阻尼牛顿法。
    
3. 算法步骤


### LevenbergMarquadt (LM，列文伯格-马尔夸特法)
