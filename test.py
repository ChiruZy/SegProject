import torch.autograd.function as Function
import torch

class Exp(Function):                    # 此层计算e^x

    @staticmethod
    def forward(ctx, i):                # 模型前向
        result = i.exp()
        ctx.save_for_backward(result)   # 保存所需内容，以备backward时使用，所需的结果会被保存在saved_tensors元组中；此处仅能保存tensor类型变量，若其余类型变量（Int等），可直接赋予ctx作为成员变量，也可以达到保存效果
        return result

    @staticmethod
    def backward(ctx, grad_output):     # 模型梯度反传
        result, = ctx.saved_tensors     # 取出forward中保存的result
        return grad_output * result     # 计算梯度并返回

# 尝试使用
x = torch.tensor([1.], requires_grad=True)  # 需要设置tensor的requires_grad属性为True，才会进行梯度反传
ret = Exp.apply(x)                          # 使用apply方法调用自定义autograd function
print(ret)                                  # tensor([2.7183], grad_fn=<ExpBackward>)
ret.backward()                              # 反传梯度
print(x.grad)


