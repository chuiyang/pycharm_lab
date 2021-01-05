import numpy as np
import math


def make_points(x_array, dx=0.0001):
    points = [x_array]  # i+dx, i-dx
    for i, val in enumerate(x_array):
        x_array_i_p = x_array.copy()
        x_array_i_m = x_array.copy()
        x_array_i_p[i] = val + dx
        x_array_i_m[i] = val - dx
        points.append(x_array_i_p)
        points.append(x_array_i_m)
    points = np.array(points, dtype=np.float)
    return points


def gradient(prediction, dx=0.0001):  # 2048*2 -> 一個分子
    grad = []
    for i in range(int(len(prediction)/2)):
        i_p = prediction[2*(i+1)-1]
        i_m = prediction[2*(i+1)]
        g = (i_p-i_m) / (dx*2)
        grad.append(float(g))
    grad_rms = math.sqrt(sum([k*k for k in grad])/len(grad))
    grad_max = max([(k*k)**(1/2) for k in grad])
    return grad_rms, grad_max


def gradient_a(prediction, b_num, dx=0.00001):  #
    if b_num == 0:
        return 0, 0
    # 有588預測值 2個bond vector atom平均 找atoms之間最大的grad當rmsgrad
    t = len(prediction) / 2
    forward_pred = prediction[:int(t)]
    backward_pred = prediction[int(t):]
    # print('len(forward_pred)', len(forward_pred))
    # print('len(backward_pred)', len(backward_pred))
    # print(forward_pred[:10])
    # print(backward_pred[:10])
    grad_max = []
    for b_vec in range(b_num):
        grad_b = []
        for j in range(147):
            i = b_vec*147 + j
            i_p = forward_pred[i]
            i_m = backward_pred[i]
            g = (i_p-i_m) / (dx*2)
            grad_b.append(g)
        grad_rms_b = math.sqrt(sum([k*k for k in grad_b])/len(grad_b))
        print('len:', len(grad_b))
        grad_max.append(grad_rms_b)
    # print(gradmax)
    return sum(grad_max)/len(grad_max), max(grad_max)




