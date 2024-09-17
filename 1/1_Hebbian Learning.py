import numpy as np

# 将训练数据转换为向量
train_data = [
    [[-1,1,1,1,-1], [1,-1,-1,-1,1], [1,-1,-1,-1,1], [1,-1,-1,-1,1], [1,-1,-1,-1,1], [-1,1,1,1,-1]],  # 0
    [[-1,1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,1,-1,-1]],  # 1
    [[1,1,1,-1,-1], [-1,-1,-1,1,-1],[-1,-1,-1,1,-1],[-1,1,1,-1,-1], [-1,1,-1,-1,-1],[-1,1,1,1,1]]   # 2
]

# 测试数据 (残缺的图片0)
# 要求中只给一张，懒得自己生成，就一张吧
test_data = [
    [-1,1,1,1,-1], [1,-1,-1,-1,1], [1,-1,-1,-1,1], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]
]

# 将2D矩阵转换为1D向量
def flatten_image(image):
    return np.array(image).flatten()

# Hebb学习规则更新权重
def hebb_train(data):
    num_features = len(data[0]) * len(data[0][0])  # 计算输入的特征数
    weight_matrix = np.zeros((num_features, num_features)) # 权重

    for img in data:
        img_vector = flatten_image(img)
        weight_matrix += np.outer(img_vector, img_vector)  # 外积计算

    np.fill_diagonal(weight_matrix, 0)  # 对角线置0，避免自联想      # 对于作业情况，不加这行也测不出什么差别
    return weight_matrix

# Hebb回忆过程
def recall(weight_matrix, input_image):
    input_vector = flatten_image(input_image)
    output = np.dot(weight_matrix, input_vector)  # W * vec_input
    output_vector = np.sign(output)  # 将输出二值化为1、-1
    return output_vector.reshape(len(input_image), len(input_image[0]))

# 测试函数，展示图片
def display_image(image):
    for row in image:
        print(' '.join(['■' if x == 1 else '□' for x in row]))

# 训练Hebb网络
weight_matrix = hebb_train(train_data)

# 测试残缺图片并回忆
recalled_image = recall(weight_matrix, test_data)

# 显示结果
print("Origianl Image:")
display_image(test_data)

print("\nRecovered Image:")
display_image(recalled_image)
