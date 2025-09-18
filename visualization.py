# 数据可视化示例代码
import matplotlib.pyplot as plt
import pandas as pd

# 示例数据
# 如果你有自己的数据文件，可以修改这里的读取方式
# data = pd.read_csv('your_file.csv')
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
})

# 绘制折线图
plt.plot(data['x'], data['y'])
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('简单数据可视化')
plt.show()
