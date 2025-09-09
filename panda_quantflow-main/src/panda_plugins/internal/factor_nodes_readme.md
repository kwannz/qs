# 因子处理节点说明

本文档介绍新增的两个因子处理节点的功能和使用方法。

## 1. 因子权重调整节点 (FactorWeightAdjustControl)

### 功能描述
对单个因子DataFrame中的所有因子列进行权重调整（乘以指定的权重系数）。

### 输入参数
- **df_factor** (DataFrame): 包含因子值的DataFrame
  - 必须包含 `date` 和 `symbol` 列
  - 至少包含一个因子列（数值类型）
- **weight** (float): 权重系数
  - 范围: -10.0 到 10.0
  - 正数表示同向调整，负数表示反向调整

### 输出结果
- **df_factor** (DataFrame): 权重调整后的因子DataFrame
  - 保持原有的 `date` 和 `symbol` 列不变
  - 所有数值型因子列都乘以指定权重

### 使用示例
```python
# 输入数据
input_df = pd.DataFrame({
    'date': ['20240101', '20240101'],
    'symbol': ['000001.SZ', '000002.SZ'],
    'momentum': [0.1, 0.2],
    'reversal': [-0.05, 0.08]
})

# 权重调整 (权重=2.0)
# 结果: momentum列变为[0.2, 0.4], reversal列变为[-0.1, 0.16]
```

### 节点属性
- **节点名称**: "因子权重调整"
- **分组**: "04-因子相关"
- **颜色**: 蓝色
- **类型**: 通用处理节点

---

## 2. 多因子合并节点 (MultiFactorMergeControl)

### 功能描述
将5个独立的因子DataFrame根据date和symbol对齐合并成一个统一的因子DataFrame。

### 输入参数
- **factor1** (DataFrame): 第一个因子DataFrame
- **factor2** (DataFrame): 第二个因子DataFrame
- **factor3** (DataFrame): 第三个因子DataFrame
- **factor4** (DataFrame): 第四个因子DataFrame
- **factor5** (DataFrame): 第五个因子DataFrame

每个输入DataFrame的要求：
- 必须包含 `date` 和 `symbol` 列
- 必须包含且仅包含一个因子列（除date和symbol外）
- 因子列必须为数值类型

### 输出结果
- **merged_factors** (DataFrame): 合并后的因子DataFrame
  - 列名: `date`, `symbol`, `factor1`, `factor2`, `factor3`, `factor4`, `factor5`
  - 使用外连接(outer join)保留所有数据点
  - 按 `date` 和 `symbol` 排序

### 合并规则
1. **对齐方式**: 根据 `date` 和 `symbol` 进行对齐
2. **合并类型**: 外连接 (保留所有数据组合)
3. **列名标准化**: 因子列重命名为 `factor1`, `factor2`, ..., `factor5`
4. **缺失值处理**: 如果某个因子在特定(date, symbol)组合下没有数据，则填充为NaN

### 使用示例
```python
# 输入: 5个因子DataFrame
factor1_df = pd.DataFrame({
    'date': ['20240101', '20240102'],
    'symbol': ['000001.SZ', '000001.SZ'],
    'momentum': [0.1, 0.15]
})

factor2_df = pd.DataFrame({
    'date': ['20240101'],  # 缺少20240102的数据
    'symbol': ['000001.SZ'],
    'reversal': [-0.05]
})

# ... factor3_df, factor4_df, factor5_df

# 输出: 合并后的DataFrame
# date     symbol     factor1  factor2  factor3  factor4  factor5
# 20240101 000001.SZ  0.1      -0.05    ...      ...      ...
# 20240102 000001.SZ  0.15     NaN      ...      ...      ...
```

### 节点属性
- **节点名称**: "多因子合并"
- **分组**: "04-因子相关"
- **颜色**: 绿色
- **类型**: 通用处理节点

---

## 使用场景

### 典型工作流
1. **因子计算** → 生成5个独立的因子DataFrame
2. **权重调整** → 对各个因子进行权重调整（可选）
3. **因子合并** → 将5个因子合并成统一的多因子DataFrame
4. **后续处理** → 进行因子分析、回测等操作

### 注意事项
1. **数据格式**: 确保所有DataFrame都包含正确的date和symbol列
2. **数据类型**: 因子列必须为数值类型
3. **缺失值**: 合并后可能产生缺失值，需要根据实际情况处理
4. **内存使用**: 大数据集合并时注意内存使用情况
5. **权重范围**: 权重调整节点的权重参数有范围限制(-10.0到10.0)

### 错误处理
- 输入验证失败时会抛出详细的错误信息
- 日志记录包含详细的处理过程和统计信息
- 支持异常情况的优雅处理