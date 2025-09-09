# -*- coding: utf-8 -*-
"""
File: ml_lstm_node.py
Author: Bayoro
Date: 2025/7/15
Description: lstm机器学习模型
"""

from datetime import datetime
import logging

from typing import Optional, Type, List, Union
import uuid
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field
from pathlib import Path
from panda_plugins.base.base_work_node import BaseWorkNode
from panda_plugins.base.work_node_registery import work_node
from panda_plugins.base import ui
from panda_factor.generate.macro_factor import MacroFactor
from panda_plugins.internal.models.common_models import MLModel, MLOutputModel, FeatureModel

logger = logging.getLogger(__name__)

"""
LSTM节点
"""

@ui(
    factor={"input_type": "None"},
    time_step={"input_type": "number_field", "min": 1, "max": 100, "allow_link": False},
    units={"input_type": "number_field", "min": 10, "max": 200, "allow_link": False},
    batch_size={"input_type": "number_field", "min": 16, "max": 128, "allow_link": False},
    epochs={"input_type": "number_field", "min": 1, "max": 100, "allow_link": False},
    dropout_rate={"input_type": "slider", "min": 0.0, "max": 0.5, "allow_link": False},
    learning_rate={"input_type": "slider", "min": 0, "max": 0.01,"allow_link": False},
)
class LSTMInputModel(BaseModel):
    factor: pd.DataFrame = Field(..., title="特征值")
    time_step: int = Field(default=10, title="时间步长")
    units: int = Field(default=50, title="LSTM单元数")
    batch_size: int = Field(default=32, title="批大小")
    epochs: int = Field(default=10, title="训练轮数")
    dropout_rate: float = Field(default=0.2, title="Dropout比例")
    learning_rate: float = Field(default=0.001, title="学习率")

class LSTMModelWrapper:
    """LSTM模型包装类，封装模型和标准化器"""
    def __init__(self, model, scaler_X, scaler_y):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        predictions_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(predictions_scaled).flatten()
    
    def save(self, model_path):
        """保存模型和scaler"""
        import pickle
        save_dict = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }
        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, model_path):
        """加载模型和scaler"""
        import pickle
        with open(model_path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # 创建包装器
        wrapper = cls(save_dict['model'], save_dict['scaler_X'], save_dict['scaler_y'])
        return wrapper

@work_node(
    name="LSTM模型", 
    group="03-机器学习", 
    type="general", 
    box_color="blue"
)
class LSTMControl(BaseWorkNode):
    """Node for LSTM model training and prediction"""

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return LSTMInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MLOutputModel

    def run(self, input: LSTMInputModel) -> MLOutputModel:
        factor_values = input.factor
        time_step = input.time_step 
        units = input.units
        batch_size = input.batch_size
        epochs = input.epochs
        dropout_rate = input.dropout_rate
        learning_rate = input.learning_rate


        # 开始训练
        print("开始训练")
        
        # 初始化标准化器
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # 构造特征 X 和标签 y
        feature_cols = [col for col in factor_values.columns if col != 'label']
        X = factor_values[feature_cols]
        y = factor_values["label"]
        
        # 处理NaN值
        # 删除标签为NaN的样本
        valid_idx = ~y.isna()
        if sum(~valid_idx) > 0:
            print(f"警告：删除了 {sum(~valid_idx)} 个含有NaN标签的样本（占比 {sum(~valid_idx)/len(y):.2%}）")
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            
        if len(X) == 0:
            self.log_error("处理NaN后没有可用的训练数据，请检查label列的计算是否正确")
            raise ValueError("处理NaN后没有可用的训练数据，请检查label列的计算是否正确")
            
        self.log_info(f"训练数据样本数: {len(X)}")
        self.log_info(f"特征数量: {X.shape[1]}")

        # 数据标准化
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        # 创建时间序列数据：使用滑动窗口方法来创建 (样本数, 时间步长, 特征数)
        def create_lstm_data(X, y, time_step):
            X_lstm, y_lstm = [], []
            for i in range(len(X) - time_step):
                X_lstm.append(X[i:(i + time_step)])
                y_lstm.append(y[i + time_step])

            return np.array(X_lstm), np.array(y_lstm)
        X_scaled, y_scaled = create_lstm_data(X_scaled, y_scaled, time_step)

        self.log_info(f"[样本数, 时间步长, 特征数]: [{X_scaled.shape[0]},{time_step},{X.shape[1]}]")

        # 将数据转换为3D输入 [样本数, 时间步长, 特征数]
        X_scaled = X_scaled.reshape((X_scaled.shape[0], time_step, X.shape[1]))
        
        # 初始化并训练LSTM模型
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=False))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        
        # 训练模型
        model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=batch_size, verbose=1)
        self.log_info("训练结束")
        print("训练结束")


        # 设置模型保存路径 - 保存到models文件夹
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        short_uuid = str(uuid.uuid4())[:8]  # 生成8位短UUID
        model_path = model_dir / f"lstm_model_{datetime.now().strftime('%Y%m%d%H%M')}_{short_uuid}.pkl"
        
        # 创建包装对象并保存
        lstm_wrapper = LSTMModelWrapper(model, scaler_X, scaler_y)
        lstm_wrapper.save(model_path)

        ml_model = MLModel(model_path=str(model_path), model_type="lstm")
        return MLOutputModel(model=ml_model)

if __name__ == "__main__":

    # 创建测试数据 - 5个因子DataFrame (使用MultiIndex)
    dates = ['20240101', '20240102', '20240103', '20240104', '20240105', '20240106', '20240107']
    symbols = ['000001.SZ', '000002.SZ', '000003.SZ']
    
    # 创建MultiIndex
    index_data = []
    for date in dates:
        for symbol in symbols:
            index_data.append((date, symbol))

    index5 = pd.MultiIndex.from_tuples(index_data, names=['date', 'symbol'])
    factor5_df = pd.DataFrame({'close': np.random.normal(0, 0.11, len(index5)),'label': np.random.normal(0, 0.15, len(index5))}, index=index5)



    input_value = LSTMInputModel(factor=factor5_df,units=50,batch_size=32,epochs=10,dropout=0.2,time_step=3)
    node = LSTMControl()
    print(node.run(input_value))

