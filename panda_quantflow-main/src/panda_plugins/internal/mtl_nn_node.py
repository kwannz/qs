from datetime import datetime
import logging

from typing import Optional, Type
import uuid
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field
from pathlib import Path
from panda_plugins.base.base_work_node import BaseWorkNode
from panda_plugins.base.work_node_registery import work_node
from panda_plugins.base import ui
from panda_factor.generate.macro_factor import MacroFactor
from panda_plugins.internal.models.common_models import MLModel, MLOutputModel,FeatureModel
from common.utils.index_calculate import get_factors_mutil

class MTLNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.shared(x)
        return self.head(x)

class MTLNNModelWrapper:
    def __init__(self, model, scaler_X, scaler_y):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).numpy()
        return self.scaler_y.inverse_transform(y_pred_scaled)
    
    def save(self, model_path):
        """保存模型和scaler"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.shared[0].in_features,
                'output_dim': self.model.head.out_features,
                'hidden_dim': self.model.shared[0].out_features
            },
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }
        torch.save(save_dict, model_path)
    
    @classmethod
    def load(cls, model_path):
        """加载模型和scaler"""
        # 由于sklearn对象包含复杂的numpy依赖，使用weights_only=False
        # 这是安全的，因为这是我们内部训练和保存的模型
        save_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 重建模型
        config = save_dict['model_config']
        model = MTLNet(config['input_dim'], config['output_dim'], config['hidden_dim'])
        model.load_state_dict(save_dict['model_state_dict'])
        
        # 创建包装器
        wrapper = cls(model, save_dict['scaler_X'], save_dict['scaler_y'])
        return wrapper

@ui(
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
    epochs={"input_type": "number_field"},
    hidden_dim={"input_type": "number_field"},
    lr={"input_type": "slider", "min": 0.0001, "max": 0.1, "step": 0.0001}
)
class MTLNNInputModel(BaseModel):
    feature1: FeatureModel | None= Field(default=None, title="特征工程1")
    feature2: FeatureModel | None= Field(default=None, title="特征工程2")
    feature3: FeatureModel | None= Field(default=None, title="特征工程3")
    feature4: FeatureModel | None= Field(default=None, title="特征工程4")
    feature5: FeatureModel | None= Field(default=None, title="特征工程5")
    start_date: str = Field(default="20250101", title="训练开始时间")
    end_date: str = Field(default="20250301", title="训练结束时间")
    epochs: int = Field(default=100, title="训练轮数")
    hidden_dim: int = Field(default=64, title="隐藏层维度")
    lr: float = Field(default=0.001, title="学习率")

@work_node(
    name="多任务神经网络",
    group="03-机器学习",
    type="general",
    box_color="red"
)
class MTLNNControl(BaseWorkNode):

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return MTLNNInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MLOutputModel

    def run(self, input: MTLNNInputModel) -> MLOutputModel:
        macro_factor = MacroFactor()
        factors = [f for f in
                   [input.feature1, input.feature2, input.feature3, input.feature4, input.feature5] if
                   f and f.features and f.features.strip()]
        factor_values=get_factors_mutil(feature_list=factors,start_date=input.start_date, end_date=input.end_date)

        # if input.feature.features:
        #     factors = input.feature.features.split("\n")
        #     factor_values = macro_factor.create_factor_from_formula_pro(
        #         factor_logger=logger,
        #         formulas=factors,
        #         start_date=input.start_date,
        #         end_date=input.end_date
        #     )
        # else:
        #     raise ValueError("因子不能为空")
        #
        # if input.feature.label:
        #     labels = input.feature.label.split("\n")
        #     for lb in labels:
        #         factor_values[lb] = macro_factor.create_factor_from_formula(
        #             factor_logger=logger,
        #             formula=lb,
        #             start_date=input.start_date,
        #             end_date=input.end_date
        #         )
        # else:
        #     raise ValueError("标签不能为空")
        factor_cols = [col for col in factor_values.columns if col.startswith("factor")]
        label_cols = [col for col in factor_values.columns if col.startswith("label")]
        # feature_cols = [col for col in factor_values.columns if col not in labels]
        X = factor_values[factor_cols].values
        y = factor_values[label_cols].values

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        input_dim = X.shape[1]
        output_dim = y.shape[1]

        model = MTLNet(input_dim, output_dim, hidden_dim=input.hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=input.lr)
        loss_fn = nn.MSELoss()
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)

        for epoch in range(input.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = loss_fn(output, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        short_uuid = str(uuid.uuid4())[:8]  # 生成8位短UUID
        model_path = model_dir / f"mtl_model_{datetime.now().strftime('%Y%m%d%H%M')}_{short_uuid}.pth"

        wrapped_model = MTLNNModelWrapper(model, scaler_X, scaler_y)
        wrapped_model.save(model_path)

        ml_model = MLModel(model_path=str(model_path), model_type="mtl_nn")
        return MLOutputModel(model=ml_model)
