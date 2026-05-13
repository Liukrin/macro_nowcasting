from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os


@dataclass   # 规格说明书，物料清单
class DataConfig:
    data_dir: str = "./data"
    dataset_mode: str = "auto"   # auto / real / demo / hybrid
    quarterly_target_file: str = "quarterly_target_real.csv"
    monthly_local_file: str = "monthly_local_features_real.csv"
    monthly_national_file: str = "monthly_national_features_real.csv"
    quarterly_panel_file: str = "quarterly_feature_panel_real.csv"
    metadata_file: str = "metadata_real.csv"

    demo_quarterly_target_file: str = "quarterly_target.csv"
    demo_monthly_local_file: str = "monthly_local_features.csv"
    demo_monthly_national_file: str = "monthly_national_features.csv"
    demo_quarterly_panel_file: str = "quarterly_feature_panel.csv"
    demo_metadata_file: str = "metadata.csv"

    pmi_excel_file: str = "pmi_data_202502.xls"

    allow_demo_fallback: bool = True
    merge_demo_for_backtest: bool = False
    cache_intermediate: bool = True # 中间结果要缓存（省得重复计算）
    artifact_dir: str = "./artifacts"

    def resolve_dir(self) -> Path:
        # 定位到 sc_macro_agent_project 目录（config.py 的上级目录）
        base_dir = Path(__file__).parent.parent
        return (base_dir / self.data_dir).expanduser().resolve() # 这里是为了代码的可移植性,相对变绝对，波浪号展开，跨平台兼容

    def resolve_artifact_dir(self) -> Path:
        path = Path(self.artifact_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True) # 智能模式：柜子没有？先建柜子！再建抽屉！文件夹已有？没事， quietly pass~ 缺就建，有就用
        return path

    def as_dict(self) -> Dict[str, Any]: # 字典是所有程序都能听懂的话，更方便把配置保存为JSON，或者穿传给网页界面显示
        return asdict(self)


@dataclass
class FeatureConfig:
    # 目标变量
    target_indicator: str = "GDP_同比增速"

    # 数据清洗阈值
    max_feature_missing_ratio: float = 0.65
    min_non_na_observations: int = 3
    winsorize_quantile: float = 0.01
    use_quarterly_panel_if_available: bool = True

    # 特征工程开关（一长串True/False）
    build_monthly_aggregations: bool = True
    use_mean_agg: bool = True
    use_last_agg: bool = True
    use_std_agg: bool = True
    use_min_agg: bool = True
    use_max_agg: bool = True
    use_trend_agg: bool = True
    use_qoq_delta_agg: bool = True
    use_range_agg: bool = True
    use_availability_flags: bool = True

    # 时间特征,field--独立性而不互相干扰
    add_target_lags: bool = True
    target_lags: List[int] = field(default_factory=lambda: [1, 2, 4])
    add_target_rolling: bool = True
    target_roll_windows: List[int] = field(default_factory=lambda: [2, 4])
    add_feature_lags: bool = True
    feature_lags: List[int] = field(default_factory=lambda: [1, 2])

    include_region_prefix: bool = True
    include_family_features: bool = True
    family_top_n: int = 4
    allow_cross_region_features: bool = True
    fill_method: str = "ffill_bfill"

    # 降维与交互
    standardize_before_pca: bool = True
    pca_n_factors: int = 3
    clip_outliers: bool = True
    add_interactions: bool = True
    interaction_top_pairs: int = 8
    small_sample_feature_cap: int = 40

    def as_dict(self) -> Dict[str, Any]:  # 数据交换的通用语言
        return asdict(self) # #


@dataclass
class ModelConfig:


    # 控制"模型选择策略"的超参数,给模型定的规矩
    candidate_models: List[str] = field(default_factory=lambda: [
        "last_value",
        "mean_recent",
        "ridge_midas",
        "elastic_midas",
        "hybrid_residual",
    ])
    primary_model: str = "auto"


    # 控制"随机性"的超参数
    random_state: int = 42


    # 控制"模型复杂度"的超参数
    ridge_alphas: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    elastic_alphas: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0])
    elastic_l1_ratios: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    residual_model_kind: str = "rf"


    # 控制"树模型"结构的超参数
    residual_n_estimators: int = 200
    residual_max_depth: int = 3
    residual_min_samples_leaf: int = 2
    feature_selection_top_k: int = 60


    # 控制"安全阈值"的超参数
    min_train_rows_for_tree: int = 16
    min_train_rows_for_hybrid: int = 12
    min_train_rows_for_regularized: int = 8

    use_dfm_features: bool = True
    dfm_n_factors: int = 3
    dfm_regress_target: bool = True

    use_model_blend: bool = True
    blend_weights: Optional[Dict[str, float]] = None
    confidence_level: float = 0.8

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestConfig:
    enabled: bool = True

    # 窗口设置
    scheme: str = "expanding"
    initial_train_quarters: int = 4
    min_required_test_windows: int = 2
    max_test_windows: int = 8

    # 预测设置
    horizon: int = 1
    step: int = 1

    # 模型更新
    refit_each_window: bool = True

    # 结果保存
    store_predictions: bool = True
    store_feature_importance: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentConfig:
    enable_agent: bool = True
    plan_verbose: bool = True
    auto_generate_reports: bool = True
    default_goal: str = "audit_build_train_backtest_report"
    save_checkpoints: bool = True
    checkpoint_prefix: str = "agent_run"
    allow_repair_actions: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AppConfig: # 从哪来、往哪去、怎么存

    # 项目身份证（元信息）
    project_name: str = "四川省GDP混频预测 - Agent Pipeline"
    version: str = "2.0.0"
    debug: bool = False
    timezone: str = "Asia/Beijing"

    # 服务部署信息（FastAPI/Streamlit 用的）
    host: str = "0.0.0.0"
    port: int = 8000

    # 五大子配置,default_factory 保证每次创建 AppConfig() 时，都重新生成一个新的 DataConfig(),大家互不干扰。
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "version": self.version,
            "debug": self.debug,
            "timezone": self.timezone,
            "host": self.host,
            "port": self.port,

            "data": self.data.as_dict(),
            "features": self.features.as_dict(),
            "model": self.model.as_dict(),
            "backtest": self.backtest.as_dict(),
            "agent": self.agent.as_dict(),
        }

    @classmethod
    def from_json(cls, path: str) -> "AppConfig":
        """从 JSON 文件加载配置"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AppConfig":
        """从字典加载配置（支持部分更新）"""
        cfg = cls()

        # 1. 更新顶层简单字段（项目名、端口等）
        for key in ["project_name", "version", "debug", "timezone", "host", "port"]:
            if key in payload:
                setattr(cfg, key, payload[key])
        # 2. 更新嵌套的子配置
        if "data" in payload:
            cfg.data = DataConfig(**payload["data"])
        if "features" in payload:
            cfg.features = FeatureConfig(**payload["features"])
        if "model" in payload:
            cfg.model = ModelConfig(**payload["model"])
        if "backtest" in payload:
            cfg.backtest = BacktestConfig(**payload["backtest"])
        if "agent" in payload:
            cfg.agent = AgentConfig(**payload["agent"])

        return cfg

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg = cls()
        cfg.debug = os.getenv("SC_MACRO_DEBUG", str(cfg.debug)).lower() == "true"
        cfg.host = os.getenv("SC_MACRO_HOST", cfg.host)
        cfg.port = int(os.getenv("SC_MACRO_PORT", str(cfg.port)))
        cfg.data.data_dir = os.getenv("SC_MACRO_DATA_DIR", cfg.data.data_dir)
        cfg.data.dataset_mode = os.getenv("SC_MACRO_DATASET_MODE", cfg.data.dataset_mode)
        cfg.data.artifact_dir = os.getenv("SC_MACRO_ARTIFACT_DIR", cfg.data.artifact_dir)
        return cfg

    def save_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )