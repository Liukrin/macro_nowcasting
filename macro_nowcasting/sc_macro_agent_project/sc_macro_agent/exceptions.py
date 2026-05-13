"""自定义异常定义。"""

from typing import Optional


class ProjectError(Exception):
    """项目基础异常。"""

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} | hint={self.hint}"
        return self.message


class ConfigurationError(ProjectError):
    """配置不合法。"""


class DataContractError(ProjectError):
    """数据字段或口径不符合预期。"""


class DataNotReadyError(ProjectError):
    """数据未准备好。"""


class FeatureBuildError(ProjectError):
    """特征构建失败。"""


class ModelTrainingError(ProjectError):
    """模型训练失败。"""


class BacktestError(ProjectError):
    """回测失败。"""


class ArtifactError(ProjectError):
    """产物读写失败。"""


class APIServingError(ProjectError):
    """API 服务错误。"""
