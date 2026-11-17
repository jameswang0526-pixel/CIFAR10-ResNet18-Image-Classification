"""数据模块统一入口：from dataload import DataConfig, build_dataloaders。"""
from .cogif import DataConfig
from .dataset import build_dataloaders

__all__ = ["DataConfig", "build_dataloaders"]