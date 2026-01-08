"""插件系统模块

提供动态插件加载能力：
- 从目录加载插件
- 插件生命周期管理
- 插件依赖检查
- 热重载支持
"""

import importlib.util
import inspect
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 插件元数据
# ============================================================


class PluginStatus(Enum):
    """插件状态"""
    DISCOVERED = "discovered"  # 已发现
    LOADED = "loaded"  # 已加载
    ENABLED = "enabled"  # 已启用
    DISABLED = "disabled"  # 已禁用
    ERROR = "error"  # 错误


@dataclass
class PluginMeta:
    """插件元数据"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }


@dataclass
class PluginInfo:
    """插件信息"""
    meta: PluginMeta
    path: Path
    status: PluginStatus = PluginStatus.DISCOVERED
    instance: Optional["Plugin"] = None
    error: Optional[str] = None


# ============================================================
# 插件基类
# ============================================================


class Plugin(ABC):
    """插件基类"""

    # 子类应该重写这些类属性
    meta = PluginMeta(name="BasePlugin")

    @abstractmethod
    def on_load(self) -> None:
        """插件加载时调用"""
        pass

    @abstractmethod
    def on_enable(self) -> None:
        """插件启用时调用"""
        pass

    def on_disable(self) -> None:
        """插件禁用时调用"""
        pass

    def on_unload(self) -> None:
        """插件卸载时调用"""
        pass


class StrategyPlugin(Plugin):
    """策略插件基类"""

    @abstractmethod
    def get_strategy_class(self) -> Type:
        """返回策略类"""
        pass


class DataSourcePlugin(Plugin):
    """数据源插件基类"""

    @abstractmethod
    def get_source_class(self) -> Type:
        """返回数据源类"""
        pass


class IndicatorPlugin(Plugin):
    """指标插件基类"""

    @abstractmethod
    def get_indicators(self) -> Dict[str, Callable]:
        """返回指标函数字典"""
        pass


# ============================================================
# 插件管理器
# ============================================================


class PluginManager:
    """插件管理器"""

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """
        Args:
            plugin_dirs: 插件目录列表
        """
        self.plugin_dirs = plugin_dirs or []
        self._plugins: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = {}

    def add_plugin_dir(self, path: Path) -> "PluginManager":
        """添加插件目录"""
        if path not in self.plugin_dirs:
            self.plugin_dirs.append(path)
        return self

    def discover(self) -> List[PluginInfo]:
        """发现所有插件"""
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.warning(f"插件目录不存在: {plugin_dir}")
                continue

            # 查找 .py 文件
            for py_file in plugin_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    info = self._discover_plugin(py_file)
                    if info:
                        discovered.append(info)
                        self._plugins[info.meta.name] = info
                except Exception as e:
                    logger.error(f"发现插件失败 {py_file}: {e}")

            # 查找包（带 __init__.py 的目录）
            for subdir in plugin_dir.iterdir():
                if subdir.is_dir() and (subdir / "__init__.py").exists():
                    try:
                        info = self._discover_plugin(subdir / "__init__.py")
                        if info:
                            discovered.append(info)
                            self._plugins[info.meta.name] = info
                    except Exception as e:
                        logger.error(f"发现插件失败 {subdir}: {e}")

        logger.info(f"发现 {len(discovered)} 个插件")
        return discovered

    def _discover_plugin(self, path: Path) -> Optional[PluginInfo]:
        """发现单个插件"""
        # 加载模块
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 查找 Plugin 子类
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Plugin) and obj is not Plugin:
                # 跳过基类
                if obj in (StrategyPlugin, DataSourcePlugin, IndicatorPlugin):
                    continue

                meta = getattr(obj, "meta", None)
                if meta is None:
                    meta = PluginMeta(name=name)

                return PluginInfo(
                    meta=meta,
                    path=path,
                    status=PluginStatus.DISCOVERED,
                )

        return None

    def load(self, name: str) -> bool:
        """加载插件"""
        if name not in self._plugins:
            logger.error(f"插件不存在: {name}")
            return False

        info = self._plugins[name]

        if info.status in (PluginStatus.LOADED, PluginStatus.ENABLED):
            logger.warning(f"插件已加载: {name}")
            return True

        try:
            # 检查依赖
            for dep in info.meta.dependencies:
                if dep not in self._plugins:
                    raise ValueError(f"缺少依赖: {dep}")
                if self._plugins[dep].status not in (PluginStatus.LOADED, PluginStatus.ENABLED):
                    self.load(dep)

            # 加载模块
            spec = importlib.util.spec_from_file_location(info.path.stem, info.path)
            if spec is None or spec.loader is None:
                raise ValueError("无法加载模块")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"plugins.{info.meta.name}"] = module
            spec.loader.exec_module(module)

            # 实例化插件
            for obj_name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Plugin) and obj not in (
                    Plugin, StrategyPlugin, DataSourcePlugin, IndicatorPlugin
                ):
                    instance = obj()
                    instance.on_load()
                    info.instance = instance
                    break

            info.status = PluginStatus.LOADED
            logger.info(f"插件已加载: {name}")
            return True

        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error = str(e)
            logger.error(f"加载插件失败 {name}: {e}")
            return False

    def enable(self, name: str) -> bool:
        """启用插件"""
        if name not in self._plugins:
            return False

        info = self._plugins[name]

        if info.status == PluginStatus.ENABLED:
            return True

        if info.status != PluginStatus.LOADED:
            if not self.load(name):
                return False

        try:
            if info.instance:
                info.instance.on_enable()
            info.status = PluginStatus.ENABLED
            logger.info(f"插件已启用: {name}")
            return True

        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error = str(e)
            logger.error(f"启用插件失败 {name}: {e}")
            return False

    def disable(self, name: str) -> bool:
        """禁用插件"""
        if name not in self._plugins:
            return False

        info = self._plugins[name]

        if info.status != PluginStatus.ENABLED:
            return True

        try:
            if info.instance:
                info.instance.on_disable()
            info.status = PluginStatus.DISABLED
            logger.info(f"插件已禁用: {name}")
            return True

        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error = str(e)
            logger.error(f"禁用插件失败 {name}: {e}")
            return False

    def unload(self, name: str) -> bool:
        """卸载插件"""
        if name not in self._plugins:
            return False

        info = self._plugins[name]

        try:
            if info.status == PluginStatus.ENABLED:
                self.disable(name)

            if info.instance:
                info.instance.on_unload()
                info.instance = None

            # 从 sys.modules 移除
            module_name = f"plugins.{info.meta.name}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            info.status = PluginStatus.DISCOVERED
            logger.info(f"插件已卸载: {name}")
            return True

        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error = str(e)
            logger.error(f"卸载插件失败 {name}: {e}")
            return False

    def reload(self, name: str) -> bool:
        """重新加载插件"""
        self.unload(name)
        return self.enable(name)

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """获取插件实例"""
        if name in self._plugins:
            return self._plugins[name].instance
        return None

    def get_plugins(
        self,
        plugin_type: Optional[Type[T]] = None,
        status: Optional[PluginStatus] = None,
    ) -> List[T]:
        """获取插件列表"""
        results = []

        for info in self._plugins.values():
            if status and info.status != status:
                continue

            if info.instance is None:
                continue

            if plugin_type and not isinstance(info.instance, plugin_type):
                continue

            results.append(info.instance)

        return results

    def list_plugins(self) -> Dict[str, PluginInfo]:
        """列出所有插件"""
        return dict(self._plugins)

    # ============================================================
    # 钩子系统
    # ============================================================

    def register_hook(self, name: str, callback: Callable) -> None:
        """注册钩子"""
        if name not in self._hooks:
            self._hooks[name] = []
        self._hooks[name].append(callback)

    def unregister_hook(self, name: str, callback: Callable) -> None:
        """取消注册钩子"""
        if name in self._hooks and callback in self._hooks[name]:
            self._hooks[name].remove(callback)

    def call_hook(self, name: str, *args, **kwargs) -> List[Any]:
        """调用钩子"""
        results = []
        for callback in self._hooks.get(name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"钩子调用失败 {name}: {e}")
        return results


# ============================================================
# 全局插件管理器
# ============================================================

_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def set_plugin_manager(manager: PluginManager) -> None:
    """设置全局插件管理器"""
    global _plugin_manager
    _plugin_manager = manager
