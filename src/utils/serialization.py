"""序列化模块

提供高效的序列化方案：
- msgpack: 高性能二进制序列化
- JSON: 通用文本序列化
- 自动类型转换
"""

import json
import pickle
from abc import ABC, abstractmethod
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Optional, Type, TypeVar, Union

import msgpack
import numpy as np
import pandas as pd

T = TypeVar("T")


# ============================================================
# 序列化器接口
# ============================================================


class Serializer(ABC):
    """序列化器基类"""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """序列化对象为字节"""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """反序列化字节为对象"""
        pass

    def serialize_to_str(self, obj: Any) -> str:
        """序列化为字符串 (Base64 编码)"""
        import base64
        return base64.b64encode(self.serialize(obj)).decode("utf-8")

    def deserialize_from_str(self, data: str) -> Any:
        """从字符串反序列化"""
        import base64
        return self.deserialize(base64.b64decode(data.encode("utf-8")))


# ============================================================
# MsgPack 序列化器
# ============================================================


class MsgPackSerializer(Serializer):
    """
    MsgPack 序列化器

    性能比 pickle 更好，比 JSON 更快更紧凑
    """

    def __init__(self, use_bin_type: bool = True):
        self.use_bin_type = use_bin_type

    def serialize(self, obj: Any) -> bytes:
        """序列化"""
        return msgpack.packb(
            self._encode(obj),
            use_bin_type=self.use_bin_type,
        )

    def deserialize(self, data: bytes) -> Any:
        """反序列化"""
        return self._decode(
            msgpack.unpackb(data, raw=False)
        )

    def _encode(self, obj: Any) -> Any:
        """编码对象为可序列化格式"""
        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, bytes):
            return {"__type__": "bytes", "data": list(obj)}

        if isinstance(obj, datetime):
            return {"__type__": "datetime", "data": obj.isoformat()}

        if isinstance(obj, date):
            return {"__type__": "date", "data": obj.isoformat()}

        if isinstance(obj, Decimal):
            return {"__type__": "decimal", "data": str(obj)}

        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
            }

        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "dataframe",
                "data": obj.to_dict(orient="records"),
                "columns": list(obj.columns),
                "index": list(obj.index),
            }

        if isinstance(obj, pd.Series):
            return {
                "__type__": "series",
                "data": obj.tolist(),
                "name": obj.name,
                "index": list(obj.index),
            }

        if isinstance(obj, dict):
            return {k: self._encode(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            result = [self._encode(item) for item in obj]
            if isinstance(obj, tuple):
                return {"__type__": "tuple", "data": result}
            return result

        if isinstance(obj, set):
            return {"__type__": "set", "data": [self._encode(item) for item in obj]}

        # 尝试使用 __dict__
        if hasattr(obj, "__dict__"):
            return {
                "__type__": "object",
                "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                "data": self._encode(obj.__dict__),
            }

        raise TypeError(f"无法序列化类型: {type(obj)}")

    def _decode(self, obj: Any) -> Any:
        """解码序列化格式为对象"""
        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, dict):
            type_marker = obj.get("__type__")

            if type_marker == "bytes":
                return bytes(obj["data"])

            if type_marker == "datetime":
                return datetime.fromisoformat(obj["data"])

            if type_marker == "date":
                return date.fromisoformat(obj["data"])

            if type_marker == "decimal":
                return Decimal(obj["data"])

            if type_marker == "ndarray":
                return np.array(obj["data"], dtype=obj["dtype"])

            if type_marker == "dataframe":
                df = pd.DataFrame(obj["data"])
                if obj["columns"]:
                    df = df[obj["columns"]]
                return df

            if type_marker == "series":
                return pd.Series(
                    obj["data"],
                    name=obj["name"],
                    index=obj["index"],
                )

            if type_marker == "tuple":
                return tuple(self._decode(item) for item in obj["data"])

            if type_marker == "set":
                return set(self._decode(item) for item in obj["data"])

            if type_marker == "object":
                # 简单返回字典，不尝试重建对象
                return self._decode(obj["data"])

            return {k: self._decode(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._decode(item) for item in obj]

        return obj


# ============================================================
# JSON 序列化器
# ============================================================


class JsonSerializer(Serializer):
    """JSON 序列化器"""

    def __init__(self, indent: Optional[int] = None):
        self.indent = indent

    def serialize(self, obj: Any) -> bytes:
        """序列化"""
        return json.dumps(
            obj,
            cls=ExtendedJsonEncoder,
            indent=self.indent,
            ensure_ascii=False,
        ).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        """反序列化"""
        return json.loads(data.decode("utf-8"))

    def serialize_to_str(self, obj: Any) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(
            obj,
            cls=ExtendedJsonEncoder,
            indent=self.indent,
            ensure_ascii=False,
        )

    def deserialize_from_str(self, data: str) -> Any:
        """从 JSON 字符串反序列化"""
        return json.loads(data)


class ExtendedJsonEncoder(json.JSONEncoder):
    """扩展 JSON 编码器"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()

        if isinstance(obj, date):
            return obj.isoformat()

        if isinstance(obj, Decimal):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")

        if isinstance(obj, pd.Series):
            return obj.tolist()

        if isinstance(obj, (set, frozenset)):
            return list(obj)

        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)


# ============================================================
# Pickle 序列化器 (保留兼容性)
# ============================================================


class PickleSerializer(Serializer):
    """
    Pickle 序列化器

    注意: 不安全，不建议用于不信任的数据
    """

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


# ============================================================
# 序列化器注册表
# ============================================================


class SerializerRegistry:
    """序列化器注册表"""

    def __init__(self):
        self._serializers: Dict[str, Serializer] = {
            "msgpack": MsgPackSerializer(),
            "json": JsonSerializer(),
            "pickle": PickleSerializer(),
        }
        self._default = "msgpack"

    def register(self, name: str, serializer: Serializer) -> None:
        """注册序列化器"""
        self._serializers[name] = serializer

    def get(self, name: Optional[str] = None) -> Serializer:
        """获取序列化器"""
        name = name or self._default
        if name not in self._serializers:
            raise KeyError(f"未知的序列化器: {name}")
        return self._serializers[name]

    def set_default(self, name: str) -> None:
        """设置默认序列化器"""
        if name not in self._serializers:
            raise KeyError(f"未知的序列化器: {name}")
        self._default = name

    @property
    def available(self) -> list:
        """可用的序列化器列表"""
        return list(self._serializers.keys())


# 全局注册表
_registry = SerializerRegistry()


def get_serializer(name: Optional[str] = None) -> Serializer:
    """获取序列化器"""
    return _registry.get(name)


def serialize(obj: Any, format: str = "msgpack") -> bytes:
    """序列化对象"""
    return get_serializer(format).serialize(obj)


def deserialize(data: bytes, format: str = "msgpack") -> Any:
    """反序列化数据"""
    return get_serializer(format).deserialize(data)
