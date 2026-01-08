"""序列化模块测试"""

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.utils.serialization import (
    JsonSerializer,
    MsgPackSerializer,
    PickleSerializer,
    SerializerRegistry,
    deserialize,
    get_serializer,
    serialize,
)


class TestMsgPackSerializer:
    """MsgPack 序列化器测试"""

    def setup_method(self):
        self.serializer = MsgPackSerializer()

    def test_basic_types(self):
        """基本类型"""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }

        serialized = self.serializer.serialize(data)
        result = self.serializer.deserialize(serialized)

        assert result == data

    def test_datetime(self):
        """日期时间"""
        dt = datetime(2024, 1, 15, 10, 30, 0)

        serialized = self.serializer.serialize(dt)
        result = self.serializer.deserialize(serialized)

        assert result == dt

    def test_date(self):
        """日期"""
        d = date(2024, 1, 15)

        serialized = self.serializer.serialize(d)
        result = self.serializer.deserialize(serialized)

        assert result == d

    def test_decimal(self):
        """Decimal"""
        dec = Decimal("123.456")

        serialized = self.serializer.serialize(dec)
        result = self.serializer.deserialize(serialized)

        assert result == dec

    def test_bytes(self):
        """字节"""
        data = b"hello bytes"

        serialized = self.serializer.serialize(data)
        result = self.serializer.deserialize(serialized)

        assert result == data

    def test_numpy_array(self):
        """NumPy 数组"""
        arr = np.array([1, 2, 3, 4, 5])

        serialized = self.serializer.serialize(arr)
        result = self.serializer.deserialize(serialized)

        np.testing.assert_array_equal(result, arr)

    def test_numpy_2d_array(self):
        """NumPy 2D 数组"""
        arr = np.array([[1, 2], [3, 4]])

        serialized = self.serializer.serialize(arr)
        result = self.serializer.deserialize(serialized)

        np.testing.assert_array_equal(result, arr)

    def test_pandas_dataframe(self):
        """Pandas DataFrame"""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })

        serialized = self.serializer.serialize(df)
        result = self.serializer.deserialize(serialized)

        pd.testing.assert_frame_equal(result, df)

    def test_pandas_series(self):
        """Pandas Series"""
        series = pd.Series([1, 2, 3], name="test")

        serialized = self.serializer.serialize(series)
        result = self.serializer.deserialize(serialized)

        pd.testing.assert_series_equal(result, series)

    def test_nested_dict(self):
        """嵌套字典"""
        data = {
            "level1": {
                "level2": {
                    "value": 42,
                },
            },
        }

        serialized = self.serializer.serialize(data)
        result = self.serializer.deserialize(serialized)

        assert result == data

    def test_list_with_mixed_types(self):
        """混合类型列表"""
        data = [1, "hello", 3.14, None, True]

        serialized = self.serializer.serialize(data)
        result = self.serializer.deserialize(serialized)

        assert result == data

    def test_tuple(self):
        """元组"""
        data = (1, 2, 3)

        serialized = self.serializer.serialize(data)
        result = self.serializer.deserialize(serialized)

        assert result == data

    def test_set(self):
        """集合"""
        data = {1, 2, 3}

        serialized = self.serializer.serialize(data)
        result = self.serializer.deserialize(serialized)

        assert result == data

    def test_serialize_to_str(self):
        """序列化为字符串"""
        data = {"key": "value"}

        string = self.serializer.serialize_to_str(data)
        result = self.serializer.deserialize_from_str(string)

        assert result == data


class TestJsonSerializer:
    """JSON 序列化器测试"""

    def setup_method(self):
        self.serializer = JsonSerializer()

    def test_basic_types(self):
        """基本类型"""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }

        serialized = self.serializer.serialize(data)
        result = self.serializer.deserialize(serialized)

        assert result == data

    def test_datetime_serialization(self):
        """日期时间序列化"""
        dt = datetime(2024, 1, 15, 10, 30, 0)

        serialized = self.serializer.serialize(dt)
        result = self.serializer.deserialize(serialized)

        # JSON 序列化为字符串
        assert result == dt.isoformat()

    def test_numpy_array_serialization(self):
        """NumPy 数组序列化"""
        arr = np.array([1, 2, 3])

        serialized = self.serializer.serialize(arr)
        result = self.serializer.deserialize(serialized)

        assert result == [1, 2, 3]

    def test_dataframe_serialization(self):
        """DataFrame 序列化"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        serialized = self.serializer.serialize(df)
        result = self.serializer.deserialize(serialized)

        assert result == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]

    def test_serialize_to_str(self):
        """序列化为 JSON 字符串"""
        data = {"key": "value"}

        string = self.serializer.serialize_to_str(data)
        result = self.serializer.deserialize_from_str(string)

        assert result == data

    def test_indent(self):
        """缩进"""
        serializer = JsonSerializer(indent=2)
        data = {"key": "value"}

        string = serializer.serialize_to_str(data)
        assert "\n" in string


class TestPickleSerializer:
    """Pickle 序列化器测试"""

    def setup_method(self):
        self.serializer = PickleSerializer()

    def test_basic_types(self):
        """基本类型"""
        data = {"string": "hello", "int": 42}

        serialized = self.serializer.serialize(data)
        result = self.serializer.deserialize(serialized)

        assert result == data

    def test_complex_objects(self):
        """复杂对象"""
        # 使用 datetime 作为复杂对象示例
        from datetime import datetime
        obj = datetime(2024, 1, 15, 10, 30, 0)

        serialized = self.serializer.serialize(obj)
        result = self.serializer.deserialize(serialized)

        assert result == obj


class TestSerializerRegistry:
    """序列化器注册表测试"""

    def test_get_default(self):
        """获取默认序列化器"""
        registry = SerializerRegistry()
        serializer = registry.get()

        assert isinstance(serializer, MsgPackSerializer)

    def test_get_by_name(self):
        """按名称获取"""
        registry = SerializerRegistry()

        assert isinstance(registry.get("msgpack"), MsgPackSerializer)
        assert isinstance(registry.get("json"), JsonSerializer)
        assert isinstance(registry.get("pickle"), PickleSerializer)

    def test_unknown_serializer(self):
        """未知序列化器"""
        registry = SerializerRegistry()

        with pytest.raises(KeyError):
            registry.get("unknown")

    def test_set_default(self):
        """设置默认"""
        registry = SerializerRegistry()
        registry.set_default("json")

        assert isinstance(registry.get(), JsonSerializer)

    def test_available(self):
        """可用列表"""
        registry = SerializerRegistry()
        available = registry.available

        assert "msgpack" in available
        assert "json" in available
        assert "pickle" in available


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_serialize_deserialize(self):
        """序列化和反序列化"""
        data = {"key": "value"}

        serialized = serialize(data)
        result = deserialize(serialized)

        assert result == data

    def test_serialize_with_format(self):
        """指定格式序列化"""
        data = {"key": "value"}

        serialized = serialize(data, format="json")
        result = deserialize(serialized, format="json")

        assert result == data

    def test_get_serializer(self):
        """获取序列化器"""
        serializer = get_serializer("msgpack")
        assert isinstance(serializer, MsgPackSerializer)
