"""Sphinx 文档配置"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.abspath(".."))

# 项目信息
project = "FloatShare"
copyright = "2024, FloatShare Team"
author = "FloatShare Team"
version = "1.0"
release = "1.0.0"

# 扩展
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

# Napoleon 设置 (支持 Google/NumPy 风格 docstring)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# autodoc 设置
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"

# intersphinx 设置
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# 模板路径
templates_path = ["_templates"]

# 排除的模式
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML 主题
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# HTML 选项
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# TODO 扩展设置
todo_include_todos = True
