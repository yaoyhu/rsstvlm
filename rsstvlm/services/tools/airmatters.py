"""
该模块封装了 Air Matters API 的所有接口, 为 Agent 提供空气质量数据查询能力。
支持实时空气质量、历史数据、预报数据、地点搜索等功能。

API 文档: https://api.air-matters.app

主要功能:
- 实时空气质量查询 (current_air_condition)
- 历史空气质量查询 (history_air_condition)
- 空气质量预报 (aqi_forecast)
- 地点搜索与管理 (place_search, sub_places, nearby_place)
- 区域空气质量地图 (map, heatmap)
- 批量空气质量查询 (batch_air_condition)
- 附近空气质量查询 (nearby_air_condition)
- AQI 标准查询 (standard)

Agent 使用指南:
1. 首先使用 place_search() 或 nearby_place() 获取 place_id
2. 使用 place_id 调用其他接口获取空气质量数据
3. 根据用户语言偏好设置 lang 参数 ("en" / "zh-Hans")
4. 根据地区选择合适的 AQI 标准 ("aqi_us" / "aqi_cn" / "caqi")
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, ClassVar, Literal

import requests

from rsstvlm.utils import AM_API_KEY


class AirMattersError(Exception):
    """Air Matters API 错误基类"""

    pass


class APIRequestError(AirMattersError):
    """API 请求错误"""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(
            f"API请求失败: {message}"
            + (f" (状态码: {status_code})" if status_code else "")
        )


class PlaceNotFoundError(AirMattersError):
    """地点未找到错误"""

    def __init__(self, search_term: str):
        super().__init__(f"未找到地点: {search_term}")


class AirMatters:
    """
    AirMatters class implements all API endpoints provided by Air Matters,
    enabling Agents to query air quality data.

    Base URL: https://api.air-matters.app

    Attributes:
        api_key: AirMatters API KEY
        lang: "en" | "zh-Hans", default: "zh-Hans"
        standard: "aqi_us" | "aqi_cn" | "caqi", default: "aqi_cn"
    """

    BASE_URL = "https://api.air-matters.app"

    def __init__(
        self,
        api_key: str | None = None,
        lang: str = "en",
        standard: str = "aqi_us",
        timeout: int = 30,
    ):
        """
        初始化 Air Matters API 客户端

        Args:
            api_key: API 访问令牌，如果不提供则使用全局配置
            lang: 默认语言代码
                - "en": 英语 (默认)
                - "zh-Hans": 简体中文
                - "zh-Hant": 繁体中文
            standard: 默认 AQI 计算标准
                - "aqi_us": 美国 EPA 标准 (默认，国际通用)
                - "aqi_cn": 中国国家标准 (适用于中国大陆)
                - "caqi": 欧洲 CAQI 标准 (适用于欧洲)
            timeout: 请求超时时间 (秒)
        """
        self.api_key = api_key or AM_API_KEY
        self.lang = lang
        self.standard = standard
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})

    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        method: Literal["GET", "POST"] = "GET",
    ) -> dict[str, Any]:
        """
        发送 API 请求的内部方法

        Args:
            endpoint: API 端点路径 (不含基础 URL)
            params: 请求参数
            method: HTTP 方法 ("GET" 或 "POST")

        Returns:
            API 响应的 JSON 数据

        Raises:
            APIRequestError: 当 API 请求失败时
        """
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            if method == "GET":
                response = self.session.get(
                    url, params=params, timeout=self.timeout
                )
            elif method == "POST":
                response = self.session.post(
                    url, json=params, timeout=self.timeout
                )
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            raise APIRequestError(
                str(e), e.response.status_code if e.response else None
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise APIRequestError("网络连接失败,请检查网络设置") from e
        except requests.exceptions.Timeout as e:
            raise APIRequestError(f"请求超时 ({self.timeout}秒)") from e
        except requests.exceptions.RequestException as e:
            raise APIRequestError(str(e)) from e

    # ==================== 地点搜索与管理 API ====================

    def place_search(
        self,
        content: str,
        ancestor: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        """
        搜索地点 - 根据名称搜索地点并获取 place_id

        这是使用其他 API 的前置步骤，Agent 需要先通过此接口获取 place_id，
        然后才能查询该地点的空气质量数据。

        Agent 使用场景:
        - 用户提到某个城市/地点名称时，首先调用此接口获取 place_id
        - 用户问 "北京的空气质量怎么样" → 先搜索 "北京" 获取 place_id

        Args:
            content: 搜索关键词 (城市名、地点名)
                - 建议使用英文获取更好的搜索结果
                - 中文也支持，如 "北京"、"上海"
            ancestor: 上级地点名称，用于精确定位
                - 例如: 搜索 "朝阳区"，ancestor="北京" 可精确定位北京朝阳区
            lang: 响应语言
                - "en": 英语
                - "zh-Hans": 简体中文

        Returns:
            {
                "places": [
                    {
                        "lat": 39.906214,         # 纬度
                        "lon": 116.3977,          # 经度
                        "name": "Beijing",        # 地点名称
                        "type": "locality"        # 类型: country/administrativearea/locality/suburb/station
                        "place_id": "ec8399ca",   # 地点唯一标识，用于其他 API
                    }
                ]
            }

        Raises:
            APIRequestError: API 请求失败
            PlaceNotFoundError: 未找到匹配的地点

        Example:
            >>> am = AirMatters()
            >>> result = am.place_search("Beijing")
            >>> place_id = result["places"][0]["place_id"]
            >>> print(f"北京的 place_id: {place_id}")
        """
        params = {
            "content": content,
            "lang": lang or self.lang,
        }
        if ancestor is not None:
            params["ancestor"] = ancestor

        result = self._make_request("place_search", params)

        if not result.get("places"):
            raise PlaceNotFoundError(content)

        return result

    def nearby_place(
        self,
        lat: float,
        lon: float,
        lang: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        获取附近地点 - 根据经纬度坐标获取最近的监测点

        默认搜索半径约 30km，返回最近的有数据的监测节点。
        优先返回 suburb 级别节点，如果不存在则返回 locality 级别，以此类推。

        Agent 使用场景:
        - 用户提供经纬度坐标时使用
        - 用户问 "我这里的空气质量" (需要获取用户位置坐标)
        - 用户上传带有地理位置信息的数据时

        Args:
            lat: 纬度坐标
                - 范围: -90 到 90
                - 例如: 39.9 (北京)
            lon: 经度坐标
                - 范围: -180 到 180
                - 例如: 116.4 (北京)
            lang: 响应语言
                - "en": 英语
                - "zh-Hans": 简体中文
            threshold: 搜索半径限制 (单位: 公里)
                - 默认约 30km
                - 设置更小的值可获取更精确的附近地点

        Returns:
            {
                "lat": 36.7289127,          # 纬度
                "lon": -121.2788708,        # 经度
                "name": "Paicines",         # 地点名称
                "type": "locality"          # 地点类型
                "place_id": "b5f0a667",     # 地点唯一标识
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 查找北京天安门附近的监测点
            >>> result = am.nearby_place(lat=39.9087, lon=116.3975)
            >>> print(f"附近监测点: {result['name']}")
        """
        params = {
            "lat": lat,
            "lon": lon,
            "lang": lang or self.lang,
        }
        if threshold is not None:
            params["threshold"] = threshold

        return self._make_request("nearby_place", params)

    def sub_places(
        self,
        place_id: str,
        lang: str | None = None,
    ) -> dict[str, Any]:
        """
        获取子地点 - 获取指定地点的下级区域/监测站列表

        用于深入查询某个地点下的具体监测站或子区域。

        Agent 使用场景:
        - 用户想了解某个城市各区的空气质量差异
        - 需要获取更精细的监测数据时
        - e.g. 用户问 "北京各区的空气质量对比"

        Args:
            place_id: 父级地点的唯一标识
                - 通过 place_search() 或 nearby_place() 获取
            lang: 响应语言
                - "en": 英语
                - "zh-Hans": 简体中文

        Returns:
            {
                "places": [
                    {
                        "lat": 36.485001,         # 纬度
                        "lon": -121.155998,       # 经度
                        "name": "Pinnacles NM",   # 子地点名称
                        "type": "station"         # 类型 (通常是 station 监测站)
                        "place_id": "4d7e2db4",   # 子地点标识
                    }
                ]
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 获取北京的所有子区域/监测站
            >>> result = am.sub_places("ec8399ca")  # 北京的 place_id
            >>> for place in result["places"]:
            ...     print(f"{place['name']}: {place['place_id']}")
        """
        params = {
            "place_id": place_id,
            "lang": lang or self.lang,
        }
        return self._make_request("sub_places", params)

    # ==================== 空气质量查询 API ====================

    def current_air_condition(
        self,
        place_id: str,
        lang: str | None = None,
        standard: str | None = None,
    ) -> dict[str, Any]:
        """
        查询指定地点的最新空气质量数据，返回当前的 AQI 指数和各项污染物浓度。

        Agent 使用场景:
        - 用户询问某地的当前空气质量
        - 需要获取实时污染物浓度数据

        Args:
            place_id: 地点唯一标识
                - 通过 place_search() 或 nearby_place() 获取
            lang: 响应语言
                - "en": 英语 (返回 "Good", "Moderate" 等)
                - "zh-Hans": 简体中文 (返回 "优", "良" 等)
            standard: AQI 计算标准
                - "aqi_us": 美国 EPA 标准 (国际通用)
                - "aqi_cn": 中国国家标准 (适用于中国)

        Returns:
            {
                "latest": {
                    "readings": [
                        {
                            "name": "AQI (US)",      # 指标名称
                            "kind": "aqi",           # 指标类型标识
                            "color": "#31cd31",      # 等级颜色 (绿色=好)
                            "level": "Good",         # 污染等级
                            "value": "45"            # 数值
                        },
                        {
                            "name": "PM2.5",
                            "kind": "pm25",
                            "color": "#31cd31",
                            "level": "Good",
                            "value": "7",
                            "unit": "μg/m³"          # 单位
                        },
                        // ... 其他污染物
                    ],
                    "update_time": "2022-07-06 06:20:00"  # 数据更新时间
                }
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 查询北京当前空气质量
            >>> result = am.current_air_condition("ec8399ca", lang="zh-Hans", standard="aqi_cn")
        """
        params = {
            "place_id": place_id,
            "lang": lang or self.lang,
            "standard": standard or self.standard,
        }
        return self._make_request("current_air_condition", params)

    def history_air_condition(
        self,
        place_id: str,
        hourly_start: str | date | None = None,
        hourly_end: str | date | None = None,
        daily_start: str | date | None = None,
        daily_end: str | date | None = None,
        items: list[str] | str | None = None,
        lang: str | None = None,
        standard: str | None = None,
    ) -> dict[str, Any]:
        """
        获取历史空气质量 - 查询指定地点的历史空气质量数据

        支持按小时或按天查询历史数据，可同时获取两种粒度的数据。

        Agent 使用场景:
        - 用户询问过去某段时间的空气质量
        - 用户问 "上周北京的空气质量怎么样"
        - 需要分析空气质量变化趋势
        - 环境数据分析和报告生成

        注意事项:
        - hourly 数据更详细但数据量大
        - daily 数据适合长时间范围分析
        - 可以同时请求 hourly 和 daily 数据

        Args:
            place_id: 地点唯一标识
                - 通过 place_search() 或 nearby_place() 获取
            hourly_start: 小时数据开始日期
                - 格式: "YYYY-MM-DD" 或 date 对象
                - 例如: "2024-01-01"
            hourly_end: 小时数据结束日期
                - 格式: "YYYY-MM-DD" 或 date 对象
            daily_start: 日数据开始日期
                - 格式: "YYYY-MM-DD" 或 date 对象
            daily_end: 日数据结束日期
                - 格式: "YYYY-MM-DD" 或 date 对象
            items: 需要查询的污染物类型
                - 可选值: "aqi", "pm25", "pm10", "o3", "no2", "so2", "co"
                - 可传入列表如 ["aqi", "pm25"] 或逗号分隔字符串 "aqi,pm25"
                - 不传则默认只返回 aqi
            lang: 响应语言
            standard: AQI 计算标准

        Returns:
            {
                "history": [
                    {
                        "type": "index",
                        "kind": "aqi_us",          # 污染物类型
                        "interval": "hourly",      # 时间粒度: hourly/daily
                        "name": "AQI",
                        "data": [
                            {
                                "color": "#e02d1c",
                                "time": "2022-10-01 00:00:00",  # 时间点
                                "value": "161"                   # AQI 值
                            },
                            // ... 更多数据点
                        ]
                    },
                    {
                        "type": "index",
                        "kind": "aqi_us",
                        "interval": "daily",       # 日数据
                        "name": "AQI",
                        "data": [...]
                    }
                ]
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 查询北京2024年1月1日-5日的历史数据
            >>> result = am.history_air_condition(
            ...     place_id="ec8399ca",
            ...     daily_start="2024-01-01",
            ...     daily_end="2024-01-05",
            ...     items=["aqi", "pm25"],
            ...     lang="zh-Hans",
            ...     standard="aqi_cn"
            ... )
        """
        params = {
            "place_id": place_id,
            "lang": lang or self.lang,
            "standard": standard or self.standard,
        }

        # 处理日期参数
        if hourly_start:
            params["hourly_start"] = (
                str(hourly_start)
                if isinstance(hourly_start, date)
                else hourly_start
            )
        if hourly_end:
            params["hourly_end"] = (
                str(hourly_end) if isinstance(hourly_end, date) else hourly_end
            )
        if daily_start:
            params["daily_start"] = (
                str(daily_start)
                if isinstance(daily_start, date)
                else daily_start
            )
        if daily_end:
            params["daily_end"] = (
                str(daily_end) if isinstance(daily_end, date) else daily_end
            )

        # 处理 items 参数
        if items:
            if isinstance(items, list):
                params["items"] = ",".join(items)
            else:
                params["items"] = items

        return self._make_request("history_air_condition", params)

    def nearby_air_condition(
        self,
        lat: float,
        lon: float,
        lang: str | None = None,
        standard: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        获取附近空气质量 - 根据坐标直接获取最近监测点的空气质量

        这是 nearby_place + current_air_condition 的组合接口，
        一次调用即可获取附近地点及其当前空气质量。

        Agent 使用场景:
        - 用户提供坐标，需要快速获取空气质量
        - 不需要先查询 place_id 的场景
        - GPS 定位查询空气质量

        Args:
            lat: 纬度坐标 (-90 到 90)
            lon: 经度坐标 (-180 到 180)
            lang: 响应语言
            standard: AQI 计算标准
            threshold: 搜索半径限制 (单位: 公里)

        Returns:
            {
                "place": {
                    "place_id": "b5f0a667",
                    "name": "Paicines",
                    "lat": 36.7289127,
                    "lon": -121.2788708,
                    "type": "locality"
                },
                "latest": {
                    "readings": [...],           # 空气质量读数
                    "update_time": "2022-07-06 06:20:00"
                }
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 直接通过坐标查询空气质量
            >>> result = am.nearby_air_condition(lat=39.9087, lon=116.3975)
            >>> print(f"地点: {result['place']['name']}")
            >>> print(f"AQI 数据: {result['latest']['readings']}")
        """
        params = {
            "lat": lat,
            "lon": lon,
            "lang": lang or self.lang,
            "standard": standard or self.standard,
        }
        if threshold is not None:
            params["threshold"] = threshold

        return self._make_request("nearby_air_condition", params)

    def aqi_forecast(
        self,
        place_id: str,
        lang: str | None = None,
        standard: str | None = None,
    ) -> dict[str, Any]:
        """
        获取空气质量预报 - 查询指定地点未来几天的 AQI 预报

        返回未来约 7 天的每日 AQI 预报数据。

        Agent 使用场景:
        - 用户询问未来空气质量
        - 用户问 "明天北京空气质量怎么样"
        - 出行计划参考
        - 空气质量预警

        Args:
            place_id: 地点唯一标识
                - 通过 place_search() 或 nearby_place() 获取
            lang: 响应语言
            standard: AQI 计算标准

        Returns:
            {
                "forecast": [
                    {
                        "type": "index",
                        "kind": "aqi",
                        "interval": "daily",       # 每日预报
                        "unit": "",
                        "name": "AQI",
                        "data": [
                            {
                                "color": "#d9d726",           # 等级颜色
                                "level": "Moderate",          # 污染等级
                                "time": "2022-07-06 00:00:00", # 预报日期
                                "value": "35~55"              # AQI 预测范围
                            },
                            {
                                "color": "#31cd31",
                                "level": "Good",
                                "time": "2022-07-07 00:00:00",
                                "value": "15~35"
                            },
                            // ... 更多预报数据
                        ]
                    }
                ]
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 查询北京空气质量预报
            >>> result = am.aqi_forecast("ec8399ca", lang="zh-Hans")
            >>> for day in result["forecast"][0]["data"]:
            ...     print(f"{day['time']}: {day['level']} ({day['value']})")
        """
        params = {
            "place_id": place_id,
            "lang": lang or self.lang,
            "standard": standard or self.standard,
        }
        return self._make_request("aqi_forecast", params)

    def batch_air_condition(
        self,
        place_ids: list[str],
        lang: str | None = None,
        standard: str | None = None,
    ) -> dict[str, Any]:
        """
        批量获取空气质量 - 一次请求获取多个地点的当前空气质量

        最多支持同时查询 200 个地点。

        Agent 使用场景:
        - 需要对比多个城市的空气质量
        - 用户问 "北京、上海、广州的空气质量对比"
        - 区域空气质量分析
        - 批量数据采集

        注意: 此接口使用 POST 方法

        Args:
            place_ids: 地点 ID 列表
                - 最多 200 个
                - 通过 place_search() 获取各地点 ID
            lang: 响应语言
            standard: AQI 计算标准

        Returns:
            {
                "places": [
                    {
                        "place": {
                            "place_id": "ec8399ca",
                            "name": "Beijing",
                            "lat": 39.906214,
                            "lon": 116.3977,
                            "type": "locality"
                        },
                        "latest": {
                            "readings": [...],
                            "update_time": "2022-07-07 02:19:00"
                        }
                    },
                    // ... 更多地点
                ]
            }

        Raises:
            APIRequestError: API 请求失败
            ValueError: place_ids 数量超过 200

        Example:
            >>> am = AirMatters()
            >>> # 批量查询北京、上海、广州
            >>> result = am.batch_air_condition(
            ...     place_ids=["ec8399ca", "shanghai_id", "guangzhou_id"],
            ...     lang="zh-Hans",
            ...     standard="aqi_cn"
            ... )
        """
        if len(place_ids) > 200:
            raise ValueError(
                f"place_ids 数量不能超过 200，当前: {len(place_ids)}"
            )

        params = {
            "places": place_ids,
            "lang": lang or self.lang,
            "standard": standard or self.standard,
        }
        return self._make_request("batch_air_condition", params, method="POST")

    # ==================== 区域空气质量地图 API ====================

    def map(
        self,
        north_east_lat: float,
        north_east_lon: float,
        south_west_lat: float,
        south_west_lon: float,
        lang: str | None = None,
        standard: str | None = None,
    ) -> dict[str, Any]:
        """
        获取区域空气质量地图数据 - 查询指定矩形区域内所有监测点的空气质量

        返回结果基于行政区划级别，优先显示国家级别，
        然后是省/州级别，然后是城市级别等。
        结果数量不超过 200 个。

        Agent 使用场景:
        - 用户需要查看某个区域的空气质量分布
        - 用户问 "华北地区空气质量情况"
        - 区域空气质量对比分析
        - 生成空气质量分布图

        Args:
            north_east_lat: 东北角纬度 (区域右上角)
            north_east_lon: 东北角经度
            south_west_lat: 西南角纬度 (区域左下角)
            south_west_lon: 西南角经度
            lang: 响应语言
            standard: AQI 计算标准

        Returns:
            {
                "map": [
                    {
                        "place": {
                            "place_id": "178d7bac",
                            "name": "Willoughby",
                            "lat": -33.8071059,
                            "lon": 151.1993737,
                            "type": "locality"
                        },
                        "latest": {
                            "readings": [
                                {
                                    "name": "AQI",
                                    "type": "index",
                                    "kind": "aqi",
                                    "color": "#31cd31",
                                    "level": "Good",
                                    "value": "20"
                                },
                                // ... 其他污染物
                            ]
                        }
                    },
                    // ... 更多地点 (最多 200 个)
                ]
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 查询华北地区 (大致范围)
            >>> result = am.map(
            ...     north_east_lat=42.0,
            ...     north_east_lon=120.0,
            ...     south_west_lat=35.0,
            ...     south_west_lon=110.0,
            ...     lang="zh-Hans",
            ...     standard="aqi_cn"
            ... )
        """
        params = {
            "north_east_lat": north_east_lat,
            "north_east_lon": north_east_lon,
            "south_west_lat": south_west_lat,
            "south_west_lon": south_west_lon,
            "lang": lang or self.lang,
            "standard": standard or self.standard,
        }
        return self._make_request("map", params)

    def heatmap(
        self,
        north_east_lat: float,
        north_east_lon: float,
        south_west_lat: float,
        south_west_lon: float,
        standard: str | None = None,
    ) -> dict[str, Any]:
        """
        获取空气质量热力图 - 获取指定区域的空气质量热力图图片

        返回区域被切分后的图片 URL 列表和对应的坐标范围。

        Agent 使用场景:
        - 需要可视化展示空气质量分布
        - 生成空气质量报告配图
        - 用户需要直观的空气质量分布图

        Args:
            north_east_lat: 东北角纬度
            north_east_lon: 东北角经度
            south_west_lat: 西南角纬度
            south_west_lon: 西南角经度
            standard: AQI 计算标准 (影响颜色渲染)

        Returns:
            {
                "pieces": [
                    {
                        "image_url": "https://heatmap-cn.air-matters.com/images/xxx.png",
                        "north_west": {"lat": 74.64, "lon": -35.98},
                        "north_east": {"lat": 74.64, "lon": 35.98},
                        "south_east": {"lat": 35.70, "lon": 35.98},
                        "south_west": {"lat": 35.70, "lon": -35.98}
                    },
                    // ... 更多图片块
                ]
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 获取中国区域热力图
            >>> result = am.heatmap(
            ...     north_east_lat=53.5,
            ...     north_east_lon=135.0,
            ...     south_west_lat=18.0,
            ...     south_west_lon=73.5,
            ...     standard="aqi_cn"
            ... )
        """
        params = {
            "north_east_lat": north_east_lat,
            "north_east_lon": north_east_lon,
            "south_west_lat": south_west_lat,
            "south_west_lon": south_west_lon,
            "standard": standard or self.standard,
        }
        return self._make_request("heatmap", params)

    def get_standard(
        self,
        standard: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        """
        获取 AQI 标准定义 - 查询 AQI 等级的阈值、颜色和描述

        返回指定 AQI 标准的断点值、颜色映射和等级描述。

        Agent 使用场景:
        - 需要解释 AQI 数值的含义
        - 需要了解各污染物浓度的等级划分
        - 生成空气质量报告时参考标准定义
        - 自定义空气质量等级判断

        Args:
            standard: AQI 标准类型
                - "aqi_us": 美国 EPA 标准
                - "aqi_cn": 中国国家标准
                - "caqi": 欧洲 CAQI 标准
                - 无效或为空时默认返回 aqi_us
            lang: 响应语言

        Returns:
            {
                "break_point": {
                    "aqi_us": {
                        "aqi": [0, 50, 100, 150, 200, 300, 400, 500],  # AQI 断点
                        "pm25": [0, 12, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4],  # PM2.5 断点 (μg/m³)
                        "pm10": [0, 54, 154, 254, 354, 424, 504, 604],
                        "o3": [0, 126.4, 160.7, 351.4, 437.1, 865.7, 1080, 1294.3],
                        "no2": [0, 108, 205, 739, 1332, 2564, 3386, 4207],
                        "so2": [0, 100, 214, 528, 868, 1725, 2297, 2868],
                        "co": [0, 5500, 11750, 15500, 19250, 38000, 50500, 63000]
                    }
                },
                "color": {
                    "aqi_us": ["#31cd31", "#d9d726", "#e88019", "#e02d1c", "#af32ba", "#950c32", "#950c32", "#333333"]
                },
                "levels": {
                    "aqi_us": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous", "Hazardous", "Beyond Index"]
                }
            }

        Raises:
            APIRequestError: API 请求失败

        Example:
            >>> am = AirMatters()
            >>> # 获取中国 AQI 标准
            >>> result = am.get_standard(standard="aqi_cn", lang="zh-Hans")
            >>> print("AQI 等级:", result["levels"]["aqi_cn"])
        """
        params = {
            "standard": standard or self.standard,
            "lang": lang or self.lang,
        }
        return self._make_request("standard", params)


def run_tests():
    """运行所有 API 功能测试"""
    am = AirMatters()

    print("=" * 60)
    print("Air Matters API 功能测试")
    print("=" * 60)

    # 用于存储测试结果
    test_results = []

    def run_test(name: str, func):
        """执行单个测试并记录结果"""
        print(f"\n{'─' * 40}")
        print(f"测试: {name}")
        print("─" * 40)
        try:
            result = func()
            print("✅ 成功")
            print(f"响应: {result}")
            test_results.append((name, True, None))
            return result
        except Exception as e:
            print(f"❌ 失败: {e}")
            test_results.append((name, False, str(e)))
            return None

    place_result = run_test(
        "1. 地点搜索 (place_search)",
        lambda: am.place_search(content="Hefei", lang="en"),
    )

    # 获取 place_id 供后续测试使用
    place_id = None
    if place_result and place_result.get("places"):
        place_id = place_result["places"][0]["place_id"]
        print(f"📍 获取到 place_id: {place_id}")

    if place_id:
        run_test(
            "2. 获取子地点 (sub_places)",
            lambda: am.sub_places(place_id=place_id, lang="en"),
        )

    run_test(
        "3. 附近地点搜索 (nearby_place)",
        lambda: am.nearby_place(lat=39.9087, lon=116.3975, lang="en"),
    )

    if place_id:
        run_test(
            "4. 实时空气质量 (current_air_condition)",
            lambda: am.current_air_condition(
                place_id=place_id, lang="zh-Hans", standard="aqi_cn"
            ),
        )

    if place_id:
        run_test(
            "5. 历史空气质量 (history_air_condition)",
            lambda: am.history_air_condition(
                place_id=place_id,
                daily_start="2026-01-01",
                daily_end="2026-01-05",
                items=["aqi", "pm25"],
                lang="zh-Hans",
                standard="aqi_cn",
            ),
        )

    if place_id:
        run_test(
            "6. 空气质量预报 (aqi_forecast)",
            lambda: am.aqi_forecast(
                place_id=place_id, lang="zh-Hans", standard="aqi_cn"
            ),
        )

    run_test(
        "7. 附近空气质量 (nearby_air_condition)",
        lambda: am.nearby_air_condition(
            lat=39.9087, lon=116.3975, lang="zh-Hans", standard="aqi_cn"
        ),
    )

    if place_id:
        # 搜索上海获取第二个 place_id
        shanghai_result = am.place_search(content="Shanghai", lang="en")
        shanghai_id = (
            shanghai_result["places"][0]["place_id"]
            if shanghai_result.get("places")
            else None
        )

        if shanghai_id:
            run_test(
                "8. 批量空气质量查询 (batch_air_condition)",
                lambda: am.batch_air_condition(
                    place_ids=[place_id, shanghai_id],
                    lang="zh-Hans",
                    standard="aqi_cn",
                ),
            )

    run_test(
        "9. 区域空气质量地图 (map)",
        lambda: am.map(
            north_east_lat=42.0,
            north_east_lon=120.0,
            south_west_lat=35.0,
            south_west_lon=110.0,
            lang="zh-Hans",
            standard="aqi_cn",
        ),
    )

    run_test(
        "10. 空气质量热力图 (heatmap)",
        lambda: am.heatmap(
            north_east_lat=42.0,
            north_east_lon=120.0,
            south_west_lat=35.0,
            south_west_lon=110.0,
            standard="aqi_cn",
        ),
    )

    run_test(
        "11. AQI 标准查询 (get_standard)",
        lambda: am.get_standard(standard="aqi_cn", lang="zh-Hans"),
    )

    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, success, _ in test_results if success)
    failed = sum(1 for _, success, _ in test_results if not success)

    print(f"\n总计: {len(test_results)} 个测试")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")

    if failed > 0:
        print("\n失败的测试:")
        for name, success, error in test_results:
            if not success:
                print(f"  - {name}: {error}")

    return test_results


if __name__ == "__main__":
    run_tests()
