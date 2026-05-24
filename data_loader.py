"""data_loader.py - Robust data loading with auto schema detection."""
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger("data_loader")


class SchemaDetector:
    """Auto-detect column meanings from any column names."""

    ORDER_ID_PATTERNS = ["订单单号", "订单号", "order_id", "订单编号", "order_no", "单号"]
    CUSTOMER_ID_PATTERNS = ["货主编码", "客户编号", "customer_id", "客户编码", "cust_id", "customer_code", "货主"]
    PRODUCT_ID_PATTERNS = ["商品编码", "产品编号", "product_id", "产品编码", "prod_id", "sku", "item_id"]
    PRODUCT_NAME_PATTERNS = ["商品名称", "产品名称", "product_name", "prod_name", "item_name", "sku_name", "名称"]
    QTY_PATTERNS = ["求和项:预计发货数量-EA", "预计发货数量EA", "quantity", "数量", "qty", "发货数量", "需求数量"]
    DATE_PATTERNS = ["创建时间", "订单日期", "date", "order_date", "时间", "日期", "dt", "timestamp"]
    WAREHOUSE_PATTERNS = ["仓库", "warehouse", "仓", "库房"]
    STORE_PATTERNS = ["收货门店", "门店", "store", "店铺", "收货方"]
    PROVINCE_PATTERNS = ["省市区", "省份", "province", "省"]

    @classmethod
    def find_col(cls, columns: List[str], patterns: List[str]) -> Optional[str]:
        for p in patterns:
            for c in columns:
                if p.lower() in c.lower():
                    return c
        return None

    @classmethod
    def find_all_cols(cls, columns: List[str], patterns: List[str]) -> List[str]:
        matches = []
        for p in patterns:
            for c in columns:
                if p.lower() in c.lower() and c not in matches:
                    matches.append(c)
        return matches


def load_excel_auto(file_path: str) -> pd.DataFrame:
    """Load Excel, auto-detect sheets, auto-detect columns."""
    fname = os.path.basename(file_path)
    logger.info("Loading %s", fname)

    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    logger.info("  Sheets found: %s", sheet_names)

    # Find order details sheet
    detail_sheet = None
    for name in sheet_names:
        if any(k in name for k in ["明细", "detail", "line", "item", "product"]):
            detail_sheet = name
            break
    if detail_sheet is None:
        detail_sheet = sheet_names[0] if len(sheet_names) > 0 else None

    # Find header sheet
    header_sheet = None
    for name in sheet_names:
        if any(k in name for k in ["订单表", "header", "order", "主表", "master"]):
            header_sheet = name
            break
    if header_sheet is None and len(sheet_names) > 1:
        header_sheet = sheet_names[1] if sheet_names[1] != detail_sheet else sheet_names[0]

    logger.info("  Detail sheet: '%s', Header sheet: '%s'", detail_sheet, header_sheet)

    # Load sheets
    details = pd.read_excel(file_path, sheet_name=detail_sheet) if detail_sheet else pd.DataFrame()
    headers = pd.read_excel(file_path, sheet_name=header_sheet) if header_sheet else pd.DataFrame()

    # Auto-detect columns
    d_cols = list(details.columns) if len(details) > 0 else []
    h_cols = list(headers.columns) if len(headers) > 0 else []

    order_id_col = (SchemaDetector.find_col(d_cols, SchemaDetector.ORDER_ID_PATTERNS) or
                    SchemaDetector.find_col(d_cols, [c for c in d_cols if "单号" in c or "订单" in c]))
    qty_col = (SchemaDetector.find_col(d_cols, SchemaDetector.QTY_PATTERNS) or
               SchemaDetector.find_col(d_cols, [c for c in d_cols if "EA" in c.upper() or "数量" in c]))
    prod_name_col = SchemaDetector.find_col(d_cols, SchemaDetector.PRODUCT_NAME_PATTERNS)
    cust_id_col = SchemaDetector.find_col(h_cols, SchemaDetector.CUSTOMER_ID_PATTERNS)
    date_col = SchemaDetector.find_col(h_cols, SchemaDetector.DATE_PATTERNS)
    prov_col = SchemaDetector.find_col(h_cols, SchemaDetector.PROVINCE_PATTERNS)
    wh_col = SchemaDetector.find_col(h_cols, SchemaDetector.WAREHOUSE_PATTERNS)

    logger.info("  Detected: order_id='%s', qty='%s', product='%s', customer='%s', date='%s'",
                order_id_col, qty_col, prod_name_col, cust_id_col, date_col)

    if not order_id_col:
        raise ValueError(f"Cannot find order ID column. Available: {d_cols}")
    if not qty_col:
        raise ValueError(f"Cannot find quantity column. Available: {d_cols}")

    # Build result
    result = details.copy()
    result["_qty"] = result[qty_col].fillna(0)

    # Merge with headers if available
    if len(headers) > 0 and cust_id_col and date_col:
        merge_cols = [order_id_col, cust_id_col, date_col]
        if prov_col:
            merge_cols.append(prov_col)
        if wh_col:
            merge_cols.append(wh_col)
        # Deduplicate headers on order_id
        headers_dedup = headers.drop_duplicates(subset=[order_id_col], keep="first")
        available_cols = [c for c in merge_cols if c in headers_dedup.columns]
        result = result.merge(headers_dedup[available_cols], on=order_id_col, how="left")

    # Extract fields
    result["date"] = pd.to_datetime(result[date_col], errors="coerce").dt.normalize() if date_col and date_col in result.columns else pd.NaT
    result["customer_code"] = result[cust_id_col] if cust_id_col and cust_id_col in result.columns else "C01"
    result["product_name"] = result[prod_name_col] if prod_name_col and prod_name_col in result.columns else "unknown"
    result["quantity"] = result["_qty"]
    result["province"] = result[prov_col].str.split("-").str[0] if prov_col and prov_col in result.columns else ""
    result["warehouse"] = result[wh_col] if wh_col and wh_col in result.columns else ""

    # Extract year from data
    valid_dates = result["date"].dropna()
    year = int(valid_dates.dt.year.mode()[0]) if len(valid_dates) > 0 else 2026
    result["_year"] = year

    logger.info("  Loaded %d rows, year=%d, date range=%s to %s",
                len(result), year,
                valid_dates.min().date() if len(valid_dates) > 0 else "N/A",
                valid_dates.max().date() if len(valid_dates) > 0 else "N/A")

    keep_cols = ["date", "customer_code", "product_name", "quantity", "province", "warehouse", "_year"]
    keep_cols = [c for c in keep_cols if c in result.columns]
    return result[keep_cols]


def categorize_products(df: pd.DataFrame) -> pd.DataFrame:
    """Keyword-based product categorization."""
    def cat(name):
        if pd.isna(name):
            return "other"
        n = str(name)
        if any(k in n for k in ["脆卜", "萝卜", "外婆菜", "酸菜"]):
            return "pickled_vegetables"
        if any(k in n for k in ["大米", "米"]):
            return "rice"
        if any(k in n for k in ["油", "植物", "调和油"]):
            return "oil"
        if any(k in n for k in ["肉", "肉片", "肉丝", "肉馅", "肥膘", "腊肉", "狮子头", "鸡腿", "鸡腿肉", "猪肝", "牛肉", "瘦肉", "猪肉"]):
            return "meat"
        if any(k in n for k in ["豆干", "腐竹", "木耳", "鹿茸菇", "青豆"]):
            return "bean_tofu"
        if any(k in n for k in ["蛋", "卤蛋", "鸡蛋"]):
            return "egg"
        if any(k in n for k in ["辣椒", "剁椒", "泡椒"]):
            return "chili"
        if any(k in n for k in ["酱油", "生抽", "醋"]):
            return "sauce"
        if any(k in n for k in ["生粉", "淀粉"]):
            return "starch"
        if any(k in n for k in ["袋", "餐具", "手提袋", "碗"]):
            return "packaging"
        if any(k in n for k in ["杨梅", "汁", "豆奶", "饮料"]):
            return "beverage"
        return "other"

    df["product_category"] = df["product_name"].apply(cat)
    return df


def aggregate_daily(df: pd.DataFrame, end_date: Optional[str] = None) -> pd.DataFrame:
    """Aggregate to daily level with complete date grid."""
    daily = df.groupby(["date", "customer_code", "product_category"]).agg(
        quantity=("quantity", "sum"),
    ).reset_index()

    start = daily["date"].min()
    end = pd.Timestamp(end_date) if end_date else daily["date"].max()
    dates = pd.date_range(start, end, freq="D")
    customers = daily["customer_code"].unique()
    categories = daily["product_category"].unique()

    grid = pd.MultiIndex.from_product([dates, customers, categories],
                                       names=["date", "customer_code", "product_category"]).to_frame(index=False)
    daily = grid.merge(daily, on=["date", "customer_code", "product_category"], how="left")
    daily["quantity"] = daily["quantity"].fillna(0)

    # Fill province/warehouse from training
    for col in ["province", "warehouse"]:
        if col in df.columns:
            daily[col] = daily["customer_code"].map(df.groupby("customer_code")[col].first().to_dict())

    daily["year"] = daily["date"].dt.year
    logger.info("Daily: %d rows (%d dates x %d customers x %d cats)",
                len(daily), len(dates), len(customers), len(categories))
    return daily
