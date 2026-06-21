"""
Stats Pipeline API
يستقبل ملف اكسيل خام، ينظفه، يعمل تحليل احصائي، ويرجع النتائج لـ n8n
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
import re

app = FastAPI(title="Stats Pipeline API")


@app.get("/")
def health_check():
    """endpoint بسيط للتأكد ان السيرفر شغال"""
    return {"status": "ok", "message": "Stats Pipeline API is running"}


def clean_numeric_text(series: pd.Series) -> pd.Series:
    """
    بيشيل الرموز زي < > و mg/dL من القيم النصية
    ويحاول يحولها لارقام
    """
    def clean_value(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return val
        val_str = str(val).strip()
        # شيل رموز المقارنة والوحدات الشائعة
        val_str = re.sub(r'[<>≤≥]', '', val_str)
        val_str = re.sub(r'(mg/dL|mg|dL|ml|mmol/L|g/L|%)', '', val_str, flags=re.IGNORECASE)
        val_str = val_str.strip()
        try:
            return float(val_str)
        except ValueError:
            return np.nan
    return series.apply(clean_value)


def detect_column_type(series: pd.Series, threshold_unique_ratio: float = 0.05) -> str:
    """
    بيحدد هل العمود رقمي ولا فئوي
    حتى لو القيم متخزنة كنص فيه رموز
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unknown"

    # جرب تنضف وتحول لرقم
    cleaned = clean_numeric_text(series)
    numeric_success_ratio = cleaned.notna().sum() / max(len(non_null), 1)

    if numeric_success_ratio > 0.8:
        # غالبية القيم اتحولت بنجاح لرقم
        unique_count = cleaned.nunique()
        # لو القيم الفريدة قليلة جدا (زي 0,1,2) ممكن تكون فئوية مشفرة كأرقام
        if unique_count <= 10 and unique_count / max(len(non_null), 1) < threshold_unique_ratio:
            return "categorical_coded"
        return "numerical"
    else:
        return "categorical"


@app.post("/clean")
async def clean_data(file: UploadFile = File(...)):
    """
    Step 2: استقبال البيانات الخام وتنظيفها
    - تصنيف الاعمدة (رقمي / فئوي)
    - حذف الاعمدة الغير مفيدة (ID, names, etc)
    - تعويض القيم الناقصة (median / MICE حسب نسبة الفقد)
    """
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    original_shape = df.shape
    original_columns = list(df.columns)

    # ---------- 1. حذف الاعمدة الغير مفيدة احصائياً ----------
    columns_to_drop_patterns = [
        r'^id$', r'_id$', r'^name$', r'first_?name', r'last_?name',
        r'email', r'phone', r'tel', r'address', r'ssn'
    ]
    dropped_columns = []
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(re.search(pattern, col_lower) for pattern in columns_to_drop_patterns):
            dropped_columns.append(col)

    df = df.drop(columns=dropped_columns, errors='ignore')

    # ---------- 2. تصنيف نوع كل عمود ----------
    column_types = {}
    for col in df.columns:
        column_types[col] = detect_column_type(df[col])
        # لو العمود رقمي بس متخزن كنص فيه رموز، ننضفه فعلياً
        if column_types[col] == "numerical":
            df[col] = clean_numeric_text(df[col])

    # ---------- 3. حساب نسبة القيم الناقصة لكل عمود ----------
    missing_report = {}
    columns_dropped_missing = []
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df) * 100
        missing_report[col] = round(missing_pct, 2)
        if missing_pct > 50:
            columns_dropped_missing.append(col)

    df = df.drop(columns=columns_dropped_missing, errors='ignore')

    # ---------- 4. تعويض القيم الناقصة ----------
    imputation_log = {}
    for col in df.columns:
        if column_types.get(col) not in ("numerical",):
            continue
        missing_pct = missing_report.get(col, 0)
        if missing_pct == 0:
            continue
        if missing_pct < 10:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputation_log[col] = f"Median imputation ({median_val:.2f})"
        elif missing_pct <= 50:
            # MICE - هنحطها في الخطوة الجاية، دلوقتي نستخدم median كـ placeholder
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputation_log[col] = f"MICE imputation (placeholder: median {median_val:.2f})"

    result = {
        "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
        "cleaned_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dropped_columns_unuseful": dropped_columns,
        "dropped_columns_missing": columns_dropped_missing,
        "column_types": column_types,
        "missing_report_pct": missing_report,
        "imputation_log": imputation_log,
        "cleaned_columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records")
    }

    return JSONResponse(content=result)
