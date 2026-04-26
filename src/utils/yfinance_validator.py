"""Validate LLM-extracted KPIs against ground-truth data from yfinance."""

from typing import Optional
import yfinance as yf


def get_ground_truth(ticker: str) -> dict:
    """Retrieve structured financial KPIs for a ticker using yfinance."""
    try:
        t = yf.Ticker(ticker)
        financials = t.financials
        balance_sheet = t.balance_sheet
        info = t.info

        def safe_get(df, row_key, col_index=0):
            try:
                if row_key in df.index:
                    val = df.loc[row_key].iloc[col_index]
                    if val is not None and str(val) != "nan":
                        return float(val)
            except Exception:
                pass
            return None

        revenue = safe_get(financials, "Total Revenue")
        gross_profit = safe_get(financials, "Gross Profit")
        operating_income = safe_get(financials, "Operating Income")
        net_income = safe_get(financials, "Net Income")
        ebitda = info.get("ebitda")
        total_debt = safe_get(balance_sheet, "Total Debt")
        stockholders_equity = safe_get(balance_sheet, "Stockholders Equity")
        cash = safe_get(balance_sheet, "Cash And Cash Equivalents")

        gross_margin_pct = None
        if revenue and gross_profit:
            gross_margin_pct = round((gross_profit / revenue) * 100, 2)

        operating_margin_pct = None
        if revenue and operating_income:
            operating_margin_pct = round((operating_income / revenue) * 100, 2)

        debt_equity_ratio = None
        if total_debt and stockholders_equity and stockholders_equity != 0:
            debt_equity_ratio = round(total_debt / stockholders_equity, 4)

        return {
            "revenue": round(revenue / 1e6, 2) if revenue else None,
            "gross_margin_pct": gross_margin_pct,
            "operating_margin_pct": operating_margin_pct,
            "net_income": round(net_income / 1e6, 2) if net_income else None,
            "ebitda": round(ebitda / 1e6, 2) if ebitda else None,
            "debt_equity_ratio": debt_equity_ratio,
            "cash_and_equivalents": round(cash / 1e6, 2) if cash else None,
        }

    except Exception as e:
        return {"error": str(e)}


def compare_with_llm_output(ground_truth: dict, llm_output: dict) -> dict:
    """Compare LLM-extracted KPIs against yfinance ground truth with 5% tolerance."""
    TOLERANCE = 0.05
    matched = 0
    total = 0
    mismatches = []

    numeric_keys = [
        "gross_margin_pct",
        "operating_margin_pct",
        "debt_equity_ratio",
    ]
    nested_keys = {
        "revenue": "value",
        "net_income": "value",
        "ebitda": "value",
        "cash_and_equivalents": "value",
    }

    for key in numeric_keys:
        gt_val = ground_truth.get(key)
        llm_val = llm_output.get(key)
        if gt_val is None or llm_val is None:
            continue
        total += 1
        try:
            diff = abs(float(llm_val) - float(gt_val))
            if gt_val != 0 and (diff / abs(gt_val)) <= TOLERANCE:
                matched += 1
            elif gt_val == 0 and diff == 0:
                matched += 1
            else:
                mismatches.append({
                    "key": key,
                    "ground_truth": gt_val,
                    "llm_output": llm_val,
                    "diff_pct": round((diff / abs(gt_val)) * 100, 2) if gt_val != 0 else None,
                })
        except (TypeError, ValueError):
            mismatches.append({"key": key, "error": "non-numeric value"})

    for key, subkey in nested_keys.items():
        gt_val = ground_truth.get(key)
        llm_nested = llm_output.get(key)
        if gt_val is None or not isinstance(llm_nested, dict):
            continue
        llm_val = llm_nested.get(subkey)
        if llm_val is None:
            continue
        total += 1
        try:
            diff = abs(float(llm_val) - float(gt_val))
            if gt_val != 0 and (diff / abs(gt_val)) <= TOLERANCE:
                matched += 1
            elif gt_val == 0 and diff == 0:
                matched += 1
            else:
                mismatches.append({
                    "key": key,
                    "ground_truth": gt_val,
                    "llm_output": llm_val,
                    "diff_pct": round((diff / abs(gt_val)) * 100, 2) if gt_val != 0 else None,
                })
        except (TypeError, ValueError):
            mismatches.append({"key": key, "error": "non-numeric value"})

    accuracy_pct = round((matched / total) * 100, 2) if total > 0 else 0.0

    return {
        "matched": matched,
        "total": total,
        "accuracy_pct": accuracy_pct,
        "mismatches": mismatches,
    }
