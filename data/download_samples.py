"""Script to download real financial data from SEC EDGAR and yfinance."""

import os
import sys
import traceback

import yfinance as yf

# Ensure data directories exist
SAMPLE_10K_DIR = os.path.join(os.path.dirname(__file__), "sample_10k")
COMPETITOR_DIR = os.path.join(os.path.dirname(__file__), "competitor_docs")
os.makedirs(SAMPLE_10K_DIR, exist_ok=True)
os.makedirs(COMPETITOR_DIR, exist_ok=True)

MAIN_TICKERS = ["AAPL", "MSFT", "NVDA"]
COMPETITOR_TICKERS = ["AMD", "GOOGL"]


def download_10k_edgar(ticker: str, company_name: str, cik: str, output_dir: str) -> None:
    """Download the most recent 10-K filing from SEC EDGAR for a given company."""
    try:
        from edgar import Company

        company = Company(company_name, cik)

        # edgar >= 5.x API: get_10K() returns an lxml HtmlElement
        filing_element = company.get_10K()
        if filing_element is None:
            print(f"  ERROR: No 10-K filing found for {ticker}")
            return

        # Extract plain text from the HTML element
        text = filing_element.text_content()
        if not text or not text.strip():
            print(f"  ERROR: 10-K for {ticker} returned empty text")
            return

        year = "2024"
        out_path = os.path.join(output_dir, f"{ticker}_10K_{year}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Downloaded: {out_path}")

    except Exception as e:
        print(f"  ERROR downloading 10-K for {ticker}: {e}")
        traceback.print_exc()


def download_yfinance_data(ticker: str, output_dir: str) -> None:
    """Download income statement, balance sheet, and cash flow from yfinance."""
    try:
        t = yf.Ticker(ticker)

        financials = t.financials
        balance_sheet = t.balance_sheet
        cashflow = t.cashflow

        fin_path = os.path.join(output_dir, f"{ticker}_financials.csv")
        bs_path = os.path.join(output_dir, f"{ticker}_balance_sheet.csv")
        cf_path = os.path.join(output_dir, f"{ticker}_cashflow.csv")

        financials.to_csv(fin_path)
        print(f"Downloaded: {fin_path}")

        balance_sheet.to_csv(bs_path)
        print(f"Downloaded: {bs_path}")

        cashflow.to_csv(cf_path)
        print(f"Downloaded: {cf_path}")

    except Exception as e:
        print(f"  ERROR downloading yfinance data for {ticker}: {e}")


# Mapping: ticker -> (company name, CIK)
EDGAR_INFO = {
    "AAPL": ("Apple Inc", "0000320193"),
    "MSFT": ("Microsoft Corp", "0000789019"),
    "NVDA": ("NVIDIA Corp", "0001045810"),
    "AMD": ("Advanced Micro Devices Inc", "0000002488"),
    "GOOGL": ("Alphabet Inc", "0001652044"),
}


def main() -> None:
    """Run all downloads."""
    print("=== Part A: Downloading 10-K filings from SEC EDGAR ===")
    for ticker in MAIN_TICKERS:
        name, cik = EDGAR_INFO[ticker]
        print(f"Fetching 10-K for {ticker} ({name})...")
        download_10k_edgar(ticker, name, cik, SAMPLE_10K_DIR)

    print("\n=== Part B: Downloading structured financials via yfinance ===")
    for ticker in MAIN_TICKERS:
        print(f"Fetching financials for {ticker}...")
        download_yfinance_data(ticker, SAMPLE_10K_DIR)

    print("\n=== Part C: Downloading competitor 10-K filings ===")
    for ticker in COMPETITOR_TICKERS:
        name, cik = EDGAR_INFO[ticker]
        print(f"Fetching 10-K for {ticker} ({name})...")
        download_10k_edgar(ticker, name, cik, COMPETITOR_DIR)

    print("\nAll downloads complete. Run 'python app.py' to start.")


if __name__ == "__main__":
    main()
