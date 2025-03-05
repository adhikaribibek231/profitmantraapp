import requests
from bs4 import BeautifulSoup
import csv
import os

def scrape_stock_data(stock_symbol):
    """
    Scrapes stock data for the given stock symbol and saves it as a CSV file.
    Returns True if successful, False otherwise.
    """
    folder_name = "Stock"
    os.makedirs(folder_name, exist_ok=True)
    
    # Load company ID data
    company_id_map = {}
    try:
        with open("companyid.txt", "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    symbol, company_id = parts
                    company_id_map[symbol.upper()] = company_id
    except FileNotFoundError:
        print("Error: companyid.txt not found.")
        return False

    if stock_symbol not in company_id_map:
        print(f"Error: Company symbol {stock_symbol} not found in companyid.txt")
        return False

    company_id = company_id_map[stock_symbol]
    url = f"https://www.sharesansar.com/company/{stock_symbol}"

    session = requests.Session()
    initial_response = session.get(url)
    initial_soup = BeautifulSoup(initial_response.content, "html.parser")
    token_input = initial_soup.find("input", {"name": "_token"})
    
    if not token_input:
        print("Error: Unable to retrieve CSRF token.")
        return False
    
    token_value = token_input["value"]
    api_url = "https://www.sharesansar.com/company-price-history"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": url,
        "X-Csrf-Token": token_value,
        "X-Requested-With": "XMLHttpRequest",
    }

    filename = f"{stock_symbol}_price_history.csv"
    filepath = os.path.join(folder_name, filename)
    payload = {"draw": 1, "start": 0, "length": 50, "search[value]": "", "search[regex]": "false", "company": company_id}
    csv_headers = ["published_date", "open", "high", "low", "close", "per_change", "traded_quantity", "traded_amount", "status", "DT_Row_Index"]
    
    with open(filepath, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_headers)
        writer.writeheader()
    
    start = 0
    length = 50
    while True:
        payload["start"] = start
        response = session.post(api_url, data=payload, headers=headers)
        response_data = response.json()
        data = response_data.get("data", [])
        
        if not data:
            break
        
        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_headers)
            for item in data:
                writer.writerow(item)
        
        start += length
    
    print(f"Data has been saved to {filepath}")
    return True

if __name__ == "__main__":
    stock_symbol = input("Enter the company symbol: ").upper()
    scrape_stock_data(stock_symbol)
