import datetime
import os
import requests
import sympy
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.
    Use this tool when the user asks for mathematical calculations or evaluations.
    Input should be a mathematical expression like '2 + 2' or 'sin(30)' or 'sqrt(16) * 5'.
    """
    try:
        # Use SymPy to parse and evaluate the expression safely
        expr = sympy.sympify(expression)
        result = expr.evalf()
        return f"Result of {expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for the given query and return a summary of the page.
    Use this tool when the user asks general knowledge questions, info about famous people, places, history, etc.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "formatversion": 2
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return f"No results found on Wikipedia for '{query}'."
        
        # Get page summary for the first result
        pageid = search_results[0]["pageid"]
        title = search_results[0]["title"]
        summary_params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "pageids": pageid,
            "formatversion": 2
        }
        res_summary = requests.get(url, params=summary_params, headers=headers, timeout=5)
        summary_data = res_summary.json()
        pages = summary_data.get("query", {}).get("pages", [])
        if pages:
            extract = pages[0].get("extract", "")
            return f"Wikipedia article: {title}\nSummary: {extract}"
        return f"Could not retrieve details for '{title}'."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@tool
def get_current_time() -> str:
    """
    Get the current local date and time.
    Use this tool when the user asks for the current date, current time, day of the week, etc.
    """
    now = datetime.datetime.now()
    return f"The current local date and time is: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price and key statistics for a given stock ticker symbol.
    Use this tool when the user asks for stock prices, stock quotes, or market value of a specific ticker (e.g. AAPL, MSFT, GOOG).
    Input should be a stock ticker symbol (e.g., 'AAPL' for Apple, 'TSLA' for Tesla).
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Error: ALPHA_VANTAGE_API_KEY is not set in the environment."
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker.upper(),
        "apikey": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        quote = data.get("Global Quote", {})
        if not quote:
            note = data.get("Note")
            if note:
                return f"Alpha Vantage API Notice: {note}"
            return f"Error: Could not retrieve quote for symbol '{ticker}'. Make sure it is a valid stock ticker symbol."
            
        symbol = quote.get("01. symbol", ticker)
        price = quote.get("05. price", "N/A")
        change = quote.get("09. change", "N/A")
        change_percent = quote.get("10. change percent", "N/A")
        latest_day = quote.get("07. latest trading day", "N/A")
        
        return (
            f"Value metric for {symbol}: {price}\n"
            f"- Change metric: {change} ({change_percent})\n"
            f"- Reference date: {latest_day}"
        )
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"
