import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from COMP3012_classifier_combined import merge_surprise_dates
import COMP3012_HMM_classifier as hmm_classifier
import COMP3012_DSPOT_classifier as dspot_classifier

#Define trading parameters
notional_base = 100000
lookback_period = 120  # Last 120 trading days

pricing_currency = {
    "EURUSD": "EUR",
    "AUDJPY": "JPY",
    "USDJPY": "JPY",
    "EURGBP": "GBP",
    "USDCAD": "CAD",
    "AUDUSD": "AUD",
    "EURCHF": "CHF",
    "FTSE100": "GBP",  # GBP-denominated
    "S&P500": "USD",  # USD-denominated
    "Nasdaq100": "USD",  # USD-denominated
    "Nikkei": "JPY",  # JPY-denominated
    "Eurostoxx50": "EUR",  # EUR-denominated
    "Gold": "USD",
    "Silver": "USD"
}

asset_currency_mapping = {
    "FTSE100": "FTSE/GBP",
    "S&P500": "S&P/USD",
    "Nasdaq100": "Nasdaq/USD",
    "Nikkei": "Nikkei/JPY",
    "Eurostoxx50": "Eurostoxx/EUR",
    "Gold": "Gold/USD",
    "Silver": "Silver/USD"
}

inverse_fx_pairs = {
    "USDJPY": "JPYUSD",
    "USDCAD": "CADUSD",
    "USDCHF": "CHFUSD"
}

fx_derivation_map = {
    "GBPUSD": ("EURUSD", "EURGBP"),
    "CHFUSD": ("EURUSD", "EURCHF")
}

# Function to fetch prices dynamically for cross pairs
def get_related_price(date, related_pair, price_data_dict):
    """Fetch the current price for the related pair on a given date."""
    if related_pair not in price_data_dict:
        raise ValueError(f"Missing price data for related pair: {related_pair}")

    # Fetch data for the related pair
    data = price_data_dict[related_pair]

    # Fix: Use 'DATES' instead of 'Date'
    if 'DATES' not in data.columns:
        raise ValueError(f"Column 'DATES' not found in {related_pair}. Available columns: {data.columns}")

    # Filter price data for the given date
    price_row = data[data['DATES'] == date]

    if price_row.empty:
        raise ValueError(f"No price data available for {related_pair} on date {date}")

    return price_row['Price'].values[0]  # Fix: Ensure 'Price' column is correctly referenced


def calculate_trade_size(asset_name, std_dev_asset, date, price_data_dict):
    if std_dev_asset == 0:
        return 0.0  # Avoid division by zero

    notional_base = 100000  # USD Risk
    # If it's an FX pair (not in asset_currency_mapping), extract base and quote normally
    if asset_name not in asset_currency_mapping:
        base_currency, quote_currency = asset_name[:3], asset_name[3:]
        if quote_currency == "USD":
            fx_rate = get_related_price(date, f"DC {asset_name}", price_data_dict)
        else:
            if base_currency == "USD":
                fx_rate = 1  # USDUSD case (shouldn't happen)
            else:
                # Convert via an intermediary currency (e.g., AUDJPY -> AUDUSD)
                if f"DC {base_currency}USD" in price_data_dict:
                    fx_rate = get_related_price(date, f"DC {base_currency}USD", price_data_dict)
                else:
                    raise ValueError(f"Cannot determine FX rate for {asset_name} on {date}")

        trade_size = (notional_base / fx_rate) / std_dev_asset
        return trade_size
        
    else:
        quote_currency = pricing_currency[asset_name]  # Stocks only have a quote currency
        # If stock is priced in USD, no conversion needed
        if quote_currency == "USD":
            fx_rate = 1  

        # If stock is priced in a non-USD currency, find conversion rate
        elif f"DC USD{quote_currency}" in price_data_dict:
            fx_rate = 1/get_related_price(date, f"DC USD{quote_currency}", price_data_dict)  # Invert USD/JPY

        elif f"DC {quote_currency}USD" in price_data_dict:
            fx_rate = get_related_price(date, f"DC {quote_currency}USD", price_data_dict)

        elif quote_currency + "USD" in fx_derivation_map:
            # Handle missing GBPUSD, CHFUSD using derivation
            fx1, fx2 = fx_derivation_map[quote_currency + "USD"]
            fx_rate = get_related_price(date, f"DC {fx1}", price_data_dict) / get_related_price(date, f"DC {fx2}", price_data_dict)

        else:
            raise ValueError(f"Cannot determine FX rate for {asset_name} on {date}")

        trade_size = (notional_base / fx_rate) / std_dev_asset
        return trade_size


def calculate_pnl_fx(entry_price, exit_price, trade_size, asset_name, date, price_data_dict):
    """
    Calculate PnL for FX pairs and stock indices.

    Parameters:
    - entry_price (float): Entry FX or asset price
    - exit_price (float): Exit FX or asset price
    - trade_size (float): Position size in base currency
    - asset_name (str): FX pair (e.g., "EURUSD") or stock index (e.g., "FTSE100")
    - date (datetime): Trade date for FX conversion
    - price_data_dict (dict): FX and asset price data dictionary

    Returns:
    - float: PnL in USD
    """
    # **CASE 1: FX PAIR**
    if asset_name not in asset_currency_mapping:
        base_currency, quote_currency = asset_name[:3], asset_name[3:]

        # Step 1: Compute profit in base currency
        amount_sold_in_quote = trade_size * entry_price  # Amount of quote currency (e.g., USD for EURUSD)
        amount_bought_in_base = amount_sold_in_quote / exit_price  # Convert back to base currency
        profit_in_base_currency = amount_bought_in_base - trade_size  # Profit in base currency

        # Step 2: Convert profit from base currency to USD
        if base_currency == "USD":
            usd_conversion_rate = 1
        elif f"DC {base_currency}USD" in price_data_dict:
            usd_conversion_rate = get_related_price(date, f"DC {base_currency}USD", price_data_dict)
        elif f"DC USD{base_currency}" in price_data_dict:
            usd_conversion_rate = 1 / get_related_price(date, f"DC USD{base_currency}", price_data_dict)  # Invert USD/JPY
        elif base_currency + "USD" in fx_derivation_map:
            fx1, fx2 = fx_derivation_map[base_currency + "USD"]
            usd_conversion_rate = get_related_price(date, f"DC {fx1}", price_data_dict) / get_related_price(date, f"DC {fx2}", price_data_dict)
        else:
            raise ValueError(f"Cannot convert {base_currency} to USD on {date}")

        profit_in_usd = profit_in_base_currency * usd_conversion_rate

    # **CASE 2: STOCK INDEX (TREATED LIKE A CURRENCY PAIR)**
    else:
        quote_currency = pricing_currency[asset_name]  # Stock index only has a quote currency

        # Step 1: Compute amount sold in quote currency (Trade Size is already in quote currency)
        amount_sold_in_quote = trade_size

        # Step 2: Compute amount bought in stock (Base Currency)
        amount_bought_in_stock = amount_sold_in_quote / entry_price

        # Step 3: Compute amount of stock sold at exit price
        amount_sold_in_stock = amount_sold_in_quote / exit_price

        # Step 4: Compute profit in stock units
        profit_in_stock_units = amount_sold_in_stock - amount_bought_in_stock

        # Step 2: Convert profit from base currency to USD
        if quote_currency == "USD":
            usd_conversion_rate=1
        elif f"DC USD{quote_currency}" in price_data_dict:
            usd_conversion_rate = 1 / get_related_price(date, f"DC USD{quote_currency}", price_data_dict)  # Invert USD/JPY
        elif f"DC {quote_currency}USD" in price_data_dict:
            usd_conversion_rate = get_related_price(date, f"DC {quote_currency}USD", price_data_dict)
        elif quote_currency + "USD" in fx_derivation_map:
            fx1, fx2 = fx_derivation_map[quote_currency + "USD"]
            usd_conversion_rate = get_related_price(date, f"DC {fx1}", price_data_dict) / get_related_price(date, f"DC {fx2}", price_data_dict)
        else:
            raise ValueError(f"Cannot convert {quote_currency} to USD on {date}")

        profit_in_usd = profit_in_stock_units * usd_conversion_rate * exit_price


    return profit_in_usd



def record_basic_strategy(data, pricing_currency, asset_name, price_data_dict, short, large):
    """Apply a basic moving average crossover strategy with trade-triggered PnL calculation and log buy and sell trades."""
    data['13-day MA'] = data['Price'].rolling(window=short).mean()
    data['25-day MA'] = data['Price'].rolling(window=large).mean()
    data['Signal'] = 0
    data.loc[data['13-day MA'] > data['25-day MA'], 'Signal'] = 1
    data.loc[data['13-day MA'] < data['25-day MA'], 'Signal'] = -1
    records = []  # List to store trade records
    contracts_held = 0
    total_pnl = 0.0
    last_trade_price = None
    data['Basic_PnL'] = 0.0
    data['Trade_Action_Basic'] = 'Hold'


    for i in range(1, len(data)):
        current_price = data.iloc[i]['Price']
        current_date = data.iloc[i]['DATES']
        std_dev = data['Price'].iloc[max(0, i - lookback_period):i].std() / current_price
        trade_size = calculate_trade_size(asset_name, std_dev, current_date, price_data_dict)

        signal = data.iloc[i]['Signal']
        
        if signal == 1 and contracts_held <= 0:  # Buy signal while short
            if contracts_held != 0:
                profit_change = calculate_pnl_fx(
                    last_trade_price, current_price, abs(contracts_held), asset_name, current_date, price_data_dict
                )

                total_pnl += profit_change  # Accumulate USD PnL

                records.append({
                    "Date": current_date,
                    "Signal": "Buy",
                    "Position Size Held": contracts_held,
                    "Last Trade Price": last_trade_price,
                    "Current Price": current_price,
                    "Profit Change (USD)": profit_change,
                    "Std Dev": std_dev,
                    "Total PnL": total_pnl
                })

            contracts_held = trade_size  # Open a new position
            last_trade_price = current_price
            data.iloc[i, data.columns.get_loc('Trade_Action_Basic')] = f'Buy {trade_size:.2f} units'

        elif signal == -1 and contracts_held >= 0:  # Sell signal while long
            if contracts_held != 0:
                profit_change = calculate_pnl_fx(
                    last_trade_price, current_price, abs(contracts_held), asset_name, current_date, price_data_dict
                )

                total_pnl += profit_change  # Accumulate USD PnL

                records.append({
                    "Date": current_date,
                    "Signal": "Sell",
                    "Position Size Held": contracts_held,
                    "Last Trade Price": last_trade_price,
                    "Current Price": current_price,
                    "Profit Change (USD)": profit_change,
                    "Std Dev": std_dev,
                    "Total PnL": total_pnl
                })

            contracts_held = -trade_size  # Open a new short position
            last_trade_price = current_price
            data.iloc[i, data.columns.get_loc('Trade_Action_Basic')] = f'Sell {trade_size:.2f} units'

        data.iloc[i, data.columns.get_loc('Basic_PnL')] = total_pnl  # Update PnL in data
    return total_pnl, records  # Return both total PnL and records




# Enhanced Trading System with correct trade-triggered PnL updates
def record_enhanced_strategy(data, signal_strategy, pricing_currency, asset_name, price_data_dict, short, large):
    """Apply an enhanced strategy with trade-triggered PnL calculation based on the signal strategy."""
    data['13-day MA'] = data['Price'].rolling(window=short).mean()
    data['25-day MA'] = data['Price'].rolling(window=large).mean()
    data['Signal'] = 0
    data.loc[data['13-day MA'] > data['25-day MA'], 'Signal'] = 1
    data.loc[data['13-day MA'] < data['25-day MA'], 'Signal'] = -1
    records=[]
    contracts_held = 0
    total_pnl = 0.0
    last_trade_price = None
    data['Enhanced_PnL'] = 0.0
    data['Trade_Action_Enhanced'] = 'Hold'

    for i in range(1, len(data)):
        current_price = data.iloc[i]['Price']
        current_date = data.iloc[i]['DATES']  # Ensure correct date reference
        std_dev = (data['Price'].iloc[max(0, i - lookback_period):i].std()/ current_price)
        
        # Calculate trade size with correct date handling
        trade_size = calculate_trade_size(asset_name, std_dev, current_date, price_data_dict)

        is_surprise = data.iloc[i]['Surprise']
        signal = data.iloc[i]['Signal']

        # Apply strategy-specific trade rules
        if signal_strategy == 1 and is_surprise:  # Double trade size on surprise
            trade_size *= 2
        elif signal_strategy == 0 and not is_surprise:  # Trade only when surprise occurs
            continue
        elif signal_strategy == -1 and is_surprise:  # Do not trade on surprise
            continue

         # Trading logic and PnL updates based on signals
        if signal == 1 and contracts_held <= 0:  # Buy signal
            if contracts_held != 0:
                profit_change = calculate_pnl_fx(
                    last_trade_price, current_price, abs(contracts_held), asset_name, current_date, price_data_dict
                )

                total_pnl += profit_change  # Accumulate USD PnL
                records.append({
                    "Date": current_date,
                    "Signal": "Buy",
                    "Position Size Held": contracts_held,
                    "Last Trade Price": last_trade_price,
                    "Current Price": current_price,
                    "Profit Change (USD)": profit_change,
                    "Std Dev": std_dev,
                    "Total PnL": total_pnl
                })

            contracts_held = trade_size  # Open a new position
            last_trade_price = current_price
            data.iloc[i, data.columns.get_loc('Trade_Action_Enhanced')] = f'Buy {trade_size:.2f} units'

        elif signal == -1 and contracts_held >= 0:  # Sell signal
            if contracts_held != 0:
                profit_change = calculate_pnl_fx(
                    last_trade_price, current_price, abs(contracts_held), asset_name, current_date, price_data_dict
                )
                total_pnl += profit_change  # Accumulate USD PnL

                records.append({
                    "Date": current_date,
                    "Signal": "Sell",
                    "Position Size Held": contracts_held,
                    "Last Trade Price": last_trade_price,
                    "Current Price": current_price,
                    "Profit Change (USD)": profit_change,
                    "Std Dev": std_dev,
                    "Total PnL": total_pnl
                })

            contracts_held = -trade_size  # Open a new short position
            last_trade_price = current_price
            data.iloc[i, data.columns.get_loc('Trade_Action_Enhanced')] = f'Sell {trade_size:.2f} units'

        data.iloc[i, data.columns.get_loc('Enhanced_PnL')] = total_pnl  # Update PnL in data

    return total_pnl, records  # Return both total PnL and records

# Function to filter the last 5 years of data
def filter_last_5_years(data):
    cutoff_date = data["DATES"].max() - pd.DateOffset(years=5)
    return data[data["DATES"] >= cutoff_date].reset_index(drop=True)



# Load and process each asset class
file_path = '/Users/caelenbrown/Downloads/log_transformed_output.xlsx'

# Run HMM and ML classifiers, then merge results
print("Running HMM and ML Classifiers")
hmm_surprises = hmm_classifier.train_hmm_model(file_path, hmm_classifier.asset_currency_mapping)
dspot_surprises = dspot_classifier.train_and_detect_dspot(file_path, dspot_classifier.asset_currency_mapping, dspot_classifier.PARAMS)

# Merge surprises from both classifiers
print("Merging Surprises from HMM & ML")
surprise_results = merge_surprise_dates(hmm_surprises, dspot_surprises)
# Debugging: Check available assets in surprise results
print("Checking Available Assets in Merged Surprises:")
print(list(surprise_results.keys()))  # Print all asset names in merged surprises

# Dictionary to store processed price data with surprises
price_data_dict = {}

# Load price data for each FX pair and merge with test set surprises
excel_file = pd.ExcelFile(file_path)
sheet_names = [sheet for sheet in excel_file.sheet_names if sheet.startswith('D')]

for sheet_name in sheet_names:
    
    price_data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=13)

    if 'DATES' not in price_data.columns or 'Price' not in price_data.columns:
        print(f"Missing required columns in {sheet_name}. Skipping...")
        continue

    price_data['DATES'] = pd.to_datetime(price_data['DATES'], errors='coerce')
    price_data = price_data.dropna(subset=['DATES']).sort_values(by='DATES').reset_index(drop=True)
    price_data = price_data[['DATES', 'Price']].dropna()
    price_data['Price'] = pd.to_numeric(price_data['Price'], errors='coerce')

    if price_data.empty or price_data['Price'].isna().all():
        print(f"Skipping {sheet_name} due to insufficient price data.")
        continue

    # **Filter only the last 5 years of test data**
    price_data_filtered = filter_last_5_years(price_data)

    # Debugging: Check if asset exists in surprise results
    print(f"Processing {sheet_name} (Mapped to {sheet_name.replace('DC ', '')})")
    print(f"Checking for surprises under: {sheet_name}")
    
    if sheet_name in surprise_results:
        surprise_dates = surprise_results[sheet_name]
        print(f"Found surprises for {sheet_name} → {len(surprise_dates)} events")
        price_data_filtered = pd.merge(
            price_data_filtered,
            pd.DataFrame({'DATES': pd.to_datetime(surprise_dates), 'Surprise': True}),
            on='DATES', how='left'
        )
    else:
        print(f"No surprises found for {sheet_name} in merged_surprises!")

    # Ensure 'Surprise' column exists after merging
    if 'Surprise' not in price_data_filtered.columns:
        print(f"ERROR: 'Surprise' column missing in {sheet_name} after merging!")
        price_data_filtered['Surprise'] = False
    
    price_data_filtered['Surprise'] = price_data_filtered['Surprise'].fillna(False)
    price_data_dict[sheet_name] = price_data_filtered

    print(f"Successfully processed {sheet_name} for last 5 years.")
        
    # Ensure data is not empty before processing
    if not price_data_filtered.empty:
        first_date = price_data_filtered['DATES'].min()
        last_date = price_data_filtered['DATES'].max()
        
        print(f"Successfully processed {sheet_name} for last 5 years. Date range: {first_date} → {last_date}")
    else:
        print(f"Skipped {sheet_name} due to insufficient data.")

print(f"Number of processed sheets: {len(price_data_dict)}")


#Running Strategies on Test Data
portfolio_results = []
# Initialize total PnL counters
total_pnl_basic = 0
total_pnl_enhanced = 0
for sheet_name, data in price_data_dict.items():
    asset_name = sheet_name.replace("DC ", "")
    print(price_data_dict.keys())
    pnl_basic, basic_trade_records = record_basic_strategy(data.copy(), pricing_currency, asset_name, price_data_dict, short=13, large=25)
    # Call enhanced strategy for comparison
    pnl_enhanced, enhanced_trade_records = record_enhanced_strategy(
        data=data.copy(),
        signal_strategy=1,
        pricing_currency=pricing_currency,
        asset_name=asset_name,
        price_data_dict=price_data_dict,
        short=13,
        large=25
    )

     # Save basic trade records to an Excel file in Downloads directory
    output_file = f"/Users/caelenbrown/Downloads/{asset_name}_basic_strategy_trades.xlsx"
    pd.DataFrame(basic_trade_records).to_excel(output_file, index=False)
    print(f"Saved {asset_name} basic strategy trade records to {output_file}")

    # Save basic trade records to an Excel file in Downloads directory
    output_file = f"/Users/caelenbrown/Downloads/{asset_name}_enhanced_strategy_trades.xlsx"
    pd.DataFrame(enhanced_trade_records).to_excel(output_file, index=False)
    print(f"Saved {asset_name} basic strategy trade records to {output_file}")

    portfolio_results.append({
        'Asset Class': asset_name,
        'Basic Strategy PnL': pnl_basic,
        'Enhanced Strategy PnL': pnl_enhanced
    })
    # Add PnL values to total counters
    total_pnl_basic += pnl_basic
    total_pnl_enhanced += pnl_enhanced

    
    print(f"Results for {asset_name}: Basic PnL: {pnl_basic}, Enhanced PnL: {pnl_enhanced}")

    print(f"Total Portfolio PnL: Basic Strategy: {total_pnl_basic}, Enhanced Strategy: {total_pnl_enhanced}")
    # 📊 Visualization and Summary
    portfolio_df = pd.DataFrame(portfolio_results)
    print(portfolio_df.head())  # Check what columns exist

# Add Total Portfolio PnL as a new row
total_pnl_row = {
    "Asset Class": "Total Portfolio",
    "Basic Strategy PnL": total_pnl_basic,
    "Enhanced Strategy PnL": total_pnl_enhanced
}

#Fix: Use pd.concat() instead of append()
portfolio_df = pd.concat([portfolio_df, pd.DataFrame([total_pnl_row])], ignore_index=True)

#Plot the bar chart with distinct colours (no blending)
plt.figure(figsize=(12, 8))

bar_width = 0.4  # Ensures bars don't fully overlap

# X-axis positions
x_positions = range(len(portfolio_df))

#Plot bars with clear edges to prevent blending
plt.bar(x_positions, portfolio_df["Basic Strategy PnL"], width=bar_width, color='blue', edgecolor='black', label="Basic Strategy")
plt.bar([x + bar_width for x in x_positions], portfolio_df["Enhanced Strategy PnL"], width=bar_width, color='orange', edgecolor='black', label="Enhanced Strategy")

#Formatting
plt.xlabel("Asset Class")
plt.ylabel("Final P&L ($)")
plt.title("Portfolio P&L Comparison Across Asset Classes (Including Total)")
plt.xticks([x + bar_width/2 for x in x_positions], portfolio_df["Asset Class"], rotation=45)  # Shift labels to match grouped bars
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.7)

#Show the chart
plt.tight_layout()
plt.show()

print("Portfolio Strategy Comparison")
print(portfolio_df)


def compute_final_pnl_with_plot(price_data_dict, pricing_currency):
    all_enhanced_records = []
    all_basic_records = []
    final_pnl_summary = {}

    for asset_name, data in price_data_dict.items():
        clean_asset_name = asset_name.replace("DC ", "")
        print(f"Processing {clean_asset_name}...")

        # Run strategies
        pnl_basic, basic_trade_records = record_basic_strategy(
            data.copy(), pricing_currency, clean_asset_name, price_data_dict, short=13, large=25
        )
        pnl_enhanced, enhanced_trade_records = record_enhanced_strategy(
            data.copy(), signal_strategy=1, pricing_currency=pricing_currency,
            asset_name=clean_asset_name, price_data_dict=price_data_dict, short=13, large=25
        )

        # Check for empty enhanced trade records
        if isinstance(enhanced_trade_records, list) and enhanced_trade_records:
            enhanced_trade_records = pd.DataFrame(enhanced_trade_records)
        else:
            print(f"Skipping {clean_asset_name}: No enhanced trade records.")
            continue

        if not all(col in enhanced_trade_records.columns for col in ["Date", "Profit Change (USD)"]):
            print(f"Skipping {clean_asset_name}: Missing required columns in enhanced trade records.")
            continue

        if isinstance(basic_trade_records, list) and basic_trade_records:
            basic_trade_records = pd.DataFrame(basic_trade_records)
        else:
            print(f"Skipping {clean_asset_name}: No basic trade records.")
            continue

        if not all(col in basic_trade_records.columns for col in ["Date", "Profit Change (USD)"]):
            print(f"Skipping {clean_asset_name}: Missing required columns in basic trade records.")
            continue

        # Parse dates and label
        enhanced_trade_records["Date"] = pd.to_datetime(enhanced_trade_records["Date"])
        enhanced_trade_records["Asset"] = clean_asset_name
        all_enhanced_records.append(enhanced_trade_records)

        basic_trade_records["Date"] = pd.to_datetime(basic_trade_records["Date"])
        basic_trade_records["Asset"] = clean_asset_name
        all_basic_records.append(basic_trade_records)

        final_pnl_summary[clean_asset_name] = {
            "Final Basic PnL": pnl_basic,
            "Final Enhanced PnL": pnl_enhanced
        }

    if not all_enhanced_records or not all_basic_records:
        print("No valid trade data found. Exiting.")
        return None, None

    # Combine and aggregate
    all_enhanced_df = pd.concat(all_enhanced_records)
    all_basic_df = pd.concat(all_basic_records)

    enhanced_pnl_by_date = all_enhanced_df.groupby("Date")["Profit Change (USD)"].sum().reset_index()
    basic_pnl_by_date = all_basic_df.groupby("Date")["Profit Change (USD)"].sum().reset_index()

    enhanced_pnl_by_date = enhanced_pnl_by_date.sort_values("Date").reset_index(drop=True)
    basic_pnl_by_date = basic_pnl_by_date.sort_values("Date").reset_index(drop=True)

    enhanced_pnl_by_date["Cumulative_EnhancedPnL"] = enhanced_pnl_by_date["Profit Change (USD)"].cumsum()
    basic_pnl_by_date["Cumulative_BasicPnL"] = basic_pnl_by_date["Profit Change (USD)"].cumsum()

    # Continuous date range
    full_date_range = pd.date_range(
        start=min(enhanced_pnl_by_date["Date"].min(), basic_pnl_by_date["Date"].min()),
        end=max(enhanced_pnl_by_date["Date"].max(), basic_pnl_by_date["Date"].max())
    )

    enhanced_pnl_by_date = enhanced_pnl_by_date.set_index("Date").reindex(full_date_range).fillna(0).rename_axis("Date").reset_index()
    enhanced_pnl_by_date["Cumulative_EnhancedPnL"] = enhanced_pnl_by_date["Profit Change (USD)"].cumsum()

    basic_pnl_by_date = basic_pnl_by_date.set_index("Date").reindex(full_date_range).fillna(0).rename_axis("Date").reset_index()
    basic_pnl_by_date["Cumulative_BasicPnL"] = basic_pnl_by_date["Profit Change (USD)"].cumsum()

    # Merge for plotting
    pnl_df = pd.merge(
        basic_pnl_by_date[["Date", "Cumulative_BasicPnL"]],
        enhanced_pnl_by_date[["Date", "Cumulative_EnhancedPnL"]],
        on="Date",
        how="outer"
    ).sort_values("Date").fillna(method="ffill").fillna(0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(pnl_df["Date"], pnl_df["Cumulative_BasicPnL"], label="Basic Strategy Cumulative P&L", linestyle="-")
    plt.plot(pnl_df["Date"], pnl_df["Cumulative_EnhancedPnL"], label="Enhanced Strategy Cumulative P&L", linestyle="-")
    plt.title("Cumulative P&L Over Time")
    plt.xlabel("Date")
    plt.ylabel("P&L (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final PnL Summary
    print("Final PnL by Asset:")
    for asset, pnl in final_pnl_summary.items():
        print(f"{asset}: Basic = ${pnl['Final Basic PnL']:,.2f}, Enhanced = ${pnl['Final Enhanced PnL']:,.2f}")

    total_pnl_basic = sum(p["Final Basic PnL"] for p in final_pnl_summary.values())
    total_pnl_enhanced = sum(p["Final Enhanced PnL"] for p in final_pnl_summary.values())
    print(f"Total Portfolio PnL — Basic: ${total_pnl_basic:,.2f}, Enhanced: ${total_pnl_enhanced:,.2f}")

    return pnl_df

cumulative_pnl_df = compute_final_pnl_with_plot(price_data_dict, pricing_currency)



def evaluate_ma_pairs(ma_pairs, price_data_dict, pricing_currency):
    def run_strategy_for_signal(signal_strategy):
        results = []
        for short, large in ma_pairs:
            total_basic_pnl = 0.0
            total_enhanced_pnl = 0.0

            for sheet_name, data in price_data_dict.items():
                asset_name = sheet_name.replace("DC ", "")
                try:
                    pnl_basic, _ = record_basic_strategy(
                        data.copy(), pricing_currency, asset_name, price_data_dict, short, large
                    )
                    pnl_enhanced, _ = record_enhanced_strategy(
                        data.copy(), signal_strategy, pricing_currency, asset_name, price_data_dict, short, large
                    )
                    total_basic_pnl += pnl_basic
                    total_enhanced_pnl += pnl_enhanced
                except Exception as e:
                    print(f"Error processing {asset_name} for MA ({short}/{large}): {e}")

            if total_basic_pnl != 0:
                percent_improvement = ((total_enhanced_pnl - total_basic_pnl) / abs(total_basic_pnl)) * 100
            else:
                percent_improvement = float('inf') if total_enhanced_pnl > 0 else float('-inf')

            results.append({
                "Short MA": short,
                "Long MA": large,
                "Total Basic PL": total_basic_pnl,
                "Total Enhanced PnL": total_enhanced_pnl,
                "% Improvement": percent_improvement,
                "$ Delta": total_enhanced_pnl - total_basic_pnl
            })
        return pd.DataFrame(results)

    def plot_absolute_delta(df, strategy_label, ma_pairs_ordered):
        df["MA Label"] = df.apply(lambda row: f"{int(row['Short MA'])}/{int(row['Long MA'])}", axis=1)
        ordered_labels = [f"{s}/{l}" for s, l in ma_pairs_ordered]
        df["MA Label"] = pd.Categorical(df["MA Label"], categories=ordered_labels, ordered=True)
        df_sorted = df.sort_values("MA Label")

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            df_sorted["MA Label"],
            df_sorted["$ Delta"],
            color='seagreen',
            edgecolor='black'
        )

        hyplabel= 0
        if strategy_label==1:
            hyplabel=1
        elif strategy_label ==0:
            hyplabel=2

        ax.set_title(f"Δ P&L (Enhanced − Basic) in USD (Hypothesis {hyplabel})", fontsize=14)
        ax.set_xlabel("MA Pair (Short/Long)", fontsize=12)
        ax.set_ylabel("P&L Difference ($)", fontsize=12)
        ax.axhline(0, color='black', linewidth=1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        for rect in bars:
            height = rect.get_height()
            ax.annotate(f'${height/1e6:.2f}M', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

        plt.tight_layout()
        plt.show()

    # Run evaluations
    df_strategy_1 = run_strategy_for_signal(signal_strategy=1)
    df_strategy_0 = run_strategy_for_signal(signal_strategy=0)
    ma_order = ma_pairs

    #DISPLAY STRATEGY 1
    print("MA Strategy Evaluation Summary (Signal Strategy = 1):")
    print(df_strategy_1.to_string(index=False))
    plot_absolute_delta(df_strategy_1, strategy_label=1, ma_pairs_ordered=ma_order)

    #DISPLAY STRATEGY 0
    print("MA Strategy Evaluation Summary (Signal Strategy = 0):")
    print(df_strategy_0.to_string(index=False))
    plot_absolute_delta(df_strategy_0, strategy_label=0, ma_pairs_ordered=ma_order)

    return df_strategy_1, df_strategy_0

ma_pairs_to_test = [(5, 13), (13, 25), (20, 50), (50, 100)]
df1, df0 = evaluate_ma_pairs(ma_pairs_to_test, price_data_dict, pricing_currency)
