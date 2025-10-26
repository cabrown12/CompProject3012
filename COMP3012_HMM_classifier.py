import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler

# Mapping for Stocks vs. FX
asset_currency_mapping = {
    "FTSE100": "FTSE/GBP",
    "S&P500": "S&P/USD",
    "Nasdaq100": "Nasdaq/USD",
    "Nikkei": "Nikkei/JPY",
    "Eurostoxx50": "Eurostoxx/EUR",
    "Gold": "Gold/USD",
    "Silver": "Silver/USD"
}

def load_and_clean_data(file_path, sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    start_row = 13
    date_col = 1
    iv_cols = [3, 4, 5, 6, 7]

    dates = pd.to_datetime(data.iloc[start_row:, date_col], errors="coerce")
    dates.name = "DATES"
    implied_vols = data.iloc[start_row:, iv_cols].apply(pd.to_numeric, errors='coerce')
    implied_vols.columns = ["25d put", "10d put", "ATM", "10d call", "25d call"]

    df = pd.concat([dates, implied_vols], axis=1)
    df = df.dropna(subset=["DATES"]).reset_index(drop=True)

    for col in implied_vols.columns:
        df[f"MA_{col}"] = df[col].rolling(window=30).mean()
        df[f"STD_{col}"] = df[col].rolling(window=30).std()

    sigma_posterior_fixed = 0.1
    for col in implied_vols.columns:
        df[f"STD_POST_{col}"] = sigma_posterior_fixed

    return df

def compute_kl_scores(data):
    kl_scores = []
    iv_columns = ["25d put", "10d put", "ATM", "10d call", "25d call"]
    sigma_posterior_fixed = 0.1

    for i in range(29, len(data)):
        total_kl = 0
        for col in iv_columns:
            mu_prior = data[f"MA_{col}"].iloc[i]
            mu_posterior = data[col].iloc[i]
            sigma_prior = data[f"STD_{col}"].iloc[i]
            sigma_posterior = sigma_posterior_fixed

            if pd.isna(mu_prior) or pd.isna(mu_posterior) or pd.isna(sigma_prior):
                continue

            kl = np.log(sigma_posterior / sigma_prior) \
                 + (sigma_prior**2 + (mu_prior - mu_posterior)**2) / (2 * sigma_posterior**2) \
                 - 0.5
            total_kl += kl

        if total_kl > 0:
            kl_scores.append((data["DATES"].iloc[i], total_kl))

    return kl_scores

def load_market_data(file_path, asset_currency_mapping):
    xls = pd.ExcelFile(file_path)
    all_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("DC")]

    stock_kl, fx_kl = [], []
    asset_data = {}

    for sheet_name in all_sheets:
        asset_name = sheet_name.replace("DC ", "")
        df = load_and_clean_data(file_path, sheet_name)

        if df.empty:
            print(f"No IV data for {asset_name}. Skipping")
            continue

        kl_scores = compute_kl_scores(df)
        if len(kl_scores) < 30:
            print(f"Not enough KL data for {asset_name}. Skipping")
            continue

        kl_df = pd.DataFrame(kl_scores, columns=["DATES", "KL"])
        kl_df["ASSET"] = asset_name

        if asset_name in asset_currency_mapping:
            stock_kl.extend(kl_df[["KL"]].values)
        else:
            fx_kl.extend(kl_df[["KL"]].values)

        asset_data[asset_name] = kl_df

    return np.array(stock_kl), np.array(fx_kl), asset_data

def train_hmm_for_market(iv_data, min_states=2, max_states=6, max_iter=5000):
    scaler = MinMaxScaler()

    if np.isnan(iv_data).sum() > 0:
        iv_data = np.nan_to_num(iv_data, nan=np.nanmean(iv_data, axis=0))

    iv_data_scaled = scaler.fit_transform(iv_data)
    criteria_results = []

    for n_states in range(min_states, max_states + 1):
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=max_iter,
            tol=1e-4,
            random_state=42
        )

        try:
            model.fit(iv_data_scaled)
            log_likelihood = model.score(iv_data_scaled)
        except Exception as e:
            print(f"HMM failed for {n_states} states: {e}")
            continue

        k = (n_states * (n_states - 1)) + (n_states - 1) + (2 * n_states * iv_data_scaled.shape[1])
        #all measure how well data is fittee versus complexity of model
        aic = -2 * log_likelihood + 2 * k                     # AIC: balances fit vs. complexity, favours predictive accuracy
        bic = -2 * log_likelihood + k * np.log(iv_data_scaled.shape[0])   # BIC: penalises complexity more than AIC, better for large samples
        hqc = -2 * log_likelihood + k * np.log(np.log(iv_data_scaled.shape[0]))  # HQC: softer penalty than BIC, stable for mid-sized data
        caic = -2 * log_likelihood + k * (np.log(iv_data_scaled.shape[0]) + 1)   # CAIC: even stricter than BIC, discourages overfitting


        criteria_results.append({
            'states': n_states,
            'model': model,
            'log_likelihood': log_likelihood,
            'AIC': aic,
            'BIC': bic,
            'HQC': hqc,
            'CAIC': caic
        })

    results_df = pd.DataFrame(criteria_results)

    if results_df.empty:
        raise ValueError("No HMM model was successfully trained!")

    results_df['Aggregate_Criterion'] = results_df[['AIC', 'BIC', 'HQC', 'CAIC']].mean(axis=1)
    best_row = results_df.loc[results_df['Aggregate_Criterion'].idxmin()]

    return best_row['model'], int(best_row['states']), scaler

def apply_hmm_to_market(asset_data, model, scaler):
    surprises = {}
    for asset_name, df in asset_data.items():
        kl_scaled = scaler.transform(df[["KL"]])
        df["REGIME"] = model.predict(kl_scaled)

        df["UNEXPECTED_SHIFT"] = False
        transmat = model.transmat_
        for i in range(1, len(df)):
            old_state = df.loc[i - 1, "REGIME"]
            new_state = df.loc[i, "REGIME"]
            if old_state != new_state:
                if transmat[old_state, new_state] < 0.05:
                    df.loc[i, "UNEXPECTED_SHIFT"] = True

        surprise_dates = df.loc[df["UNEXPECTED_SHIFT"], "DATES"].tolist()
        surprises[asset_name] = sorted(set(surprise_dates))

        asset_data[asset_name] = df
    return asset_data, surprises

def train_hmm_model(file_path, asset_currency_mapping):
    print("Loading Data and Training HMMs on Bayesian Surprise")
    stock_kl, fx_kl, asset_data = load_market_data(file_path, asset_currency_mapping)

    print("Training Stock HMM.")
    stock_hmm, stock_states, stock_scaler = train_hmm_for_market(stock_kl)
    print(f"✅ Stock HMM trained with {stock_states} hidden states.")

    print("\Training FX HMM")
    fx_hmm, fx_states, fx_scaler = train_hmm_for_market(fx_kl)
    print(f"FX HMM trained with {fx_states} hidden states.")

    stock_assets = {k: v for k, v in asset_data.items() if k in asset_currency_mapping}
    fx_assets = {k: v for k, v in asset_data.items() if k not in asset_currency_mapping}

    stock_assets, stock_surprises = apply_hmm_to_market(stock_assets, stock_hmm, stock_scaler)
    fx_assets, fx_surprises = apply_hmm_to_market(fx_assets, fx_hmm, fx_scaler)

    surprises = {**stock_surprises, **fx_surprises}
    return surprises

if __name__ == "__main__":
    file_path = '/Users/caelenbrown/Downloads/log_transformed_output.xlsx'
    surprises = train_hmm_model(file_path, asset_currency_mapping)

    print("Total Market Surprises:")
    for asset, dates in surprises.items():
        print(f"{asset}: {len(dates)} surprises detected ⇒ {dates}")

    print("Total Market Surprises:")
    for asset, dates in surprises.items():
        print(f"{asset} produced {len(dates)} 'rare surprise' transitions.")

    total_surprise_count = sum(len(dates) for dates in surprises.values())
    print(f"Total number of rare surprise transitions across all assets: {total_surprise_count}")