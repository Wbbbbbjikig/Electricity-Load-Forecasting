# -*- coding: utf-8 -*-
# ============================================================
# forecast.py — 电力负荷时序预测:特征工程 + 多模型对比 + 评估
# ------------------------------------------------------------
# 完整流程(对应 JD 要求3:特征提取→模型建立→效果评估→优化迭代):
#   1. 时序特征工程:日历特征 + 滞后特征(lag)+ 滑动统计 + 温度
#   2. 时序划分:按时间切分训练/测试(绝不能随机打乱!)
#   3. 三级模型对比:朴素基线 → 线性回归 → LightGBM
#   4. 评估:MAE / RMSE / MAPE
#   5. 可视化:预测 vs 实际 + 特征重要性
#
# 运行: python3 forecast.py  ->  控制台指标 + 两张图
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# ---------- 中文字体 ----------
def pick_cn_font():
    for c in ["PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS", "Hiragino Sans GB"]:
        if c in {f.name for f in font_manager.fontManager.ttflist}:
            return c
    return None
cn = pick_cn_font()
if cn:
    plt.rcParams["font.sans-serif"] = [cn]
    plt.rcParams["axes.unicode_minus"] = False
def L(zh, en): return zh if cn else en

# ---------- 1. 读数据 ----------
df = pd.read_csv("data/load.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ---------- 2. 特征工程 ----------
def build_features(df):
    d = df.copy()
    ts = d["timestamp"]
    # 日历特征
    d["hour"] = ts.dt.hour
    d["dayofweek"] = ts.dt.dayofweek
    d["month"] = ts.dt.month
    d["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    # 周期性编码(把小时/星期变成正弦余弦,让模型理解"23点和0点相邻")
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    # 滞后特征:昨天同时刻、一周前同时刻、上一小时
    d["lag_1h"] = d["load"].shift(1)
    d["lag_24h"] = d["load"].shift(24)
    d["lag_168h"] = d["load"].shift(168)     # 一周 = 168 小时
    # 滑动统计:过去 24h 的均值/标准差(反映近期水平与波动)
    d["roll_mean_24h"] = d["load"].shift(1).rolling(24).mean()
    d["roll_std_24h"] = d["load"].shift(1).rolling(24).std()
    # 外生变量:温度(气象数据)
    # temperature 已在列中
    d = d.dropna().reset_index(drop=True)     # 丢掉因 lag 产生的前 168 行 NaN
    return d

data = build_features(df)

FEATURES = ["hour", "dayofweek", "month", "is_weekend", "hour_sin", "hour_cos",
            "lag_1h", "lag_24h", "lag_168h", "roll_mean_24h", "roll_std_24h",
            "temperature"]
TARGET = "load"

# ---------- 3. 时序划分(最后 20% 作测试,严格按时间,不打乱) ----------
split = int(len(data) * 0.8)
train, test = data.iloc[:split], data.iloc[split:]
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
print(f"训练集 {len(train)} 行,测试集 {len(test)} 行(按时间切分)\n")

# ---------- 4. 评估函数 ----------
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name:<16} MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:5.2f}%")
    return {"name": name, "mae": mae, "rmse": rmse, "mape": mape}

results = []

# ---- 模型1:朴素基线(用"昨天同时刻"预测今天) ----
pred_naive = X_test["lag_24h"].values
results.append(evaluate("朴素基线(lag24)", y_test.values, pred_naive))

# ---- 模型2:线性回归 ----
lr = LinearRegression().fit(X_train, y_train)
pred_lr = lr.predict(X_test)
results.append(evaluate("线性回归", y_test.values, pred_lr))

# ---- 模型3:LightGBM ----
lgb_model = lgb.LGBMRegressor(
    n_estimators=600, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
pred_lgb = lgb_model.predict(X_test)
results.append(evaluate("LightGBM", y_test.values, pred_lgb))

# ---- 汇总:相比基线提升 ----
base_mae = results[0]["mae"]
best = min(results, key=lambda r: r["mae"])
print(f"\n最优模型: {best['name']},MAE 相比朴素基线降低 "
      f"{(base_mae - best['mae']) / base_mae * 100:.1f}%")

# ---------- 5. 可视化 ----------
# 图1:预测 vs 实际(取测试集前 7 天 = 168 小时,看得清)
fig, ax = plt.subplots(figsize=(11, 4.2))
n = 168
t = test["timestamp"].values[:n]
ax.plot(t, y_test.values[:n], label=L("实际负荷", "Actual"), color="#111827", lw=1.8)
ax.plot(t, pred_lgb[:n], label=L("LightGBM 预测", "LightGBM"), color="#f59e0b", lw=1.6)
ax.plot(t, pred_naive[:n], label=L("朴素基线", "Naive"), color="#94a3b8", lw=1.0, ls="--")
ax.set_title(L("电力负荷预测:测试集前 7 天", "Load Forecast: first 7 days of test set"))
ax.set_ylabel(L("负荷", "Load"))
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_plot.png", dpi=150, bbox_inches="tight")
print("\n已保存 forecast_plot.png")

# 图2:特征重要性(体现"哪些因素驱动负荷")
fig, ax = plt.subplots(figsize=(7, 4.5))
imp = pd.Series(lgb_model.feature_importances_, index=FEATURES).sort_values()
ax.barh(imp.index, imp.values, color="#3b82f6")
ax.set_title(L("LightGBM 特征重要性", "LightGBM Feature Importance"))
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("已保存 feature_importance.png")

# 保存指标供文档引用
pd.DataFrame(results).to_csv("metrics.csv", index=False)
print("已保存 metrics.csv")
