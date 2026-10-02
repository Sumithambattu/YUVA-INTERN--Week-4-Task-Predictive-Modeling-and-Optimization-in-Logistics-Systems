import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import linprog

rng = np.random.default_rng(42)
N = 5000
# ---------- 1. Simulate data ----------
df = pd.DataFrame({
    "distance_km": rng.gamma(3.0, 5.0, N).clip(1, 80),
    "package_weight_kg": rng.gamma(2.0, 4.0, N).clip(0.2, 50),
    "num_stops": rng.integers(1, 16, N),
    "hour_of_day": rng.integers(8, 21, N),
    "day_of_week": rng.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], N),
    "weather": rng.choice(["Clear","Rain","Heavy Rain"], N, p=[0.6,0.3,0.1]),
    "traffic_index": rng.uniform(1, 10, N),
    "vehicle_type": rng.choice(["Bike","Van","Truck"], N, p=[0.3,0.5,0.2]),
    "warehouse_zone": rng.choice(["Zone A","Zone B","Zone C","Zone D","Zone E"], N),
})
peak = df.hour_of_day.isin([9,10,17,18,19]).astype(int)
speed = df.vehicle_type.map({"Bike":18,"Van":28,"Truck":24})
wx = df.weather.map({"Clear":0,"Rain":6,"Heavy Rain":15})
wk = df.day_of_week.isin(["Sat","Sun"]).astype(int)
zone_pen = df.warehouse_zone.map({"Zone A":0,"Zone B":3,"Zone C":6,"Zone D":2,"Zone E":9})
t = (df.distance_km/speed*60 + df.num_stops*4.5 + df.package_weight_kg*0.35
     + df.traffic_index*2.2 + peak*8 + wx + wk*-3 + zone_pen
     + 0.015*df.distance_km*df.traffic_index + rng.normal(0, 4, N))
df["delivery_time_min"] = t.clip(10).round(1)
# inject missing values / outliers to demonstrate cleaning
df.loc[rng.choice(N, 100, replace=False), "traffic_index"] = np.nan
df.loc[rng.choice(N, 50, replace=False), "package_weight_kg"] = np.nan
df.to_csv("logistics_deliveries.csv", index=False)

summary = df.describe().round(2)
summary.to_csv("summary.csv")
print(df.isna().sum()); print(summary)

# ---------- 2. Prep ----------
df["traffic_index"] = df.traffic_index.fillna(df.traffic_index.median())
df["package_weight_kg"] = df.package_weight_kg.fillna(df.package_weight_kg.median())
df["is_peak"] = df.hour_of_day.isin([9,10,17,18,19]).astype(int)
df["is_weekend"] = df.day_of_week.isin(["Sat","Sun"]).astype(int)
df["dist_x_traffic"] = df.distance_km*df.traffic_index
y = df.delivery_time_min; X = df.drop(columns="delivery_time_min")
cat = ["day_of_week","weather","vehicle_type","warehouse_zone"]
num = [c for c in X.columns if c not in cat]
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
 "Linear Regression": LinearRegression(),
 "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
 "Random Forest": RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42),
 "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
kf = KFold(5, shuffle=True, random_state=42)
rows = []
for n, m in models.items():
    p = Pipeline([("pre", pre), ("m", m)])
    cv = -cross_val_score(p, Xtr, ytr, cv=kf, scoring="neg_root_mean_squared_error")
    p.fit(Xtr, ytr); pr = p.predict(Xte)
    rows.append([n, cv.mean(), cv.std(), np.sqrt(mean_squared_error(yte, pr)),
                 mean_absolute_error(yte, pr), r2_score(yte, pr)])
base = pd.DataFrame(rows, columns=["Model","CV RMSE","CV Std","Test RMSE","Test MAE","Test R2"]).round(3)
print(base)

# ---------- 3. Tuning ----------
gb = Pipeline([("pre", pre), ("m", GradientBoostingRegressor(random_state=42))])
grid = {"m__n_estimators":[100,200,300,400], "m__learning_rate":[0.03,0.05,0.1],
        "m__max_depth":[2,3,4], "m__subsample":[0.7,0.85,1.0], "m__min_samples_leaf":[1,5,10]}
rs = RandomizedSearchCV(gb, grid, n_iter=20, cv=kf, scoring="neg_root_mean_squared_error",
                        random_state=42, n_jobs=-1)
rs.fit(Xtr, ytr)
best = rs.best_estimator_; pr = best.predict(Xte)
tuned = dict(rmse=np.sqrt(mean_squared_error(yte, pr)), mae=mean_absolute_error(yte, pr), r2=r2_score(yte, pr),
             cv=-rs.best_score_)
print(rs.best_params_, tuned)

# feature importance (permutation)
from sklearn.inspection import permutation_importance
pi = permutation_importance(best, Xte, yte, n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
print(imp.round(3))

# ---------- charts ----------
plt.rcParams.update({"font.size":10})
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].scatter(yte, pr, s=6, alpha=.4, color="#1F4E79"); lo,hi = yte.min(), yte.max()
ax[0].plot([lo,hi],[lo,hi],"r--"); ax[0].set_xlabel("Actual (min)"); ax[0].set_ylabel("Predicted (min)")
ax[0].set_title("Actual vs Predicted")
res = yte-pr; ax[1].hist(res, bins=40, color="#2E75B6"); ax[1].set_title("Residual Distribution")
ax[1].set_xlabel("Error (min)")
plt.tight_layout(); plt.savefig("fig_eval.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(7,4))
imp.head(8)[::-1].plot.barh(ax=ax, color="#1F4E79"); ax.set_title("Permutation Feature Importance (top 8)")
ax.set_xlabel("Mean increase in MSE when shuffled"); plt.tight_layout(); plt.savefig("fig_imp.png", dpi=160); plt.close()

# ---------- 4. Optimization ----------
# 4a. Predicted time by hour -> staffing/peak analysis
Xte2 = Xte.copy(); Xte2["pred"] = pr
hourly = Xte2.groupby("hour_of_day").pred.mean().round(1)
print(hourly)

# 4b. LP: allocate zone demand to depots minimising predicted delivery minutes
zones = ["Zone A","Zone B","Zone C","Zone D","Zone E"]
depots = ["Depot North","Depot Central","Depot South"]
demand = np.array([320, 260, 410, 180, 230])          # shipments / day
capacity = np.array([600, 500, 450])                  # shipments / day per depot
# average predicted time of a typical shipment from each zone, scaled by depot-zone distance factor
zone_base = Xte2.groupby("warehouse_zone").pred.mean().reindex(zones).values
dist_factor = np.array([[0.9,1.1,1.4],[1.0,0.9,1.2],[1.3,1.0,0.9],[1.2,0.95,1.1],[1.5,1.1,0.8]])
cost = zone_base[:,None]*dist_factor                  # minutes per shipment
c = cost.flatten()
A_eq = np.zeros((5,15)); 
for i in range(5): A_eq[i, i*3:(i+1)*3] = 1
A_ub = np.zeros((3,15))
for j in range(3): A_ub[j, j::3] = 1
lp = linprog(c, A_ub=A_ub, b_ub=capacity, A_eq=A_eq, b_eq=demand, bounds=(0,None), method="highs")
alloc = lp.x.reshape(5,3).round(0)
# baseline: nearest-by-label (each zone to its lowest-cost depot ignoring capacity -> then proportional) vs equal split
base_cost = (demand[:,None]/3*cost).sum()
opt_cost = lp.fun
print(alloc, opt_cost, base_cost, 1-opt_cost/base_cost)

# 4c. Route sequencing: nearest neighbour + 2-opt on a 12-stop route
rs2 = np.random.default_rng(7); pts = rs2.uniform(0,25,(12,2)); pts[0]=[12,12]
D = np.linalg.norm(pts[:,None]-pts[None],axis=2)
def L(r): return sum(D[r[i],r[(i+1)%len(r)]] for i in range(len(r)))
rand_route = list(rs2.permutation(12))
nn=[0]; left=set(range(1,12))
while left:
    k=min(left,key=lambda j:D[nn[-1],j]); nn.append(k); left.remove(k)
r=nn[:]; imp_=True
while imp_:
    imp_=False
    for i in range(1,len(r)-1):
        for j in range(i+1,len(r)):
            nr=r[:i]+r[i:j+1][::-1]+r[j+1:]
            if L(nr)<L(r)-1e-9: r=nr; imp_=True
print("route", L(rand_route), L(nn), L(r))
fig, ax = plt.subplots(1,2, figsize=(10,4))
for a,(rt,tt) in zip(ax,[(rand_route,"Unoptimised (as received)"),(r,"Optimised (NN + 2-opt)")]):
    xs=[pts[i,0] for i in rt+[rt[0]]]; ys=[pts[i,1] for i in rt+[rt[0]]]
    a.plot(xs,ys,"-o",color="#1F4E79",ms=5); a.plot(*pts[0],"r*",ms=14)
    a.set_title(f"{tt}: {L(rt):.1f} km")
plt.tight_layout(); plt.savefig("fig_route.png", dpi=160); plt.close()

out = dict(base=base.to_dict("records"), best_params=rs.best_params_, tuned=tuned,
           imp=imp.round(4).to_dict(), hourly=hourly.to_dict(), alloc=alloc.tolist(),
           cost=cost.round(1).tolist(), opt_cost=float(opt_cost), base_cost=float(base_cost),
           route=[L(rand_route),L(nn),L(r)], zone_base=zone_base.round(1).tolist(),
           demand=demand.tolist(), capacity=capacity.tolist(),
           desc=summary.loc[["mean","std","min","max"],["distance_km","package_weight_kg","num_stops","traffic_index","delivery_time_min"]].to_dict())
json.dump(out, open("results.json","w"), indent=1, default=float)
