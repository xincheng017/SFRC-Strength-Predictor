import numpy as np
import pandas as pd
import pickle
import os
from tkinter import *
import customtkinter as ctk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor

CLR_BG          = "#F5F7FA"
CLR_CARD        = "#FFFFFF"
CLR_HEADER_BG   = "#1A3A5C"
CLR_ACCENT      = "#E07B39"
CLR_ACCENT2     = "#2E7D9F"
CLR_BORDER      = "#D0D8E4"
CLR_TEXT_DARK   = "#1A2B3C"
CLR_TEXT_MID    = "#4A5568"
CLR_TEXT_LIGHT  = "#8A9BB0"
CLR_SUCCESS     = "#2D6A4F"
CLR_WARNING     = "#C05C1A"
CLR_CS_BG       = "#EBF4FF"
CLR_STS_BG      = "#FFF3E8"
CLR_BTN_PRED    = "#1A3A5C"
CLR_BTN_CLEAR   = "#8A9BB0"
CLR_ENTRY_BG    = "#F0F4F8"
CLR_SECTION     = "#E8EEF5"

FEATURES = [
    'Water [kg]', 'Cement [kg]', 'Sand [kg]', 'CA [kg]',
    'smax [mm]',  'SP [kg]',    'Vf [%]',    'df [mm]', 'Lf [mm]',
]
FIBER_CATEGORIES = [
    'No fiber', 'Hooked', 'Crimped',
    'Mill-cut', 'Straight smooth', 'Chopped with butt ends',
]

BASE_DIR  = r"E:\运行界面"

def _resolve_path(filename):
    candidates = []

    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass

    candidates.append(os.getcwd())

    candidates.append(BASE_DIR)

    for d in candidates:
        full = os.path.join(d, filename)
        if os.path.exists(full):
            return full

    return os.path.join(BASE_DIR, filename)

def load_models():
    pkl_path  = _resolve_path("gui_models.pkl")
    data_path = _resolve_path("SFRC_cleaned_datasets_v2.xlsx")

    if os.path.exists(pkl_path):
        print(f"[INFO] 加载预训练模型: {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"\n\n未找到预训练模型，也未找到数据文件。\n"
            f"请执行以下任一操作：\n"
            f"  A. 将 gui_models.pkl 放在脚本同目录（推荐）\n"
            f"  B. 将 SFRC_cleaned_datasets_v2.xlsx 放在脚本同目录\n"
            f"  C. 修改代码中的 BASE_DIR 为您的数据文件夹路径\n"
            f"\n当前搜索路径：{data_path}"
        )

    print(f"[INFO] 未找到预训练模型，从数据集重新训练: {data_path}")
    pkl       = pkl_path

    def build(sheet, col, seed):
        df  = pd.read_excel(data_path, sheet_name=sheet).reset_index(drop=True)
        y   = df[col].values
        yb  = pd.qcut(y, q=3, labels=[0,1,2], duplicates='drop').astype(int)
        sk  = yb*10 + df['Fiber type'].map(
            lambda t: 0 if t=='No fiber' else (1 if t=='Hooked' else 2)).values
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr, _ = next(sss.split(df, sk))
        ohe = pd.get_dummies(
            pd.Categorical(df['Fiber type'], categories=FIBER_CATEGORIES), prefix='Fiber')
        Xa  = np.hstack([df[FEATURES].values, ohe.values])
        sc  = StandardScaler(); n = len(FEATURES)
        Xt  = np.hstack([sc.fit_transform(Xa[tr,:n]), Xa[tr,n:]])
        return Xt, y[tr], sc

    Xc, yc, sc_c = build('CS_cleaned_v2', 'fcy [MPa]', 7)
    mc = XGBRegressor(n_estimators=900, max_depth=4, learning_rate=0.0312,
        subsample=0.9996, colsample_bytree=0.9747, reg_alpha=4.735,
        reg_lambda=0.2128, min_child_weight=5, random_state=42, verbosity=0)
    mc.fit(Xc, yc)

    Xs, ys, sc_s = build('STS_cleaned_v2', 'fst [MPa]', 3)
    ms = XGBRegressor(n_estimators=650, learning_rate=0.049139,
        reg_lambda=1.848707, subsample=0.76836, colsample_bytree=0.763578,
        max_depth=5, random_state=42, verbosity=0)
    ms.fit(Xs, ys)

    pkg = {
        'cs':  {'model': mc, 'scaler': sc_c},
        'sts': {'model': ms, 'scaler': sc_s},
    }
    with open(pkl, 'wb') as f:
        pickle.dump(pkg, f)
    return pkg

pkg = load_models()
catb_cs  = pkg['cs']['model']
catb_sts = pkg['sts']['model']
sc_cs    = pkg['cs']['scaler']
sc_sts   = pkg['sts']['scaler']

def predict():
    try:
        W  = float(entry_water.get())
        C  = float(entry_cement.get())
        S  = float(entry_sand.get())
        CA = float(entry_ca.get())
        SM = float(entry_smax.get())
        SP = float(entry_sp.get())
        VF = float(entry_vf.get())
        DF = float(entry_df.get())
        LF = float(entry_lf.get())
        FT = fiber_var.get()

        ohe       = [1 if cat == FT else 0 for cat in FIBER_CATEGORIES]
        feat_vals = [W, C, S, CA, SM, SP, VF, DF, LF]

        X_cs     = np.hstack([sc_cs.transform([feat_vals]),  [ohe]])
        X_sts    = np.hstack([sc_sts.transform([feat_vals]), [ohe]])
        pred_cs  = catb_cs.predict(X_cs)[0]
        pred_sts = catb_sts.predict(X_sts)[0]

        cs_value_label.configure(
            text=f"{pred_cs:.2f}",
            text_color=CLR_ACCENT2
        )
        cs_unit_label.configure(text="MPa", text_color=CLR_TEXT_MID)
        cs_bar_canvas.delete("all")
        bar_w = min(int((pred_cs / 100) * 260), 260)
        cs_bar_canvas.create_rectangle(0, 0, 260, 12, fill="#E2EAF2", outline="")
        cs_bar_canvas.create_rectangle(0, 0, bar_w, 12, fill=CLR_ACCENT2, outline="")

        sts_value_label.configure(
            text=f"{pred_sts:.3f}",
            text_color=CLR_ACCENT
        )
        sts_unit_label.configure(text="MPa", text_color=CLR_TEXT_MID)
        sts_bar_canvas.delete("all")
        bar_w2 = min(int((pred_sts / 12) * 260), 260)
        sts_bar_canvas.create_rectangle(0, 0, 260, 12, fill="#F5E8DC", outline="")
        sts_bar_canvas.create_rectangle(0, 0, bar_w2, 12, fill=CLR_ACCENT, outline="")

        wc = W / C if C > 0 else 0
        if wc > 0.55:
            wc_label.configure(
                text=f"  ⚠  W/C = {wc:.3f}  —  High water-to-cement ratio detected",
                text_color=CLR_WARNING,
                font=ctk.CTkFont(family="Arial", size=11, weight="bold")
            )
        else:
            wc_label.configure(
                text=f"  ✓  W/C = {wc:.3f}  —  Within normal range",
                text_color=CLR_SUCCESS,
                font=ctk.CTkFont(family="Arial", size=11, weight="bold")
            )

        status_label.configure(
            text="● Prediction completed successfully",
            text_color=CLR_SUCCESS
        )

    except ValueError:
        status_label.configure(
            text="⚠  Please enter valid numeric values in all fields.",
            text_color=CLR_WARNING
        )
        cs_value_label.configure(text="—", text_color=CLR_TEXT_LIGHT)
        sts_value_label.configure(text="—", text_color=CLR_TEXT_LIGHT)

def clear_all():
    for e in all_entries:
        e.delete(0, END)
    cs_value_label.configure(text="—", text_color=CLR_TEXT_LIGHT)
    cs_unit_label.configure(text="")
    sts_value_label.configure(text="—", text_color=CLR_TEXT_LIGHT)
    sts_unit_label.configure(text="")
    wc_label.configure(text="")
    status_label.configure(text="")
    cs_bar_canvas.delete("all")
    cs_bar_canvas.create_rectangle(0, 0, 260, 12, fill="#E2EAF2", outline="")
    sts_bar_canvas.delete("all")
    sts_bar_canvas.create_rectangle(0, 0, 260, 12, fill="#F5E8DC", outline="")
    fiber_var.set("Hooked")

def make_entry(parent, label_text, row, col, default="", width=115):
    lbl = ctk.CTkLabel(
        parent,
        text=label_text,
        font=ctk.CTkFont(family="Arial", size=11),
        text_color=CLR_TEXT_MID,
        fg_color="transparent",
        anchor="w"
    )
    lbl.grid(row=row*2,   column=col, padx=(12,4), pady=(6,1), sticky="w")

    e = ctk.CTkEntry(
        parent,
        width=width,
        height=32,
        font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
        fg_color=CLR_ENTRY_BG,
        border_color=CLR_BORDER,
        border_width=1,
        text_color=CLR_TEXT_DARK,
        corner_radius=6
    )
    e.grid(row=row*2+1, column=col, padx=(12,4), pady=(0,6), sticky="w")
    if default:
        e.insert(0, default)
    return e

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("SFRC Strength Predictor")
app.geometry("1020x760")
app.resizable(True, True)
app.configure(fg_color=CLR_BG)

header_frame = ctk.CTkFrame(
    app, fg_color=CLR_HEADER_BG,
    corner_radius=0, height=88
)
header_frame.pack(fill="x", padx=0, pady=0)
header_frame.pack_propagate(False)

title_label = ctk.CTkLabel(
    header_frame,
    text="Steel Fiber Reinforced Concrete  |  Strength Predictor",
    font=ctk.CTkFont(family="Times New Roman", size=26, weight="bold"),
    text_color="#FFFFFF",
    fg_color="transparent"
)
title_label.pack(side="left", padx=28, pady=(18, 4))

badge_frame = ctk.CTkFrame(
    header_frame, fg_color="#0F2640",
    corner_radius=8
)
badge_frame.pack(side="right", padx=24, pady=20)
ctk.CTkLabel(
    badge_frame,
    text="  BO-XGBoost  ·  CS R²=0.9206  ·  STS R²=0.8928  ",
    font=ctk.CTkFont(family="Arial", size=11),
    text_color="#A8C8E8",
    fg_color="transparent"
).pack(padx=6, pady=4)

body_frame = ctk.CTkFrame(app, fg_color="transparent")
body_frame.pack(fill="both", expand=True, padx=20, pady=14)

left_panel = ctk.CTkFrame(
    body_frame,
    fg_color=CLR_CARD,
    corner_radius=12,
    border_width=1,
    border_color=CLR_BORDER
)
left_panel.pack(side="left", fill="both", expand=True, padx=(0,10))

sec1_bar = ctk.CTkFrame(left_panel, fg_color=CLR_SECTION, corner_radius=0, height=32)
sec1_bar.pack(fill="x", padx=0, pady=(0,4))
sec1_bar.pack_propagate(False)
ctk.CTkLabel(
    sec1_bar,
    text="  ▌  Concrete Composition",
    font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
    text_color=CLR_HEADER_BG,
    fg_color="transparent",
    anchor="w"
).pack(side="left", fill="y", padx=4, pady=4)

grid1 = ctk.CTkFrame(left_panel, fg_color="transparent")
grid1.pack(fill="x", padx=4, pady=2)

entry_water  = make_entry(grid1, "Water content  (kg/m³)",           0, 0, "180")
entry_cement = make_entry(grid1, "Cement content  (kg/m³)",          0, 1, "450")
entry_sand   = make_entry(grid1, "Sand content  (kg/m³)",            0, 2, "700")
entry_ca     = make_entry(grid1, "Coarse aggregate  (kg/m³)",        0, 3, "1000")
entry_smax   = make_entry(grid1, "Max. aggregate size  (mm)",        1, 0, "20")
entry_sp     = make_entry(grid1, "Superplasticizer  (kg/m³)",        1, 1, "5")

ctk.CTkFrame(left_panel, fg_color=CLR_BORDER, height=1).pack(fill="x", padx=14, pady=6)

sec2_bar = ctk.CTkFrame(left_panel, fg_color=CLR_SECTION, corner_radius=0, height=32)
sec2_bar.pack(fill="x", padx=0, pady=(0,4))
sec2_bar.pack_propagate(False)
ctk.CTkLabel(
    sec2_bar,
    text="  ▌  Fiber Configuration",
    font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
    text_color=CLR_HEADER_BG,
    fg_color="transparent",
    anchor="w"
).pack(side="left", fill="y", padx=4, pady=4)

grid2 = ctk.CTkFrame(left_panel, fg_color="transparent")
grid2.pack(fill="x", padx=4, pady=2)

entry_vf = make_entry(grid2, "Fiber volume fraction  (%)",   0, 0, "1.5")
entry_df = make_entry(grid2, "Fiber diameter  (mm)",         0, 1, "0.55")
entry_lf = make_entry(grid2, "Fiber length  (mm)",           0, 2, "35")

lbl_ft = ctk.CTkLabel(
    grid2, text="Fiber type",
    font=ctk.CTkFont(family="Arial", size=11),
    text_color=CLR_TEXT_MID, fg_color="transparent", anchor="w"
)
lbl_ft.grid(row=0, column=3, padx=(12,4), pady=(6,1), sticky="w")

fiber_var = ctk.StringVar(value="Hooked")
fiber_menu = ctk.CTkOptionMenu(
    grid2,
    variable=fiber_var,
    values=FIBER_CATEGORIES,
    width=148,
    height=32,
    font=ctk.CTkFont(family="Arial", size=12),
    fg_color=CLR_ENTRY_BG,
    button_color=CLR_ACCENT2,
    button_hover_color="#1A6080",
    text_color=CLR_TEXT_DARK,
    dropdown_fg_color=CLR_CARD,
    dropdown_text_color=CLR_TEXT_DARK,
    dropdown_hover_color=CLR_SECTION,
    corner_radius=6
)
fiber_menu.grid(row=1, column=3, padx=(12,4), pady=(0,6), sticky="w")

hint = ctk.CTkLabel(
    left_panel,
    text="  Tip: Set Vf = 0, df = 0, Lf = 0 for plain concrete (no fiber) mixes",
    font=ctk.CTkFont(family="Arial", size=10),
    text_color=CLR_TEXT_LIGHT,
    fg_color="transparent",
    anchor="w"
)
hint.pack(fill="x", padx=16, pady=(0,4))

ctk.CTkFrame(left_panel, fg_color=CLR_BORDER, height=1).pack(fill="x", padx=14, pady=6)

btn_row = ctk.CTkFrame(left_panel, fg_color="transparent")
btn_row.pack(pady=10, padx=16, fill="x")

predict_btn = ctk.CTkButton(
    btn_row,
    text="  ▶   Predict Strength",
    font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
    command=predict,
    width=240, height=44,
    fg_color=CLR_BTN_PRED,
    hover_color="#0F2640",
    text_color="#FFFFFF",
    corner_radius=8,
    border_width=0
)
predict_btn.pack(side="left", padx=(0,12))

clear_btn = ctk.CTkButton(
    btn_row,
    text="✕  Clear",
    font=ctk.CTkFont(family="Arial", size=14),
    command=clear_all,
    width=110, height=44,
    fg_color="transparent",
    hover_color=CLR_SECTION,
    text_color=CLR_TEXT_MID,
    corner_radius=8,
    border_width=1,
    border_color=CLR_BORDER
)
clear_btn.pack(side="left")

status_label = ctk.CTkLabel(
    left_panel, text="",
    font=ctk.CTkFont(family="Arial", size=11),
    fg_color="transparent",
    anchor="w"
)
status_label.pack(fill="x", padx=16, pady=(0,8))

right_panel = ctk.CTkFrame(
    body_frame,
    fg_color=CLR_CARD,
    corner_radius=12,
    border_width=1,
    border_color=CLR_BORDER,
    width=310
)
right_panel.pack(side="right", fill="y", padx=(0,0))
right_panel.pack_propagate(False)

res_header = ctk.CTkFrame(right_panel, fg_color=CLR_HEADER_BG, corner_radius=0, height=40)
res_header.pack(fill="x")
res_header.pack_propagate(False)
ctk.CTkLabel(
    res_header,
    text="Prediction Results",
    font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
    text_color="#FFFFFF",
    fg_color="transparent"
).pack(expand=True)

cs_card = ctk.CTkFrame(
    right_panel,
    fg_color=CLR_CS_BG,
    corner_radius=10,
    border_width=1,
    border_color="#B8D4EE"
)
cs_card.pack(fill="x", padx=14, pady=(18,8))

ctk.CTkLabel(
    cs_card,
    text="Compressive Strength  (CS)",
    font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
    text_color=CLR_ACCENT2,
    fg_color="transparent",
    anchor="w"
).pack(fill="x", padx=14, pady=(12,2))

cs_val_row = ctk.CTkFrame(cs_card, fg_color="transparent")
cs_val_row.pack(fill="x", padx=14, pady=(0,4))

cs_value_label = ctk.CTkLabel(
    cs_val_row,
    text="—",
    font=ctk.CTkFont(family="Times New Roman", size=44, weight="bold"),
    text_color=CLR_TEXT_LIGHT,
    fg_color="transparent"
)
cs_value_label.pack(side="left")

cs_unit_label = ctk.CTkLabel(
    cs_val_row,
    text="",
    font=ctk.CTkFont(family="Arial", size=16),
    text_color=CLR_TEXT_MID,
    fg_color="transparent"
)
cs_unit_label.pack(side="left", padx=(6,0), pady=(14,0))

from tkinter import Canvas
cs_bar_frame = ctk.CTkFrame(cs_card, fg_color="transparent")
cs_bar_frame.pack(fill="x", padx=14, pady=(0,12))
cs_bar_canvas = Canvas(
    cs_bar_frame, width=260, height=12,
    bg=CLR_CS_BG, highlightthickness=0
)
cs_bar_canvas.pack(anchor="w")
cs_bar_canvas.create_rectangle(0, 0, 260, 12, fill="#E2EAF2", outline="")

sts_card = ctk.CTkFrame(
    right_panel,
    fg_color=CLR_STS_BG,
    corner_radius=10,
    border_width=1,
    border_color="#F0C8A0"
)
sts_card.pack(fill="x", padx=14, pady=(0,8))

ctk.CTkLabel(
    sts_card,
    text="Splitting Tensile Strength  (STS)",
    font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
    text_color=CLR_ACCENT,
    fg_color="transparent",
    anchor="w"
).pack(fill="x", padx=14, pady=(12,2))

sts_val_row = ctk.CTkFrame(sts_card, fg_color="transparent")
sts_val_row.pack(fill="x", padx=14, pady=(0,4))

sts_value_label = ctk.CTkLabel(
    sts_val_row,
    text="—",
    font=ctk.CTkFont(family="Times New Roman", size=44, weight="bold"),
    text_color=CLR_TEXT_LIGHT,
    fg_color="transparent"
)
sts_value_label.pack(side="left")

sts_unit_label = ctk.CTkLabel(
    sts_val_row,
    text="",
    font=ctk.CTkFont(family="Arial", size=16),
    text_color=CLR_TEXT_MID,
    fg_color="transparent"
)
sts_unit_label.pack(side="left", padx=(6,0), pady=(14,0))

sts_bar_frame = ctk.CTkFrame(sts_card, fg_color="transparent")
sts_bar_frame.pack(fill="x", padx=14, pady=(0,12))
sts_bar_canvas = Canvas(
    sts_bar_frame, width=260, height=12,
    bg=CLR_STS_BG, highlightthickness=0
)
sts_bar_canvas.pack(anchor="w")
sts_bar_canvas.create_rectangle(0, 0, 260, 12, fill="#F5E8DC", outline="")

wc_frame = ctk.CTkFrame(
    right_panel,
    fg_color="#F8F9FB",
    corner_radius=8,
    border_width=1,
    border_color=CLR_BORDER
)
wc_frame.pack(fill="x", padx=14, pady=(0,8))

wc_label = ctk.CTkLabel(
    wc_frame, text="",
    font=ctk.CTkFont(family="Arial", size=11),
    fg_color="transparent",
    anchor="w"
)
wc_label.pack(fill="x", padx=8, pady=8)

info_card = ctk.CTkFrame(
    right_panel,
    fg_color="#F0F4F8",
    corner_radius=8,
    border_width=1,
    border_color=CLR_BORDER
)
info_card.pack(fill="x", padx=14, pady=(0,8))

info_lines = [
    ("Model",    "BO-XGBoost (Optuna TPE)"),
    ("CS  R²",   "0.9206  |  RMSE = 3.42 MPa"),
    ("STS R²",   "0.8928  |  RMSE = 0.54 MPa"),
    ("Training", "CS n=344  /  STS n=192"),
    ("CV",       "5-seed × 10-fold (STS)"),
]
for i, (k, v) in enumerate(info_lines):
    row_f = ctk.CTkFrame(info_card, fg_color="transparent")
    row_f.pack(fill="x", padx=10, pady=(6 if i==0 else 2, 6 if i==len(info_lines)-1 else 0))
    ctk.CTkLabel(
        row_f, text=f"{k}:",
        font=ctk.CTkFont(family="Arial", size=10, weight="bold"),
        text_color=CLR_TEXT_LIGHT, fg_color="transparent", width=60, anchor="w"
    ).pack(side="left")
    ctk.CTkLabel(
        row_f, text=v,
        font=ctk.CTkFont(family="Arial", size=10),
        text_color=CLR_TEXT_MID, fg_color="transparent", anchor="w"
    ).pack(side="left")

footer = ctk.CTkFrame(app, fg_color="#E8EEF5", corner_radius=0, height=28)
footer.pack(fill="x", side="bottom")
footer.pack_propagate(False)
ctk.CTkLabel(
    footer,
    text="For research reference only  ·  Results valid within training data range  ·  Cement 258–651 kg/m³  ·  Vf 0–3.0%",
    font=ctk.CTkFont(family="Arial", size=9),
    text_color=CLR_TEXT_LIGHT,
    fg_color="transparent"
).pack(expand=True)

all_entries = [entry_water, entry_cement, entry_sand, entry_ca,
               entry_smax, entry_sp, entry_vf, entry_df, entry_lf]

app.mainloop()
