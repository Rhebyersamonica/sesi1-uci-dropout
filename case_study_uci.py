"""
Studi Kasus Sesi 1: Prediksi Mahasiswa Berisiko Drop Out (Data Publik UCI)
Berdasarkan webinar "Make Sense of Data with Analysis and AI" (Andria Arisal, BRIN).

Data   : UCI "Predict Students' Dropout and Academic Success" (CC BY 4.0)
         https://archive.ics.uci.edu/dataset/697  -> simpan data.csv di folder data/
Alur   : CRISP-DM + 4 level analytics
         1) Descriptive  2) Diagnostic  3) Predictive  4) Prescriptive

Konsep materi yang diterapkan:
  - Pertanyaan bisnis -> pertanyaan analitik
  - Label hanya dipakai jika hasil akhirnya sudah teramati (status 'Enrolled' dikeluarkan)
  - Data leakage / point-in-time: fitur dibatasi sesuai KAPAN prediksi dibuat
    (saat daftar, akhir semester 1, akhir semester 2)
  - Data tidak seimbang: baseline akurasi tinggi tetapi recall 0 -> pakai recall/F1/ROC-AUC
  - Keputusan akhir tetap di manusia (model hanya alat bantu)

Jalankan:  python case_study_uci.py
Opsi    :  --data data/data.csv  --drop-sensitive  --include-enrolled  --recall-target 0.8
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUT, SEED = "output", 42
URL = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
# kolom kode kategori (angka hanya label, bukan ukuran)
CAT = ["marital_status", "application_mode", "course", "previous_qualification", "nacionality",
       "mother_s_qualification", "father_s_qualification", "mother_s_occupation", "father_s_occupation"]
# fitur sensitif: pertimbangkan dibuang agar keputusan tidak diskriminatif
SENSITIVE = ["gender", "marital_status", "nacionality", "international",
             "educational_special_needs", "displaced"]
ENROLL = ["marital_status", "application_mode", "application_order", "course",
          "daytime_evening_attendance", "previous_qualification", "previous_qualification_grade",
          "nacionality", "mother_s_qualification", "father_s_qualification", "mother_s_occupation",
          "father_s_occupation", "admission_grade", "displaced", "educational_special_needs", "debtor",
          "tuition_fees_up_to_date", "gender", "scholarship_holder", "age_at_enrollment", "international",
          "unemployment_rate", "inflation_rate", "gdp"]


# ----------------------------------------------------------------- 1. DATA UNDERSTANDING
def load_data(path):
    if not os.path.exists(path):
        sys.exit(f"File '{path}' tidak ditemukan.\nUnduh dataset dari {URL}\n"
                 "lalu ekstrak zip-nya dan simpan data.csv di folder data/.")
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:                                   # jaga-jaga jika pemisahnya koma
        df = pd.read_csv(path, sep=",")
    # samakan nama kolom: huruf kecil, spasi/simbol -> underscore
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    if "target" not in df.columns:
        sys.exit(f"Kolom 'target' tidak ditemukan. Kolom yang ada: {list(df.columns)}")
    df["target"] = df["target"].astype(str).str.strip()
    return df


def understand(df):
    print(f"Ukuran data: {df.shape[0]} baris x {df.shape[1]} kolom")
    print("Missing value total:", int(df.isna().sum().sum()), "| Baris duplikat:", int(df.duplicated().sum()))
    dist = df.target.value_counts()
    print("\nDistribusi target:\n", pd.DataFrame({"jumlah": dist, "proporsi": (dist / len(df)).round(3)}))
    ax = dist.plot(kind="bar", color=["tab:green", "tab:red", "tab:blue"][:len(dist)], figsize=(5, 3.5))
    ax.set_title("Distribusi status mahasiswa"); ax.set_ylabel("Jumlah"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{OUT}/01_distribusi_target.png", dpi=130); plt.close()


def prepare(df, include_enrolled):
    """Label hanya valid jika hasil akhirnya sudah teramati. 'Enrolled' = belum tahu -> dikeluarkan."""
    d = df.copy()
    if not include_enrolled:
        n = (d.target == "Enrolled").sum()
        d = d[d.target != "Enrolled"].reset_index(drop=True)
        print(f"\n{n} mahasiswa berstatus 'Enrolled' dikeluarkan (hasil akhirnya belum diketahui).")
    y = (d.target == "Dropout").astype(int)
    print(f"Data modeling: {len(d)} mahasiswa | proporsi dropout = {y.mean():.1%}")
    return d, y


# ----------------------------------------------------------------- 2. DESCRIPTIVE
def descriptive(d, y):
    d = d.assign(dropout=y)
    d["kelompok_umur"] = pd.cut(d.get("age_at_enrollment", pd.Series(20, index=d.index)),
                                [0, 19, 22, 29, 100], labels=["<=19", "20-22", "23-29", ">=30"])
    cols = [c for c in ["tuition_fees_up_to_date", "debtor", "scholarship_holder", "gender",
                        "daytime_evening_attendance", "kelompok_umur"] if c in d.columns]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        r = d.groupby(c, observed=True).dropout.mean() * 100
        r.index = r.index.astype(str); r.plot(kind="bar", ax=ax, color="tab:red")
        ax.set_title(f"Tingkat dropout per {c} (%)", fontsize=9); ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=0)
    for ax in axes.ravel()[len(cols):]: ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/02_dropout_per_kelompok.png", dpi=130); plt.close()
    print("\n[Descriptive] grafik tingkat dropout per kelompok disimpan.")


# ----------------------------------------------------------------- 3. DIAGNOSTIC
def diagnostic(d, y):
    cols = [c for c in ["admission_grade", "age_at_enrollment",
                        "curricular_units_1st_sem_approved", "curricular_units_1st_sem_grade",
                        "curricular_units_2nd_sem_approved", "curricular_units_2nd_sem_grade"] if c in d.columns]
    t = d[cols].groupby(y.map({0: "Graduate", 1: "Dropout"})).mean().T
    t["selisih_%"] = (t["Dropout"] / t["Graduate"] - 1) * 100
    t.round(2).to_csv(f"{OUT}/diagnostik_rata_rata.csv")
    print("\n[Diagnostic] rata-rata fitur: Dropout vs Graduate\n", t.round(2))
    return t


# ----------------------------------------------------------------- 4. PREDICTIVE
def feature_sets(d, drop_sensitive):
    enroll = [c for c in ENROLL if c in d.columns]
    s1 = [c for c in d.columns if c.startswith("curricular_units_1st_sem")]
    s2 = [c for c in d.columns if c.startswith("curricular_units_2nd_sem")]
    sets = {"A. Saat pendaftaran": enroll,
            "B. Akhir semester 1": enroll + s1,
            "C. Akhir semester 2": enroll + s1 + s2}
    if drop_sensitive:
        sets = {k: [c for c in v if c not in SENSITIVE] for k, v in sets.items()}
    return sets


def make_prep(cols):
    cat = [c for c in cols if c in CAT]
    num = [c for c in cols if c not in CAT]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat)])


def make_models():
    return {
        "Baseline (tebak mayoritas)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
    }


def predictive(d, y, sets):
    tr, te = train_test_split(d.index, test_size=0.25, stratify=y, random_state=SEED)
    rows = []
    for sname, cols in sets.items():
        for mname, clf in make_models().items():
            pipe = Pipeline([("prep", make_prep(cols)), ("clf", clf)]).fit(d.loc[tr, cols], y[tr])
            proba = pipe.predict_proba(d.loc[te, cols])[:, 1]
            pred = pipe.predict(d.loc[te, cols])
            rows.append({"skenario": sname, "model": mname,
                         "accuracy": (pred == y[te]).mean(),
                         "precision": precision_score(y[te], pred, zero_division=0),
                         "recall": recall_score(y[te], pred),
                         "f1": f1_score(y[te], pred),
                         "roc_auc": roc_auc_score(y[te], proba)})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(f"{OUT}/perbandingan_model.csv", index=False)
    print("\n[Predictive] perbandingan model per skenario waktu prediksi\n", res.to_string(index=False))

    piv = res[res.model != "Baseline (tebak mayoritas)"].pivot(index="skenario", columns="model", values="recall")
    piv.plot(kind="bar", figsize=(8, 4)); plt.ylabel("Recall dropout"); plt.xticks(rotation=0)
    plt.title("Semakin akhir waktu prediksi, semakin banyak dropout terdeteksi"); plt.tight_layout()
    plt.savefig(f"{OUT}/03_recall_per_skenario.png", dpi=130); plt.close()
    return tr, te, res


def importance(d, y, cols, tr):
    pipe = Pipeline([("prep", make_prep(cols)),
                     ("clf", RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                    random_state=SEED, n_jobs=-1))]).fit(d.loc[tr, cols], y[tr])
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names).sort_values().tail(12)
    imp.index = [n.split("__", 1)[-1] for n in imp.index]
    imp.plot(kind="barh", figsize=(7, 4.5), color="tab:blue"); plt.title("12 fitur terpenting (Random Forest, skenario B)")
    plt.tight_layout(); plt.savefig(f"{OUT}/04_fitur_penting.png", dpi=130); plt.close()
    print("\n12 fitur terpenting:\n", imp.iloc[::-1].round(3).to_string())


# ----------------------------------------------------------------- 5. PRESCRIPTIVE
def prescriptive(d, y, cols, tr, te, recall_target):
    """Skor risiko -> tingkat risiko -> tindakan. Threshold dipilih dari data LATIH (cross-validation),
    target recall ditetapkan MANUSIA (pengelola prodi), bukan AI."""
    pipe = Pipeline([("prep", make_prep(cols)),
                     ("clf", RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                    random_state=SEED, n_jobs=-1))])
    cvp = cross_val_predict(pipe, d.loc[tr, cols], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y[tr], cvp)
    ok = np.where(rec[:-1] >= recall_target)[0]
    t = thr[ok].max() if len(ok) else thr.min()
    print(f"\n[Prescriptive] target recall = {recall_target:.0%} -> threshold = {t:.2f} (dipilih dari data latih)")

    pipe.fit(d.loc[tr, cols], y[tr])
    p = pipe.predict_proba(d.loc[te, cols])[:, 1]
    flag = p >= t
    print(f"Pada data uji: recall={recall_score(y[te], flag):.2f}, precision={precision_score(y[te], flag):.2f}, "
          f"mahasiswa ditandai={flag.mean():.0%}")
    print("Confusion matrix (baris=aktual, kolom=prediksi):\n", confusion_matrix(y[te], flag))

    tier = np.where(p >= t, "Tinggi", np.where(p >= t / 2, "Sedang", "Rendah"))
    action = {"Tinggi": "Konseling akademik + pendampingan dosen wali + cek kebutuhan bantuan biaya",
              "Sedang": "Pantau capaian SKS dan kehadiran, ingatkan layanan bimbingan",
              "Rendah": "Pemantauan rutin"}
    out = pd.DataFrame({"indeks_data": d.loc[te].index, "p_dropout": p.round(3), "tingkat_risiko": tier,
                        "dropout_aktual": y[te].values})
    out["rekomendasi"] = out.tingkat_risiko.map(action)
    out.sort_values("p_dropout", ascending=False).to_csv(f"{OUT}/daftar_risiko_uji.csv", index=False)
    chk = out.groupby("tingkat_risiko").dropout_aktual.agg(["size", "mean"]).reindex(["Tinggi", "Sedang", "Rendah"])
    chk["mean"] = (chk["mean"] * 100).round(1)
    print("\nPemeriksaan: tingkat dropout aktual per tingkat risiko\n", chk.rename(columns={"size": "jumlah", "mean": "dropout_aktual_%"}))
    chk["mean"].plot(kind="bar", color=["tab:red", "tab:orange", "tab:green"], figsize=(5, 3.5))
    plt.ylabel("Dropout aktual (%)"); plt.xticks(rotation=0); plt.title("Tingkat risiko vs dropout aktual")
    plt.tight_layout(); plt.savefig(f"{OUT}/05_tingkat_risiko.png", dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--include-enrolled", action="store_true", help="ikutkan 'Enrolled' sebagai bukan-dropout")
    ap.add_argument("--drop-sensitive", action="store_true", help="buang fitur sensitif (gender, kebangsaan, dll.)")
    ap.add_argument("--recall-target", type=float, default=0.80)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 170, "display.max_columns", 20)

    df = load_data(a.data)
    print("== 1) DATA UNDERSTANDING =="); understand(df)
    d, y = prepare(df, a.include_enrolled)
    print("\n== 2) DESCRIPTIVE =="); descriptive(d, y)
    print("\n== 3) DIAGNOSTIC =="); diagnostic(d, y)
    sets = feature_sets(d, a.drop_sensitive)
    print("\n== 4) PREDICTIVE ==")
    tr, te, res = predictive(d, y, sets)
    importance(d, y, sets["B. Akhir semester 1"], tr)
    print("\n== 5) PRESCRIPTIVE (skenario B: akhir semester 1) ==")
    prescriptive(d, y, sets["B. Akhir semester 1"], tr, te, a.recall_target)
    
