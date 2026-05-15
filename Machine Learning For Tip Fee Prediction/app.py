import streamlit as st
import pandas as pd
from joblib import load

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title='Cek Ongkos Kirim',
    layout="centered"
)

# =========================================================
# HEADER
# =========================================================
html_temp = """
<div style="background-color:yellow;padding:20px;border-radius:10px">
    <h1 style="color:black;text-align:center;">
        Cek Harga Ongkos Pengiriman
    </h1>
</div>
"""

st.markdown(html_temp, unsafe_allow_html=True)

st.info(
    "⚠️ This is the beta version of Logee Trans Trip Fee Prediction App!"
)

# =========================================================
# LOAD MODEL & ENCODER
# =========================================================
@st.cache_resource
def load_artifacts():
    model = load('cb_tuned.z')
    encoder_item = load('encoder1.z')
    encoder_vehicle = load('encoder2.z')

    return model, encoder_item, encoder_vehicle


try:
    catboost, encoder_item, encoder_vehicle = load_artifacts()

except Exception as e:
    st.error(f"Gagal load model/encoder: {e}")
    st.stop()

# =========================================================
# MASTER DATA
# =========================================================
vehicle_group = (
    'Blind Van',
    'CDD Bak',
    'CDD Box',
    'CDD Chiller',
    'CDD Long Bak',
    'CDD Long Box',
    'CDD Los Bak',
    'CDD Wingbox',
    'CDE Bak',
    'CDE Box',
    'Fuso Bak',
    'Fuso Box',
    'Fuso Jumbo',
    'Fuso Tronton',
    'High Bed',
    'Low Bed',
    'Pick Up Bak',
    'Pick Up Box',
    'Self Loader',
    'Trailer Chiller Container 40 Feet',
    'Trailer Dry Container 20 Feet',
    'Trailer Dry Container 40 Feet',
    'Tronton Bak',
    'Tronton Box',
    'Tronton Dump Truck',
    'Tronton Wingbox'
)

item_package_group = (
    'Bags',
    'Cartons',
    'CBM',
    'CBN',
    'Drum',
    'Palet'
)

# =========================================================
# USER INPUT
# =========================================================
distance = st.number_input(
    'Jarak (km)',
    min_value=0.0,
    max_value=5000.0,
    value=0.0
)

itemWeight = st.number_input(
    'Berat Barang (ton)',
    min_value=0.0,
    max_value=100.0,
    value=0.0
)

itemPackage = st.selectbox(
    "Jenis Barang",
    item_package_group
)

vehicleGroupName = st.selectbox(
    "Jenis Kendaraan",
    vehicle_group
)

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button('Check'):

    # =====================================================
    # VALIDATION
    # =====================================================
    if distance <= 0:
        st.warning("Jarak harus lebih besar dari 0.")
        st.stop()

    if itemWeight <= 0:
        st.warning("Berat barang harus lebih besar dari 0.")
        st.stop()

    try:

        # =================================================
        # RAW INPUT DATAFRAME
        # =================================================
        df = pd.DataFrame([{
            'itemWeight': itemWeight,
            'distance': distance,
            'vehicleGroupName': vehicleGroupName,
            'itemPackage': itemPackage
        }])

        # =================================================
        # VEHICLE ENCODING
        # =================================================
        vehicle_encoded = encoder_vehicle.transform(
            df[['vehicleGroupName']]
        )

        vehicle_encoded_df = pd.DataFrame(
            vehicle_encoded,
            columns=encoder_vehicle.get_feature_names_out(
                ['vehicleGroupName']
            ),
            index=df.index
        )

        # =================================================
        # ITEM PACKAGE ENCODING
        # =================================================
        item_encoded = encoder_item.transform(
            df[['itemPackage']]
        )

        item_encoded_df = pd.DataFrame(
            item_encoded,
            columns=encoder_item.get_feature_names_out(
                ['itemPackage']
            ),
            index=df.index
        )

        # =================================================
        # MERGE ENCODED FEATURES
        # =================================================
        df = pd.concat([
            df,
            vehicle_encoded_df,
            item_encoded_df
        ], axis=1)

        # =================================================
        # DROP ORIGINAL CATEGORICAL COLUMNS
        # =================================================
        df = df.drop(
            ['vehicleGroupName', 'itemPackage'],
            axis=1
        )

        # =================================================
        # HANDLE MISSING FEATURES
        # =================================================
        model_features = catboost.feature_names_

        for col in model_features:
            if col not in df.columns:
                df[col] = 0

        # =================================================
        # REORDER COLUMNS
        # =================================================
        df = df[model_features]

        # =================================================
        # PREDICTION
        # =================================================
        prediction = catboost.predict(df)[0]

        # =================================================
        # FORMAT RESULT
        # =================================================
        harga = f"Rp. {int(prediction):,}".replace(',', '.')

        st.markdown(f"## 💰 {harga}")

        st.success("Prediksi berhasil!")

        # OPTIONAL DEBUG
        with st.expander("Lihat Feature Input"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")