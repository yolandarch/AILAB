import streamlit as st

st.title("hai")
st.text("blabla")

nama = st.text_input("Masukan nama")
nim = st.text_input("Masukan nim")
gender = st.radio("gender", ["Pria",'Wanita'])

if nama :
    st.text("nama : " + nama)
    if (len(nim)==10):
        st.text("nim : " + nim)
    st.text("gender :" + gender)

box = st.selectbox("Pilih matkul", ["Pengpro", "Agama", "Penkar"])

st.write(box)

umur = st.slider("Umur", 1, 70,10)

st.write(umur,"\n", gender)

if gender == "Pria":
    st.write("Halo pak" , nama)
else :
    st.write("Halo bu" , nama)
    

hobi = st.text_area("Hobi","Main ,makan")
hobi = [x.strip()for x in hobi.split(",")]

st.write(hobi)
st.image("https://www.georgetown.edu/wp-content/uploads/2023/11/DSC_7947-scaled.jpg", caption="panda", use_column_width= True, channels="RGB")

st.markdown("[jangan pencet](https://www.georgetown.edu/news/the-giant-pandas-have-left-the-national-zoo-whats-next-for-u-s-china-relations/)")

import pandas as pd

data = {"Pekerjaan" : ["Programmer", "Dokter", "Pengacara"],
        "Tier" : ["E","S","SS"]}

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

st.title("membuka data")
file = st.file_uploader("Pilih file png", type=["png", "csv", "jpg", "jpeg"])

if file is not None :
    st.write(file.type)
    if file.type == "image/jpg" or file.type == "image/png" or file.type == "image/jpeg":
        st.image(file)
    else :
        data : pd.read_csv(file)
        st.dataframe(data)

st.title("kalkulator")

angka1 = st.number_input("Maasukan angka 1", value = 0)
angka2 = st.number_input("Maasukan angka 2", value = 0)
operasi = st.radio("pilih", ["(+)" , "(-)", "(/)", "(*)"])
if st.button ("hitung") :
    if operasi=="(+)" :
        hasil = angka1 + angka2
    elif operasi=="(-)" :
        hasil = angka1 - angka2
    elif operasi=="(*)" :
        hasil = angka1 * angka2
    elif operasi=="(/)" :
        hasil = angka1/ angka2
        st.success(f"Hasil {operasi} : {hasil}")

st.sidebar.header("Fitur Kiri")
if st.sidebar.checkbox("Kalkulator") :
    st.sidebar.write(f"Kuadrat dari {angka1} : {angka1**2}")