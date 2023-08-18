import numpy as np
import pandas as pd
import sklearn
import joblib
from tensorflow import keras
import streamlit as st
from streamlit.logger import get_logger


def main_predict(txtfile, model, threshold):
    txtfile = txtfile[[txtfile.columns[1]]]
    txtfile = np.asarray(txtfile).astype(np.float32).T
    txtfile = txtfile.reshape(-1, 30)
    code2rel = {0: 'Tidak berhasil', 1: 'Berhasil'}
    
    proba = model.predict(txtfile)
    predict = 1 if proba > threshold else 0
    #print(f"{code2rel[predict]}, dengan akurasi {str(proba)[2:-2]}")
    output_txt = code2rel[predict]
    return output_txt, proba

model = keras.models.load_model('app/model/')
threshold = joblib.load('app/model/acc_threshold.pkl')




def run():
    st.set_page_config(
        page_title="Live-birth Occurence Prediction"
    )

    st.title("A Simple Web Application for Predicting Live-birth Occurence")
    st.write("Some text goes here")

    expander = st.expander("How to Use")
    expander.write("""
      Please fill in the following forms based on current medical status. Unknown information can be filled in by 0. Press the "predict" button to show the prediction result. """)

    col1, col2, col3, col4 = st.columns(4)
    col1.subheader("   General")
    with col1:
        
        age_at_treatment_status = st.select_slider('Select age range', options=['18-34', '35-37', '38-39', '40-42', '43-44', '45-50'])
        if age_at_treatment_status=="18-34":
            age_at_treatment=0
        elif age_at_treatment_status=="35-37":
            age_at_treatment=1
        elif age_at_treatment_status=="38-39":
            age_at_treatment=2
        elif age_at_treatment_status=="40-42":
            age_at_treatment=3
        elif age_at_treatment_status=="43-44":
            age_at_treatment=4
        else:
            age_at_treatment=5

        stimulation_used_status = st.checkbox( "Stimulation used ")
        fresh_cycle_status  = st.checkbox("Fresh Cycle" )
        frozen_cycle_status  = st.checkbox("Frozen Cycle" )
        egg_source_status = st.radio("Egg Source" , ('Patient', 'Donor'))
        sprem_from_status  = st.radio("Sperm From" ,('Patient', 'Donor'))

        stimulation_used=1 if stimulation_used_status else 0
        fresh_cycle=1 if fresh_cycle_status else 0
        frozen_cycle=1 if frozen_cycle_status else 0
        egg_source=1 if egg_source_status=='Patient' else 0
        sprem_from=1 if sprem_from_status=='Patient' else 0
    
    col2.subheader("Information   ")
    with col2:
        both_ivf_di = st.number_input( "Total Number of Previous cycles, Both IVF and DI")
        number_of_ivf_pregnancies = st.number_input( "Total number of IVF pregnancies ")
        live_birth_ivf = st.number_input( "Total number of live births - conceived through IVF ")  
        eggs_mixed_sperm  = st.number_input("Eggs Mixed With Partner Sperm" )
        eggs_thawed  = st.number_input("Eggs Thawed" )
        embryos_transfer  = st.number_input("Embryos Transfered" )
        eggs_collected  = st.number_input("Fresh Eggs Collected" )


    col3.subheader("Type of Infertility")
    with col3:
        female_primary_status = st.checkbox( "Female Primary" )
        female_secondary_status = st.checkbox( "Female Secondary ")
        male_primary_status = st.checkbox( "Male Primary" )
        male_secondary_status = st.checkbox( "Male Secondary" )
        couple_primary_status = st.checkbox( "Couple Primary" )
        couple_secondary_status = st.checkbox( "Couple Secondary" )

        couple_secondary=1 if couple_secondary_status else 0
        female_primary=1 if female_primary_status else 0
        female_secondary=1 if female_secondary_status else 0
        male_primary=1 if male_primary_status else 0
        male_secondary=1 if male_secondary_status else 0
        couple_primary=1 if couple_primary_status else 0
 
    col4.subheader("Cause of Infertility")
    with col4:    
        tubal_disease_status = st.checkbox( "Tubal disease" )
        ovulatory_disorder_status = st.checkbox( "Ovulatory Disorder" )
        male_factor_status = st.checkbox( "Male Factor" )
        patient_unexplained_status = st.checkbox( "Patient Unexplained" )
        endometriosis_status = st.checkbox( "Endometriosis ")
        cervical_factors_status = st.checkbox( "Cervical factors ")
        female_factors_status = st.checkbox( "Female Factors" )
        partner_sperm_concentration_status = st.checkbox( "Partner Sperm Concentration" )
        partner_sperm_morphology_status = st.checkbox( "Partner Sperm Morphology" )
        partner_sperm_motility_status = st.checkbox( " Partner Sperm Motility ")
        partner_sperm_immunological_status = st.checkbox("Partner Sperm Immunological factors" )

        tubal_disease=1 if tubal_disease_status else 0
        ovulatory_disorder=1 if ovulatory_disorder_status else 0
        male_factor=1 if male_factor_status else 0
        patient_unexplained=1 if patient_unexplained_status else 0
        endometriosis=1 if endometriosis_status else 0
        cervical_factors=1 if cervical_factors_status else 0
        female_factors=1 if female_factors_status else 0
        partner_sperm_concentration=1 if partner_sperm_concentration_status else 0
        partner_sperm_morphology=1 if partner_sperm_morphology_status else 0
        partner_sperm_motility=1 if partner_sperm_motility_status else 0
        partner_sperm_immunological=1 if partner_sperm_immunological_status else 0
        
     
    val_txtfile =[age_at_treatment, both_ivf_di, number_of_ivf_pregnancies, live_birth_ivf, female_primary, female_secondary, male_primary, male_secondary, couple_primary, couple_secondary, tubal_disease, ovulatory_disorder, male_factor,  patient_unexplained, endometriosis, cervical_factors,female_factors, partner_sperm_concentration, partner_sperm_morphology, partner_sperm_motility,   partner_sperm_immunological,stimulation_used, egg_source, sprem_from, fresh_cycle, frozen_cycle, eggs_mixed_sperm, eggs_thawed, embryos_transfer, eggs_collected]
    name_txtfile =["Patient Age at Treatment","Total Number of Previous cycles, Both IVF and DI", "Total number of IVF pregnancies","Total number of live births - conceived through IVF","Type of Infertility - Female Primary",
    "Type of Infertility - Female Secondary", "Type of Infertility - Male Primary","Type of Infertility - Male Secondary","Type of Infertility -Couple Primary","Type of Infertility -Couple Secondary",
    "Cause  of Infertility - Tubal disease", "Cause of Infertility - Ovulatory Disorder","Cause of Infertility - Male Factor","Cause of Infertility - Patient Unexplained","Cause of Infertility - Endometriosis",
    "Cause of Infertility - Cervical factors","Cause of Infertility - Female Factors","Cause of Infertility - Partner Sperm Concentration","Cause of Infertility -  Partner Sperm Morphology","Causes of Infertility - Partner Sperm Motility","Cause of Infertility -  Partner Sperm Immunological factors",
    "Stimulation used","Egg Source","Sperm From","Fresh Cycle","Frozen Cycle","Eggs Mixed With Partner Sperm","Eggs Thawed","Embryos Transfered","Fresh Eggs Collected"]

    
    

    if st.button('Predict'):
        txtf = {"14":name_txtfile,"unnamed":val_txtfile, }
        txtfile = pd.DataFrame(txtf)
        txtfile.to_csv('app/txtfile.txt')
        output_txt, proba = main_predict(txtfile, model, threshold)
        st.metric(label="Live-birth Occurence Expectancy", value=proba, delta=output_txt, delta_color="inverse")


if __name__ == "__main__":
    run()