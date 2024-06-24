import streamlit as st

st.set_page_config(page_title="Acknowledgements",layout="wide")
## Add background image
st.markdown(
    """
    <div class="banner">
        <img src="https://raw.githubusercontent.com/AshkanZand-TUM/test-hbn/7d6aae582cb97a35b8ba774b8efdc5b48bf4cf41/icon/banner_file_size_3.svg" alt="Banner Image">
    </div>
    <style>
        .banner {
            width: 100%;
            height: 200px;
            overflow: hidden;
        }
        .banner img {
            width: 100%;
            object-fit: cover;
        }
    </style>
    """,  unsafe_allow_html=True,
)

css = '''
<style>
    [data-testid="stSidebar"]{
        min-width: 230px;
        max-width: 230px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

with st.container(border=True):
    colp11, colp1,colp2,colp21, colp3,colp4,colp5,colp6 = st.columns(8, gap="small")
    with colp11:
        st.page_link("DefectDashboard.py", label="Main database")
    with colp1:
        st.page_link("pages/1_DFT calculation details.py", label="DFT calculation details")
    with colp2: 
        st.page_link("pages/2_About.py", label="About")
    with colp21:
        st.page_link("pages/3_Request defect.py", label="Request data")
    with colp3:
        st.page_link("pages/4_Contact.py", label="Contact")
    with colp4:
        st.page_link("pages/5_Acknowledgements.py", label="Acknowledgements")
    with colp5:
        st.page_link("pages/6_Imprint.py", label="Impressum")
    with colp6:
        st.page_link("pages/7_Version.py", label="Version")


st.title("Acknowledgements")

st.subheader("")
with st.container(border=False):
  st.markdown('''
    This project is part of the [Munich Quantum Valley](https://www.munich-quantum-valley.de/) initiative, which is supported by 
    the Bavarian State Government with funds from the High-Tech Agenda Bayern Plus. \n

    The work is funded by the Federal Ministry of Education and Research (BMBF) under Grant No. 13N16292 \
    and is supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under \
    Germany's Excellence Strategy - [EXC-2111 - 390814868](https://gepris.dfg.de/gepris/projekt/390814868)
  ''')

