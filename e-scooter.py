import streamlit as st
from temp import function_from_script1
from future_forecasting import function_from_script2

def main():
    st.title('Streamlit App Running Multiple Scripts')

    # Option to choose which script to run
    script_choice = st.radio('Select the script to run:', ['Script 1', 'Script 2'])

    if script_choice == 'temp':
        st.subheader('Running temp')
        result1 = function_from_script1()
        st.write(result1)

    elif script_choice == 'future_forecasting':
        st.subheader('Running future_forecasting')
        result2 = function_from_script2()
        st.write(result2)

if __name__ == '__main__':
    main()
