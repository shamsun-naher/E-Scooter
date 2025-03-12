import streamlit as st
from script1 import function_from_script1
from script2 import function_from_script2

def main():
    st.title('Streamlit App Running Multiple Scripts')

    # Option to choose which script to run
    script_choice = st.radio('Select the script to run:', ['Script 1', 'Script 2'])

    if script_choice == 'Script 1':
        st.subheader('Running Script 1')
        result1 = function_from_script1()
        st.write(result1)

    elif script_choice == 'Script 2':
        st.subheader('Running Script 2')
        result2 = function_from_script2()
        st.write(result2)

if __name__ == '__main__':
    main()
