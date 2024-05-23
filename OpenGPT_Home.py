import streamlit as st

st.set_page_config(page_title="OpenGPT Prototype", page_icon="📎")
st.header("👋 Welcome to OpenGPT Home!")

st.markdown(
    """
    歡迎來到 OpenGPT! 此產品目前正處於概念性產品階段，我們需要您的參與來幫助它更加完善，此產品提供一般對話、網頁對話與文件對話等多種情境，旨在滿足您的不同需求。
    
    此外，此服務採用本地端部署的模型並架設於公司內部主機，幾乎無額外的費用產生，也不用擔心資料外流的風險。
    
    我們團隊一直在努力並嘗試，提供優質、便利的服務與持續改進，讓更多人能夠接受。我們期待您寶貴的意見，這些反饋將直接影響我們的發展。
    
    Welcome to OpenGPT! This product is currently in the conceptual stage, and we need your participation to help improve it. We offer various scenarios such as general conversation, web-based conversation, and document conversation, aiming to meet your different needs.

    Additionally, this service utilizes on-premises models and is hosted on the company's internal servers, resulting in minimal additional costs and eliminating concerns about data leakage risks.

    Our team is dedicated to providing high-quality, convenient services and continuously improving to make it more accessible. We look forward to your valuable feedback, as it will directly influence our development.

    ---
    #### Function Description
    
    ##### General Chat
    * 這是個一般性對話的窗口，您可以在這自由地與 AI 暢談。
    * This is a general conversation window where you can freely converse with AI.

    ##### URL Chat
    * 這是個網頁對話的窗口，提供網頁連結(URL)，即可進行網頁摘要或您的 AI 對話。
    * This is a web chat window where you simply need to provide the webpage link (URL) for the assistant to start the conversation. 

    ##### Document Chat
    * 這是個檔案對話的窗口，上傳您的文件 (word、pdf、ppt、csv、xlsx) 與 AI 開啟一連串檔案內容的談話。
    * 或者，您也能在對話中添加檔案的名稱，指定 AI 與您對話的文本內容。
    * This is a document chat window. Please upload your document(word, pdf, ppt, csv, xlsx) to begin a series of content conversation.
    * Additionally, you can add the file names in the conversation to specify the text content for the AI to discuss with you.
    
    #### 
    """
)
