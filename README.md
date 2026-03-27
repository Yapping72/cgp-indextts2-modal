# Steps
1. Logout of modal.com
2. ``modal setup`` 
    * Login using desired account
3. Create new secret in Modal: 
    * HF_TOKEN = see .env file
4. ``cat ~/.modal.toml``
    * Retrieve MODAL_TOKEN_ID,  
5. ``modal deploy indextts2_modal.py``