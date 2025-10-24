# gridirony
fantasy football machine  
gridirony is my first attempt at predictive modeling, taking various historical stats and trends and using them for current football matchups in an attempt to have some degree of accuracy in predicting patterns.

## Environment Setup

### Python
Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1

pip install -r requirements.txt

Rscript install.R

Rscript src\r\train_logit.R
Rscript src\r\train_linear.R
