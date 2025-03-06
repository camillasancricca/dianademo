## Welcome to the DIANA tool!

**DIANA** is a **data-centric AI-based tool** able to support and guide users in selecting and validating the data preparation tasks to perform in a data analysis pipeline.
DIANA adopts a human-centered approach: involving the users in all stages, supporting their decisions, and leaving them in control of the process.

DIANA's main functionalities are:
1) **exploration**, **profiling**, and **data quality assessment** functionalities to make the users aware of the characteristics and anomalies of the data;
2) **recommendations** on the best sequence of data preparation actions that best-fit users’ analysis purposes;
3) **explainability** to enable also non-expert practitioners to be involved in the pipeline phases;
4) **sliding autonomy** that is the system’s ability to incorporate human intervention when needed, e.g., increasing/decreasing the system support based on the user needs, skills and expertise.

---

## Installation & Usage

### Prerequisites
Ensure you have the following installed on your system:
- python 3.11 or higher
- pip (python package manager)
- an **openai key** must be added as OPENAI_API_KEY in: 
  ```diana-demo/explainability/explanations.py``` (to generate the explanations)
- it is suggested to use the datasets provided with the demo (```diana-demo/dataset/iris_dirty.csv, diana-demo/dataset/beers_dirty.csv```)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/dianademo.git
   cd dianademo
   ```
2. Install the requiements:
   ```bash
   pip install -r requirements.txt
   ```
3.  Run the tool:
   ```bash
   streamlit run app.py
   ```
