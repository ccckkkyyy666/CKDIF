## Getting Started

Follow these steps to get started with the CKDIF:

### Prerequisites
- Python 3.10

### Installation
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Install FinRL for Windows:
   ```bash
    git clone https://github.com/AI4Finance-Foundation/FinRL.git
    cd FinRL
    pip install .
    ```

### Running the Test
1. Open the `MOE/CIA.py` script and locate the following lines:
    ```python
   TRADE_START_DATE = 'YOUR_TRADE_START_DATE'
   TRADE_END_DATE = 'YOUR_TRADE_END_DATE'
   DATA_NAME = 'YOUR_TEST_DATA_FILE_NAME'
   ```
   Replace the placeholders with the actual information of your test data file.

2. Run the test using the following command:
    ```bash
    python CIA.py
    ```